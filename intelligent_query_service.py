from groq import Groq
from typing import List, Dict, Any
from local_vector_store import LocalVectorStore
from schema_analyzer import SchemaAnalyzer
from models import ChatResponse
import os
import json

class IntelligentQueryService:
    def __init__(self):
        self.vector_store = LocalVectorStore()
        try:
            from secure_config import SecureConfig
            config = SecureConfig().get_api_keys()
            groq_key = config.get("GROQ_API_KEY")
            if groq_key:
                self.groq_client = Groq(api_key=groq_key)
            else:
                self.groq_client = None
        except:
            self.groq_client = None
        self.schema_analyzer = SchemaAnalyzer()
    
    def process_query(self, question: str, user_id: str) -> ChatResponse:
        """Two-stage LLM processing: 1) Decide what to query, 2) Format results"""
        
        # Stage 1: Get file schemas and let LLM decide what to query
        schemas = self._get_user_schemas(user_id)
        if not schemas:
            return ChatResponse(
                answer="No data uploaded yet. Please upload your files first.",
                sources=[]
            )
        
        query_plan = self._create_query_plan(question, schemas)
        print(f"Query plan: {query_plan}")
        
        # Stage 2: Execute the query plan
        retrieved_data = self._execute_query_plan(query_plan, user_id)
        
        # Stage 3: Let LLM format the final response
        formatted_response = self._format_response(question, retrieved_data, query_plan)
        
        return ChatResponse(
            answer=formatted_response,
            sources=list(query_plan.get('target_files', []))
        )
    
    def _get_user_schemas(self, user_id: str) -> Dict:
        """Get schemas for user's uploaded files"""
        try:
            schema_file = "local_data/data_schemas.json"
            if os.path.exists(schema_file):
                with open(schema_file, 'r') as f:
                    all_schemas = json.load(f)
                return all_schemas.get(user_id, {})
        except Exception as e:
            print(f"Error loading schemas: {e}")
        return {}
    
    def _create_query_plan(self, question: str, schemas: Dict) -> Dict:
        """Stage 1: LLM decides which files/columns to query"""
        
        # Build schema summary for LLM
        schema_summary = "Available data files and their structures:\n\n"
        for file_name, schema in schemas.items():
            schema_summary += f"**{file_name}**:\n"
            schema_summary += f"- Type: {schema.get('data_type', 'unknown')}\n"
            schema_summary += f"- Description: {schema.get('description', 'No description')}\n"
            
            if schema.get('key_fields'):
                schema_summary += "- Key fields:\n"
                for field in schema['key_fields'][:5]:  # Show top 5 fields
                    field_name = field.get('field', 'unknown')
                    field_desc = field.get('description', 'No description')
                    sample_values = field.get('sample_values', [])
                    schema_summary += f"  • {field_name}: {field_desc}\n"
                    if sample_values:
                        schema_summary += f"    Sample values: {', '.join(sample_values[:3])}\n"
            schema_summary += "\n"
        
        prompt = f"""You are a data analyst. Based on the user's question and available data, create a query plan.

User Question: {question}

{schema_summary}

Create a JSON query plan with:
1. "target_files": List of files to search (based on data_type and relevance)
2. "target_fields": List of specific field names to extract
3. "search_strategy": "comprehensive" (get all data) or "specific" (targeted search)
4. "reasoning": Why you chose these files and fields

Return ONLY valid JSON:
{{
    "target_files": ["file1.csv", "file2.csv"],
    "target_fields": ["field1", "field2"],
    "search_strategy": "comprehensive",
    "reasoning": "explanation"
}}"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Return only valid JSON query plans."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in LLM response")
                
        except Exception as e:
            print(f"Query planning failed: {e}")
            # Fallback plan
            return {
                "target_files": list(schemas.keys()),
                "target_fields": ["name", "project_name", "title"],
                "search_strategy": "comprehensive",
                "reasoning": "Fallback: search all files for common project fields"
            }
    
    def _execute_query_plan(self, query_plan: Dict, user_id: str) -> List[Dict]:
        """Execute the query plan to retrieve relevant data from ALL target files"""
        
        target_files = query_plan.get('target_files', [])
        search_strategy = query_plan.get('search_strategy', 'comprehensive')
        
        # Smart data retrieval - get representative samples from each target file
        all_chunks = []
        
        if search_strategy == "comprehensive":
            # For each target file, get a good sample
            for file_name in target_files:
                file_chunks = self.vector_store.search(f"file:{file_name}", user_id, top_k=20)
                # Filter to ensure we get chunks from this specific file
                file_specific_chunks = [c for c in file_chunks if c.get('file_name', '') == file_name]
                all_chunks.extend(file_specific_chunks)
                print(f"Got {len(file_specific_chunks)} chunks from {file_name}")
        else:
            # Use targeted search
            search_query = f"data from files: {', '.join(target_files)}"
            all_chunks = self.vector_store.search(search_query, user_id, top_k=50)
        
        # If we didn't get enough data from file-specific searches, do a broader search
        if len(all_chunks) < 10 and search_strategy == "comprehensive":
            print("File-specific search yielded few results, doing broader search...")
            broader_chunks = self.vector_store.search("project name data", user_id, top_k=100)
            # Filter by target files
            if target_files:
                filtered_chunks = [
                    chunk for chunk in broader_chunks 
                    if chunk.get('file_name', '') in target_files
                ]
            else:
                filtered_chunks = broader_chunks
        else:
            # Filter by target files if specified
            if target_files:
                filtered_chunks = [
                    chunk for chunk in all_chunks 
                    if chunk.get('file_name', '') in target_files
                ]
            else:
                filtered_chunks = all_chunks
        
        # Debug: Show what we found from each file
        files_found = {}
        for chunk in filtered_chunks:
            file_name = chunk.get('file_name', 'Unknown')
            if file_name not in files_found:
                files_found[file_name] = 0
            files_found[file_name] += 1
        
        print(f"Retrieved {len(filtered_chunks)} chunks from files: {files_found}")
        
        return filtered_chunks
    
    def _format_response(self, question: str, data_chunks: List[Dict], query_plan: Dict) -> str:
        """Stage 2: LLM formats the retrieved data into final response"""
        
        if not data_chunks:
            return "No relevant data found for your query."
        
        # Sample data for LLM processing (to stay within token limits)
        sample_size = 25
        if len(data_chunks) > sample_size:
            # Sample evenly across files
            files = {}
            for chunk in data_chunks:
                file_name = chunk.get('file_name', 'Unknown')
                if file_name not in files:
                    files[file_name] = []
                files[file_name].append(chunk)
            
            sampled_chunks = []
            chunks_per_file = max(1, sample_size // len(files))
            for file_chunks in files.values():
                sampled_chunks.extend(file_chunks[:chunks_per_file])
        else:
            sampled_chunks = data_chunks
        
        # Build context for formatting
        context = ""
        for chunk in sampled_chunks:
            file_name = chunk.get('file_name', 'Unknown')
            content = chunk['content']
            context += f"From {file_name}:\n{content}\n\n"
        
        # Limit context size
        if len(context) > 8000:
            context = context[:8000] + "...\n[Content truncated]"
        
        prompt = f"""Based on the query plan and retrieved data, provide a well-formatted answer to the user's question.

User Question: {question}

Query Plan: {query_plan['reasoning']}
Target Fields: {', '.join(query_plan.get('target_fields', []))}

Retrieved Data:
{context}

Instructions:
1. Extract ALL unique values for the requested information from ALL files
2. Combine results from all files and remove duplicates
3. Create a single numbered list of unique items
4. Show the source file(s) for each item
5. Count the total unique items found
6. Use this format:

## All [Items] ([X] unique found):

1. **Item Name** _(from file1.csv, file2.csv)_
2. **Another Item** _(from file1.csv)_

Provide the final formatted response with ALL unique results combined:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a project management assistant. Format data responses clearly and professionally."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            formatted_answer = response.choices[0].message.content
            
            # Add metadata about the query
            if len(data_chunks) > len(sampled_chunks):
                formatted_answer += f"\n\n*Analyzed {len(sampled_chunks)} samples from {len(data_chunks)} total records across {len(query_plan.get('target_files', []))} files.*"
            
            return formatted_answer
            
        except Exception as e:
            print(f"Response formatting failed: {e}")
            return f"Retrieved {len(data_chunks)} records but failed to format the response. Error: {str(e)}"