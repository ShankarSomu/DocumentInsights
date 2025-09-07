from groq import Groq
from typing import List, Dict, Any
from local_vector_store import LocalVectorStore
from metadata_service import MetadataService
from models import ChatResponse
import os
import json
import pandas as pd
import io

class TwoStageQueryService:
    def __init__(self):
        self.vector_store = LocalVectorStore()
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.metadata_service = MetadataService()
    
    def process_query(self, question: str, user_id: str) -> ChatResponse:
        """Two-stage processing: 1) Find relevant columns, 2) Process and format data"""
        
        print(f"\n=== TWO-STAGE QUERY SERVICE ACTIVATED ===")
        print(f"Stage 1: Finding relevant columns for query: {question}")
        
        # Stage 1: Query metadata to find relevant columns and files
        relevant_columns = self._stage1_find_columns(question, user_id)
        
        if not relevant_columns:
            return ChatResponse(
                answer="No relevant data found. Please upload files first.",
                sources=[]
            )
        
        print(f"Stage 1 found: {len(relevant_columns)} relevant columns")
        for col in relevant_columns[:3]:  # Show first 3 columns
            print(f"  - {col.get('column_name', 'unknown')} from {col.get('file_name', 'unknown')}")
        
        # Stage 2: Get actual column data and process with LLM
        final_response = self._stage2_process_data(question, relevant_columns, user_id)
        
        return final_response
    
    def _stage1_find_columns(self, question: str, user_id: str) -> List[Dict]:
        """Stage 1: Use vector search on metadata to find relevant columns"""
        
        # Search metadata chunks for relevant columns
        metadata_chunks = self.vector_store.search(question, user_id, top_k=20)
        
        # Filter to only metadata chunks
        metadata_only = [
            chunk for chunk in metadata_chunks 
            if chunk.get('file_name', '').startswith('METADATA_')
        ]
        
        if not metadata_only:
            print("No metadata found, falling back to all files")
            # Fallback: get user's actual metadata
            user_metadata = self.metadata_service.get_user_metadata(user_id)
            relevant_columns = []
            for file_name, columns in user_metadata.items():
                relevant_columns.extend(columns)
            return relevant_columns
        
        # Parse metadata chunks to extract column information
        relevant_columns = []
        for chunk in metadata_only:
            try:
                content = chunk['content']
                # Extract file and column info from metadata content
                lines = content.split('\n')
                file_name = None
                column_name = None
                
                for line in lines:
                    if line.strip().startswith('File:'):
                        file_name = line.split('File:')[1].strip()
                    elif line.strip().startswith('Column:'):
                        column_name = line.split('Column:')[1].strip()
                
                if file_name and column_name:
                    # Get full metadata for this column
                    user_metadata = self.metadata_service.get_user_metadata(user_id)
                    if file_name in user_metadata:
                        for col_meta in user_metadata[file_name]:
                            if col_meta['column_name'] == column_name:
                                relevant_columns.append(col_meta)
                                break
                
            except Exception as e:
                print(f"Error parsing metadata chunk: {e}")
        
        return relevant_columns
    
    def _stage2_process_data(self, question: str, relevant_columns: List[Dict], user_id: str) -> ChatResponse:
        """Stage 2: Get actual column data and process with LLM"""
        
        print(f"\nStage 2: Processing data from {len(relevant_columns)} columns")
        
        # Group columns by file
        files_to_process = {}
        for col_meta in relevant_columns:
            file_name = col_meta['file_name']
            if file_name not in files_to_process:
                files_to_process[file_name] = []
            files_to_process[file_name].append(col_meta['column_name'])
        
        # Get actual data from relevant columns
        column_data = self._extract_column_data(files_to_process, user_id)
        
        # Process with LLM
        formatted_response = self._format_with_llm(question, column_data, relevant_columns)
        
        return ChatResponse(
            answer=formatted_response,
            sources=list(files_to_process.keys())
        )
    
    def _extract_column_data(self, files_to_process: Dict[str, List[str]], user_id: str) -> Dict:
        """Extract actual data from specified columns in specified files"""
        
        column_data = {}
        
        for file_name, column_names in files_to_process.items():
            print(f"Extracting columns {column_names} from {file_name}")
            
            # Get chunks from this specific file (exclude metadata chunks)
            file_chunks = self.vector_store.search(f"file:{file_name}", user_id, top_k=50)
            file_specific_chunks = [
                c for c in file_chunks 
                if c.get('file_name', '') == file_name and not c.get('file_name', '').startswith('METADATA_')
            ]
            
            if not file_specific_chunks:
                print(f"No data found for {file_name}")
                continue
            
            # Extract values from specified columns
            file_data = {}
            for col_name in column_names:
                values = set()
                
                for chunk in file_specific_chunks:
                    content = chunk['content']
                    # Look for column data in various formats
                    import re
                    
                    # Pattern 1: column_name: value (more flexible)
                    pattern1 = rf'{re.escape(col_name)}[:\s]*([^\n\r]+)'
                    matches1 = re.findall(pattern1, content, re.IGNORECASE)
                    
                    # Pattern 2: Look for the column name and extract nearby content
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if col_name.lower() in line.lower():
                            # Extract value from same line after colon
                            if ':' in line:
                                parts = line.split(':', 1)
                                if len(parts) > 1:
                                    value = parts[1].strip()
                                    if value and len(value) > 1 and len(value) < 200:
                                        values.add(value)
                    
                    # Add pattern 1 matches
                    for match in matches1:
                        clean_value = match.strip().replace('"', '').replace("'", '')
                        if clean_value and len(clean_value) > 1 and len(clean_value) < 200:
                            values.add(clean_value)
                
                file_data[col_name] = list(values)
                print(f"Found {len(values)} unique values for {col_name}")
            
            column_data[file_name] = file_data
        
        return column_data
    
    def _format_with_llm(self, question: str, column_data: Dict, relevant_columns: List[Dict]) -> str:
        """Stage 2 LLM: Analyze, sort, and deduplicate results"""
        
        if not column_data:
            return "No relevant data found for your query."
        
        # Build context for LLM
        context = "Extracted column data:\n\n"
        for file_name, file_data in column_data.items():
            context += f"From {file_name}:\n"
            for col_name, values in file_data.items():
                context += f"  {col_name}: {', '.join(values[:10])}{'...' if len(values) > 10 else ''}\n"
            context += "\n"
        
        # Debug: Print what data we're actually sending to LLM
        print("=== DEBUG: Data being sent to Stage 2 LLM ===")
        print(context[:1000] + "..." if len(context) > 1000 else context)
        print("=== END DEBUG ===")
        
        # Build column descriptions
        column_descriptions = "\nColumn descriptions:\n"
        for col_meta in relevant_columns:
            column_descriptions += f"- {col_meta['column_name']} ({col_meta['file_name']}): {col_meta['description']}\n"
        
        prompt = f"""You are analyzing data to answer a user's question. 

User Question: {question}

{context}
{column_descriptions}

Instructions:
1. Analyze the extracted data to answer the user's question
2. Combine data from all files and remove duplicates
3. Sort the results logically (alphabetically or by relevance)
4. Format as a clean, numbered list
5. Show which file(s) each item came from
6. Count the total unique items

Format:
## [Answer Title] ([X] unique found):

1. **Item Name** _(from file1.csv, file2.csv)_
2. **Another Item** _(from file1.csv)_

**Data Sources:** file1.csv, file2.csv

Provide the final formatted response:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Analyze extracted data, remove duplicates, sort results, and format professionally."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            llm_response = response.choices[0].message.content
            
            # Check if LLM is hallucinating (generic project names like "Project A")
            if "Project A" in llm_response or "Project B" in llm_response:
                print("*** LLM HALLUCINATION DETECTED - Showing actual data ***")
                # Return actual data instead
                actual_data = "## Actual Data Found:\n\n"
                
                # Show Key Risks specifically since that's what the user asked for
                risk_data = []
                project_data = []
                
                for file_name, file_data in column_data.items():
                    actual_data += f"**From {file_name}:**\n"
                    for col_name, values in file_data.items():
                        if values:  # Only show columns with data
                            clean_values = [v for v in values if not v.startswith('Description:') and len(v) > 3]
                            if clean_values:
                                actual_data += f"- {col_name}: {', '.join(clean_values[:3])}{'...' if len(clean_values) > 3 else ''}\n"
                                
                                # Collect risk and project data
                                if 'risk' in col_name.lower():
                                    risk_data.extend(clean_values)
                                elif 'project' in col_name.lower() and 'name' in col_name.lower():
                                    project_data.extend(clean_values)
                    actual_data += "\n"
                
                # Show specific answer for projects at risk
                if risk_data and project_data:
                    actual_data += "## Projects with Risk Information:\n\n"
                    for i, project in enumerate(project_data[:5], 1):
                        actual_data += f"{i}. **{project}**\n"
                    actual_data += f"\n**Risk Information Found:** {', '.join(risk_data[:3])}\n"
                elif project_data:
                    actual_data += "## Projects Found (no specific risk data):\n\n"
                    for i, project in enumerate(project_data[:10], 1):
                        actual_data += f"{i}. **{project}**\n"
                
                if not any(file_data for file_data in column_data.values() if any(file_data.values())):
                    actual_data += "**No relevant data found for 'projects in risk'**\n\n"
                    actual_data += "Try queries like:\n- 'list all projects'\n- 'show project names'\n- 'list project status'"
                
                return actual_data
            
            return llm_response
            
        except Exception as e:
            print(f"Stage 2 LLM processing failed: {e}")
            
            # Fallback: simple formatting
            all_items = set()
            sources = set()
            
            for file_name, file_data in column_data.items():
                sources.add(file_name)
                for col_name, values in file_data.items():
                    all_items.update(values)
            
            if all_items:
                sorted_items = sorted(list(all_items))
                response = f"## Results ({len(sorted_items)} unique found):\n\n"
                for i, item in enumerate(sorted_items, 1):
                    response += f"{i}. **{item}**\n"
                response += f"\n**Data Sources:** {', '.join(sources)}"
                return response
            else:
                return "No data found matching your query."