from groq import Groq
from typing import List, Dict
from local_vector_store import LocalVectorStore
from schema_analyzer import SchemaAnalyzer
from intelligent_query_service import IntelligentQueryService
from query_router import QueryRouter
from models import ChatResponse
import os

class RAGService:
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
        self.intelligent_service = IntelligentQueryService()
        self.query_router = QueryRouter()
    
    def generate_answer(self, question: str, user_id: str) -> ChatResponse:
        # Determine if query needs all data
        comprehensive_keywords = ['all', 'list', 'show', 'total', 'count', 'every', 'complete']
        needs_all_data = any(keyword in question.lower() for keyword in comprehensive_keywords)
        
        # Debug: Show which user we're searching for
        print(f"Searching data for user_id: {user_id}")
        
        # Debug: Check if user has any data at all
        test_search = self.vector_store.search("*", user_id, top_k=5)
        print(f"Test search found {len(test_search)} chunks for user {user_id}")
        if test_search:
            print(f"Sample files: {[c.get('file_name', 'Unknown') for c in test_search[:3]]}")
        
        # Use query router to determine target files
        available_files = ["Project Management Dataset.csv", "employee_data_pyramid.csv", "incident.csv"]
        routing_result = self.query_router.route_query(question, available_files)
        
        print(f"Router result: {routing_result['reasoning']} (confidence: {routing_result['confidence']})")
        
        # Retrieve relevant context
        if needs_all_data:
            # For comprehensive queries, get data from routed files
            if routing_result['target_files']:
                target_file = routing_result['target_files'][0]  # Use first target file
                print(f"Targeting file: {target_file}")
                relevant_chunks = self.vector_store.search(question, user_id, top_k=1000, file_filter=target_file)
            else:
                relevant_chunks = self.vector_store.search(question, user_id, top_k=1000)
            
            # Debug: Show what files we found
            if relevant_chunks:
                found_files = set([chunk.get('file_name', 'Unknown') for chunk in relevant_chunks])
                print(f"Found data from files: {found_files}")
            else:
                print("No data found for this user")
            
            # Filter chunks based on query intent
            relevant_chunks = self._filter_chunks_by_query(question, relevant_chunks)
        else:
            # For specific queries, get focused results
            if routing_result['target_files']:
                target_file = routing_result['target_files'][0]
                relevant_chunks = self.vector_store.search(question, user_id, top_k=8, file_filter=target_file)
            else:
                relevant_chunks = self.vector_store.search(question, user_id, top_k=8)
        
        if not relevant_chunks:
            print(f"No relevant chunks found for query: {question}")
            print(f"User {user_id} has {len(test_search)} total chunks")
            return ChatResponse(
                answer="I don't have enough data to answer that question. Please upload some CSV files first.",
                sources=[]
            )
        
        # For risk-related queries, search only project management data
        if 'risk' in question.lower() and ('project' in question.lower() or 'list' in question.lower()):
            print(f"*** USING DIRECT RISK QUERY: {question} ***")
            # Search only in project management files
            project_chunks = self.vector_store.search(question, user_id, top_k=100, file_filter="Project Management Dataset.csv")
            if not project_chunks:
                # Fallback to all files if no project management file found
                project_chunks = relevant_chunks
            return self._direct_risk_query(question, project_chunks, user_id)
        
        # Use two-stage metadata-driven query service for other comprehensive queries
        elif needs_all_data:
            print(f"*** USING TWO-STAGE QUERY SERVICE: {question} ***")
            from two_stage_query_service import TwoStageQueryService
            two_stage_service = TwoStageQueryService()
            return two_stage_service.process_query(question, user_id)
        else:
            print(f"*** USING REGULAR RAG SERVICE: {question} ***")
        
        # Build context from retrieved chunks
        context = self._build_context(relevant_chunks)
        
        # Generate answer using Groq
        if not self.groq_client:
            return ChatResponse(
                answer="LLM service not available. Please check API key configuration.",
                sources=list(set([chunk.get('file_name', 'Unknown') for chunk in relevant_chunks]))
            )
        
        prompt = self._build_prompt(question, context)
        
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a concise data analyst. Provide brief, direct answers using tables or bullet points. Avoid lengthy explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        # Add follow-up suggestions
        from followup_generator import generate_followup_suggestions
        followups = generate_followup_suggestions(question, relevant_chunks)
        if followups:
            answer += "\n\n**Try asking:**\n" + "\n".join([f"• {q}" for q in followups])
        
        sources = [chunk.get('file_name', 'Unknown') for chunk in relevant_chunks]
        
        return ChatResponse(answer=answer, sources=list(set(sources)))
    
    def _build_context(self, chunks: List[Dict]) -> str:
        context_parts = []
        total_chars = 0
        max_chars = 12000  # Increase limit since we're sampling better
        
        for chunk in chunks:
            chunk_text = f"From {chunk.get('file_name', 'file')}: {chunk['content']}"
            
            if total_chars + len(chunk_text) > max_chars:
                # Truncate the chunk if it would exceed limit
                remaining_chars = max_chars - total_chars
                if remaining_chars > 100:  # Only add if meaningful content fits
                    chunk_text = chunk_text[:remaining_chars] + "..."
                    context_parts.append(chunk_text)
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        context = "\n\n".join(context_parts)
        
        # Add summary if we had to truncate
        if len(chunks) > len(context_parts):
            context += f"\n\n[Note: Showing {len(context_parts)} of {len(chunks)} total data chunks due to size limits]"
        
        return context
    
    def _build_prompt(self, question: str, context: str) -> str:
        return f"""Answer concisely: {question}

Data:
{context}

Provide a brief, direct answer with key numbers/facts only:"""
    
    def _process_with_llm(self, question: str, all_chunks: List[Dict], user_id: str) -> ChatResponse:
        """Use LLM intelligence to process queries with schema awareness"""
        
        # Get schema context to help LLM understand the data structure
        schema_context = self.schema_analyzer.get_query_context(user_id, question)
        
        # Sample data strategically from all files
        sample_size = 30
        if len(all_chunks) <= sample_size:
            sampled_chunks = all_chunks
        else:
            # Sample evenly from all files
            files = {}
            for chunk in all_chunks:
                file_name = chunk.get('file_name', 'Unknown')
                if file_name not in files:
                    files[file_name] = []
                files[file_name].append(chunk)
            
            sampled_chunks = []
            chunks_per_file = max(1, sample_size // len(files))
            for file_chunks in files.values():
                sampled_chunks.extend(file_chunks[:chunks_per_file])
        
        print(f"Using LLM to process {len(sampled_chunks)} chunks from {len(set([c.get('file_name', 'Unknown') for c in sampled_chunks]))} files")
        
        # Build context for LLM
        context = self._build_context(sampled_chunks)
        
        # Create intelligent prompt with schema awareness
        prompt = f"""Answer this question concisely using the data provided: {question}

Data:
{context}

Provide a brief, direct answer. Use tables or bullet points for clarity. Avoid lengthy explanations."""
        
        try:
            if not self.groq_client:
                answer = "LLM service not available for processing this query."
            else:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a concise data analyst. Provide brief, direct answers. Use bullet points or tables."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                
                answer = response.choices[0].message.content
            
            # Add info about data completeness
            if len(all_chunks) > len(sampled_chunks):
                answer += f"\n\n*Note: Analyzed {len(sampled_chunks)} samples from {len(all_chunks)} total data entries across all uploaded files.*"
            
        except Exception as e:
            print(f"LLM processing failed: {e}")
            answer = f"Error processing {len(all_chunks)} data entries with LLM: {str(e)}"
        
        try:
            sources = list(set([chunk.get('file_name', 'Unknown') for chunk in all_chunks]))
        except Exception as e:
            print(f"Error getting sources: {e}")
            sources = ['Unknown']
        
        return ChatResponse(answer=answer, sources=sources)
    
    def _direct_risk_query(self, question: str, chunks: List[Dict], user_id: str) -> ChatResponse:
        """Direct approach to find projects with risk information"""
        
        # Look for actual risk-related content in chunks
        risk_projects = {}
        all_projects = set()
        
        for chunk in chunks:
            content = chunk['content']
            file_name = chunk.get('file_name', 'Unknown')
            
            # Extract project names and risk information from content
            import re
            
            # Look for project names
            project_patterns = [
                r'project[_\s]*name[:\s]*([^\n,;]+)',
                r'name[:\s]*([^\n,;]+)',
            ]
            
            for pattern in project_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    clean_name = match.strip().replace('"', '').replace("'", '')
                    if (clean_name and len(clean_name) > 2 and len(clean_name) < 50 and 
                        not clean_name.lower().startswith(('description', 'data in', 'show', 'list'))):
                        all_projects.add(clean_name)
            
            # Look for risk information
            if 'risk' in content.lower():
                # Extract risk details
                risk_patterns = [
                    r'risk[s]*[:\s]*([^\n]+)',
                    r'key[\s]*risk[s]*[:\s]*([^\n]+)',
                ]
                
                for pattern in risk_patterns:
                    risk_matches = re.findall(pattern, content, re.IGNORECASE)
                    for risk_match in risk_matches:
                        clean_risk = risk_match.strip().replace('"', '').replace("'", '')
                        if (clean_risk and len(clean_risk) > 5 and 
                            not clean_risk.lower().startswith(('description', 'data in', 'show', 'list'))):
                            
                            # Try to associate with project names in same chunk
                            for project in all_projects:
                                if project.lower() in content.lower():
                                    if project not in risk_projects:
                                        risk_projects[project] = []
                                    risk_projects[project].append(clean_risk)
        
        # Format response
        if risk_projects:
            answer = f"## Projects with Risk Information ({len(risk_projects)} found):\n\n"
            for i, (project, risks) in enumerate(risk_projects.items(), 1):
                answer += f"{i}. **{project}**\n"
                for risk in risks[:2]:  # Show first 2 risks
                    answer += f"   - {risk}\n"
                answer += "\n"
        elif all_projects:
            sorted_projects = sorted(list(all_projects))
            answer = f"## All Projects Found ({len(sorted_projects)} total):\n\n"
            answer += "\n".join([f"{i+1}. **{name}**" for i, name in enumerate(sorted_projects)])
            answer += "\n\n*Note: No specific risk information found in the data.*"
        else:
            answer = "No projects or risk information found in the uploaded data."
        
        # Add follow-up suggestions
        from followup_generator import generate_followup_suggestions
        followups = generate_followup_suggestions(question, chunks)
        if followups:
            answer += "\n\n**Try asking:**\n" + "\n".join([f"• {q}" for q in followups])
        
        sources = list(set([chunk.get('file_name', 'Unknown') for chunk in chunks]))
        return ChatResponse(answer=answer, sources=sources)
    
    def _determine_target_file(self, question: str) -> str:
        """Determine which file to search based on query keywords"""
        question_lower = question.lower()
        
        # Project-related queries
        if any(word in question_lower for word in ['project', 'risk', 'milestone', 'deliverable', 'phase']):
            return "Project Management Dataset.csv"
        
        # Employee-related queries
        elif any(word in question_lower for word in ['employee', 'staff', 'team', 'skill', 'capacity', 'manager']):
            return "employee_data_pyramid.csv"
        
        # Incident-related queries
        elif any(word in question_lower for word in ['incident', 'issue', 'ticket', 'bug', 'problem']):
            return "incident.csv"
        
        # No specific file determined
        return None
    
    def _filter_chunks_by_query(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """Filter chunks based on what the user is asking for"""
        question_lower = question.lower()
        
        # Define keywords for different data types
        project_keywords = ['project', 'projects']
        issue_keywords = ['issue', 'issues', 'bug', 'bugs', 'ticket', 'tickets']
        employee_keywords = ['employee', 'employees', 'staff', 'team', 'people']
        task_keywords = ['task', 'tasks', 'todo', 'work']
        
        # Determine what type of data user wants
        wants_projects = any(keyword in question_lower for keyword in project_keywords)
        wants_issues = any(keyword in question_lower for keyword in issue_keywords)
        wants_employees = any(keyword in question_lower for keyword in employee_keywords)
        wants_tasks = any(keyword in question_lower for keyword in task_keywords)
        
        if not any([wants_projects, wants_issues, wants_employees, wants_tasks]):
            # If unclear, return all chunks
            return chunks
        
        filtered_chunks = []
        
        for chunk in chunks:
            content_lower = chunk['content'].lower()
            file_name_lower = chunk.get('file_name', '').lower()
            
            # Score this chunk based on relevance
            score = 0
            
            if wants_projects:
                if any(word in content_lower for word in ['project', 'proj', 'initiative']):
                    score += 2
                if any(word in file_name_lower for word in ['project', 'proj']):
                    score += 3
            
            if wants_issues:
                if any(word in content_lower for word in ['issue', 'bug', 'ticket', 'problem']):
                    score += 2
                if any(word in file_name_lower for word in ['issue', 'bug', 'ticket']):
                    score += 3
            
            if wants_employees:
                if any(word in content_lower for word in ['employee', 'staff', 'team', 'person', 'name']):
                    score += 2
                if any(word in file_name_lower for word in ['employee', 'staff', 'team', 'hr']):
                    score += 3
            
            if wants_tasks:
                if any(word in content_lower for word in ['task', 'todo', 'work', 'assignment']):
                    score += 2
                if any(word in file_name_lower for word in ['task', 'todo', 'work']):
                    score += 3
            
            # Include chunks with score > 0 or if no specific filtering worked
            if score > 0:
                chunk['relevance_score'] = score
                filtered_chunks.append(chunk)
        
        # Sort by relevance score (highest first)
        filtered_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # If filtering removed too much, return original chunks
        if len(filtered_chunks) < len(chunks) * 0.1:  # Less than 10% remained
            print(f"Filtering too aggressive, returning all {len(chunks)} chunks")
            return chunks
        
        print(f"Filtered from {len(chunks)} to {len(filtered_chunks)} relevant chunks")
        return filtered_chunks