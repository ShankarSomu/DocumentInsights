import os
from typing import Dict, List, Any, Optional
from groq import Groq
import pandas as pd
import sqlite3
from local_vector_store import LocalVectorStore
from models import ChatResponse

class LLMDrivenService:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.vector_store = LocalVectorStore()
        self.db_path = "local_data/structured_data.db"
        
        os.makedirs("local_data", exist_ok=True)
    
    def process_query(self, question: str, user_id: str) -> ChatResponse:
        """LLM-driven query processing - no hardcoded rules"""
        
        print(f"*** LLM-DRIVEN PROCESSING: {question} ***")
        
        # Step 1: LLM analyzes available data and decides approach
        approach = self._llm_decide_approach(question, user_id)
        
        if approach == "structured":
            return self._llm_sql_processing(question, user_id)
        else:
            return self._llm_semantic_processing(question, user_id)
    
    def _llm_decide_approach(self, question: str, user_id: str) -> str:
        """LLM decides the best approach based on question and available data"""
        
        # Get available data context
        data_context = self._get_data_context(user_id)
        
        prompt = f"""You are an intelligent query router. Analyze the user's question and available data to decide the best approach.

USER QUESTION: "{question}"

AVAILABLE DATA:
{data_context}

DECISION CRITERIA:
- If the user is asking for specific document content, reports, or named items → use SEMANTIC search
- If the user wants calculations, aggregations, filtering, or data analysis → use STRUCTURED queries
- Consider what the user actually wants as output

Think step by step:
1. What is the user asking for?
2. What type of output do they expect?
3. Which approach would best serve their need?

Return only: STRUCTURED or SEMANTIC"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a query routing expert. Analyze and decide the best approach."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            decision = response.choices[0].message.content.strip().upper()
            print(f"LLM Decision: {decision}")
            
            return "structured" if "STRUCTURED" in decision else "semantic"
            
        except Exception as e:
            print(f"LLM decision failed: {e}")
            return "semantic"  # Default fallback
    
    def _llm_sql_processing(self, question: str, user_id: str) -> ChatResponse:
        """LLM generates and executes SQL queries"""
        
        # Get detailed schema information
        schema_info = self._get_detailed_schema(user_id)
        
        if not schema_info:
            return ChatResponse(
                answer="No structured data available for SQL queries.",
                sources=[]
            )
        
        # LLM generates SQL query
        sql_query = self._llm_generate_sql(question, schema_info)
        
        if not sql_query:
            return ChatResponse(
                answer="Could not generate appropriate SQL query for this question.",
                sources=[]
            )
        
        # Execute SQL
        try:
            conn = sqlite3.connect(self.db_path)
            result_df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # LLM formats the results
            formatted_answer = self._llm_format_results(question, result_df, sql_query)
            
            return ChatResponse(
                answer=formatted_answer,
                sources=self._extract_table_sources(schema_info)
            )
            
        except Exception as e:
            return ChatResponse(
                answer=f"SQL execution error: {str(e)}",
                sources=[]
            )
    
    def _llm_semantic_processing(self, question: str, user_id: str) -> ChatResponse:
        """LLM processes semantic queries using vector search"""
        
        # LLM decides search strategy
        search_strategy = self._llm_decide_search_strategy(question, user_id)
        
        # Execute search based on strategy
        chunks = self.vector_store.search(
            query=search_strategy["query"], 
            user_id=user_id, 
            top_k=search_strategy["top_k"],
            file_filter=search_strategy.get("file_filter")
        )
        
        if not chunks:
            return ChatResponse(
                answer="No relevant information found for your query.",
                sources=[]
            )
        
        # LLM processes and formats semantic results
        formatted_answer = self._llm_process_semantic_results(question, chunks)
        
        return ChatResponse(
            answer=formatted_answer,
            sources=list(set([chunk.get('file_name', 'Unknown') for chunk in chunks]))
        )
    
    def _llm_generate_sql(self, question: str, schema_info: str) -> Optional[str]:
        """LLM generates SQL query with exploratory data discovery"""
        
        # Step 1: LLM explores data to understand what's available
        exploration_results = self._llm_explore_data(question, schema_info)
        
        # Step 2: Generate final query based on exploration
        prompt = f"""Generate a SQL query to answer this question: {question}

AVAILABLE SCHEMA:
{schema_info}

DATA EXPLORATION RESULTS:
{exploration_results}

INSTRUCTIONS:
- Use the exploration results to understand what data actually exists
- Generate a query that will find relevant data based on the exploration
- Use exact table and column names as shown in schema
- Return ONLY the SQL query, no explanations

SQL Query:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Generate accurate SQLite queries based on actual data exploration."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up SQL
            if sql_query.startswith("```"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            # Validate SQL against actual schema
            validated_sql = self._validate_and_fix_sql(sql_query, schema_info)
            
            print(f"Generated SQL after exploration: {validated_sql}")
            return validated_sql
            
        except Exception as e:
            print(f"SQL generation failed: {e}")
            return None
    
    def _llm_explore_data(self, question: str, schema_info: str) -> str:
        """LLM explores data to understand what's actually available"""
        
        # Generate exploration queries
        exploration_queries = self._generate_exploration_queries(schema_info)
        
        exploration_results = "DATA EXPLORATION RESULTS:\n\n"
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            for query_desc, query in exploration_queries:
                try:
                    result_df = pd.read_sql_query(query, conn)
                    
                    exploration_results += f"{query_desc}:\n"
                    if not result_df.empty:
                        if len(result_df) <= 10:
                            exploration_results += result_df.to_string(index=False) + "\n\n"
                        else:
                            exploration_results += result_df.head(10).to_string(index=False)
                            exploration_results += f"\n... and {len(result_df) - 10} more rows\n\n"
                    else:
                        exploration_results += "No data found\n\n"
                        
                except Exception as e:
                    exploration_results += f"Query failed: {e}\n\n"
            
            conn.close()
            
        except Exception as e:
            exploration_results += f"Database connection failed: {e}\n"
        
        print(f"Exploration completed: {len(exploration_results)} characters of results")
        return exploration_results
    
    def _generate_exploration_queries(self, schema_info: str) -> List[tuple]:
        """Generate basic exploration queries to understand the data"""
        
        import re
        
        # Extract table names from schema
        table_matches = re.findall(r'Table: (\S+)', schema_info)
        
        queries = []
        
        for table in table_matches:
            # Sample data from each table
            queries.append((
                f"Sample data from {table}",
                f"SELECT * FROM {table} LIMIT 5"
            ))
            
            # Get distinct values for key columns that might contain names/identifiers
            column_matches = re.findall(r'- (\w+)', schema_info)
            
            for column in column_matches:
                if any(keyword in column.lower() for keyword in ['name', 'title', 'id', 'type', 'status']):
                    queries.append((
                        f"Distinct values in {table}.{column}",
                        f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 10"
                    ))
        
        return queries[:10]  # Limit exploration queries
    
    def _validate_and_fix_sql(self, sql_query: str, schema_info: str) -> str:
        """Validate SQL against actual schema and fix if needed"""
        
        # Extract actual column names from schema
        actual_columns = self._extract_actual_columns(schema_info)
        
        # Test the query first
        try:
            conn = sqlite3.connect(self.db_path)
            # Try to execute with LIMIT 0 to check syntax without returning data
            test_query = f"SELECT COUNT(*) FROM ({sql_query}) LIMIT 0"
            conn.execute(test_query)
            conn.close()
            
            # If successful, return original query
            return sql_query
            
        except Exception as e:
            print(f"SQL validation failed: {e}")
            
            # If failed, let LLM fix it
            return self._llm_fix_sql(sql_query, schema_info, str(e))
    
    def _extract_actual_columns(self, schema_info: str) -> Dict[str, List[str]]:
        """Extract actual column names from schema info"""
        
        import re
        
        table_columns = {}
        current_table = None
        
        lines = schema_info.split('\n')
        for line in lines:
            line = line.strip()
            
            # Match table names
            table_match = re.match(r'Table: (\S+)', line)
            if table_match:
                current_table = table_match.group(1)
                table_columns[current_table] = []
            
            # Match column names
            elif line.startswith('- ') and current_table:
                column_match = re.match(r'- (\w+)', line)
                if column_match:
                    table_columns[current_table].append(column_match.group(1))
        
        return table_columns
    
    def _llm_fix_sql(self, broken_sql: str, schema_info: str, error_message: str) -> str:
        """LLM fixes broken SQL query"""
        
        prompt = f"""Fix this SQL query that has an error.

BROKEN SQL:
{broken_sql}

ERROR MESSAGE:
{error_message}

CORRECT SCHEMA:
{schema_info}

INSTRUCTIONS:
- Fix the SQL to use only columns that exist in the schema
- Keep the same query intent but use correct column names
- Return ONLY the corrected SQL query

CORRECTED SQL:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a SQL debugging expert. Fix queries to match the actual schema."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            fixed_sql = response.choices[0].message.content.strip()
            
            # Clean up SQL
            if fixed_sql.startswith("```"):
                fixed_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
            
            print(f"LLM fixed SQL: {fixed_sql}")
            return fixed_sql
            
        except Exception as e:
            print(f"SQL fixing failed: {e}")
            # Return a safe fallback query
            return "SELECT * FROM (SELECT 'Error: Could not generate valid SQL' as message) LIMIT 1"
    
    def _llm_decide_search_strategy(self, question: str, user_id: str) -> Dict[str, Any]:
        """LLM decides search parameters"""
        
        available_files = self._get_available_files(user_id)
        
        prompt = f"""Decide the best search strategy for this question: "{question}"

AVAILABLE FILES: {', '.join(available_files)}

Determine:
1. What search query would find the most relevant content?
2. How many results should we retrieve? (5-50)
3. Should we filter to a specific file? (if question mentions specific document)

Return JSON format:
{{"query": "optimized search query", "top_k": 20, "file_filter": "filename.csv or null"}}"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a search strategist. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            import json
            import re
            
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                strategy = json.loads(json_match.group())
                print(f"Search strategy: {strategy}")
                return strategy
            
        except Exception as e:
            print(f"Search strategy failed: {e}")
        
        # Fallback strategy
        return {"query": question, "top_k": 20, "file_filter": None}
    
    def _llm_format_results(self, question: str, result_df: pd.DataFrame, sql_query: str) -> str:
        """LLM formats SQL results"""
        
        if result_df.empty:
            return "No results found for your query."
        
        # Convert DataFrame to text
        data_summary = f"Found {len(result_df)} results with columns: {', '.join(result_df.columns)}"
        if len(result_df) <= 20:
            data_text = result_df.to_string(index=False)
        else:
            data_text = result_df.head(20).to_string(index=False)
            data_text += f"\n... and {len(result_df) - 20} more rows"
        
        prompt = f"""Format these query results for the user.

USER QUESTION: {question}
RESULTS SUMMARY: {data_summary}

RESULTS DATA:
{data_text}

Create a clear, well-formatted response that:
1. Directly answers the user's question
2. Presents the data in an organized way (use tables/lists as appropriate)
3. Includes key insights if relevant
4. Focuses on the data and insights

Format the response professionally with proper markdown."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a data presentation expert. Format results clearly and professionally."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Result formatting failed: {e}")
            return f"Query Results:\n\n{data_text}"
    
    def _llm_process_semantic_results(self, question: str, chunks: List[Dict]) -> str:
        """LLM processes semantic search results"""
        
        # Build context from chunks
        context = ""
        for i, chunk in enumerate(chunks[:10]):  # Limit context size
            context += f"[{i+1}] From {chunk.get('file_name', 'Unknown')}:\n{chunk['content']}\n\n"
        
        prompt = f"""Answer the user's question based on the retrieved information.

USER QUESTION: {question}

RETRIEVED INFORMATION:
{context}

Provide a comprehensive answer that:
1. Directly addresses the user's question
2. Uses the specific information from the retrieved content
3. Cites sources when relevant
4. If looking for a specific document, provide its full content
5. Format professionally with proper structure

Answer:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an information retrieval expert. Provide accurate, well-structured answers based on the provided content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Semantic processing failed: {e}")
            return f"Found {len(chunks)} relevant pieces of information, but failed to process them properly."
    
    def _get_data_context(self, user_id: str) -> str:
        """Get overview of available data"""
        
        # Get SQL tables
        sql_tables = self._get_user_tables(user_id)
        
        # Get vector store files
        vector_files = self.vector_store.list_user_files(user_id) if hasattr(self.vector_store, 'list_user_files') else []
        
        context = f"SQL Tables: {len(sql_tables)} available\n"
        if sql_tables:
            context += f"Table names: {', '.join(sql_tables[:3])}{'...' if len(sql_tables) > 3 else ''}\n"
        
        context += f"Document Files: {len(vector_files)} available\n"
        if vector_files:
            context += f"File names: {', '.join(vector_files[:3])}{'...' if len(vector_files) > 3 else ''}\n"
        
        return context
    
    def _get_detailed_schema(self, user_id: str) -> str:
        """Get detailed schema information"""
        
        tables = self._get_user_tables(user_id)
        if not tables:
            return ""
        
        schema_info = ""
        try:
            conn = sqlite3.connect(self.db_path)
            
            for table in tables:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                schema_info += f"\nTable: {table} ({row_count} rows)\n"
                schema_info += "Columns (EXACT NAMES - use these exactly):\n"
                for col in columns:
                    schema_info += f"  - {col[1]} (type: {col[2]})\n"
                
                # Add sample of actual column names for clarity
                column_names = [col[1] for col in columns]
                schema_info += f"\nALL COLUMN NAMES: {', '.join(column_names)}\n"
            
            conn.close()
            
        except Exception as e:
            schema_info += f"Error getting schema: {e}\n"
        
        return schema_info
    
    def _get_user_tables(self, user_id: str) -> List[str]:
        """Get user's SQL tables"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (f"{user_id}_%",))
            tables = [row[0] for row in cursor.fetchall()]
            
            # Filter out mapping tables
            tables = [t for t in tables if not t.endswith('_mapping')]
            
            conn.close()
            return tables
            
        except Exception as e:
            print(f"Error getting tables: {e}")
            return []
    
    def _get_available_files(self, user_id: str) -> List[str]:
        """Get available files for user"""
        
        # This would need to be implemented based on your vector store
        # For now, return empty list
        return []
    
    def _extract_table_sources(self, schema_info: str) -> List[str]:
        """Extract table names as sources"""
        
        import re
        tables = re.findall(r'Table: (\S+)', schema_info)
        return tables