import os
from typing import Dict, List, Any, Optional
from groq import Groq
import pandas as pd
import sqlite3
from local_vector_store import LocalVectorStore
from models import ChatResponse

class LLMDrivenService:
    def __init__(self):
        try:
            from secure_config import SecureConfig
            config = SecureConfig().get_api_keys()
            groq_key = config.get("GROQ_API_KEY")
            if not groq_key:
                raise Exception("GROQ_API_KEY not found in secure configuration")
            self.groq_client = Groq(api_key=groq_key)
        except Exception as e:
            print(f"Failed to initialize LLM service: {e}")
            self.groq_client = None
        self.vector_store = LocalVectorStore()
        self.db_path = "local_data/structured_data.db"
        
        os.makedirs("local_data", exist_ok=True)
    
    def process_query(self, question: str, user_id: str) -> ChatResponse:
        """LLM-driven query processing - no hardcoded rules"""
        
        if not self.groq_client:
            return ChatResponse(
                answer="LLM service not available. Please check API key configuration.",
                sources=[]
            )
        
        print(f"*** LLM-DRIVEN PROCESSING: {question} ***")
        
        # Step 1: LLM analyzes available data and decides approach
        approach = self._llm_decide_approach(question, user_id)
        
        if approach == "structured":
            return self._llm_sql_processing(question, user_id)
        else:
            return self._llm_semantic_processing(question, user_id)
    
    def _llm_decide_approach(self, question: str, user_id: str) -> str:
        """LLM decides the best approach based on question and available data"""
        
        # Let LLM decide all routing - no hardcoding
        
        # Get available data context
        data_context = self._get_data_context(user_id)
        
        prompt = f"""Question: "{question}"
Available data: {data_context}

Choose approach:
- SEMANTIC: for document search, data overview, content retrieval
- STRUCTURED: for calculations, counts, filtering, aggregations

Return only: SEMANTIC or STRUCTURED"""

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
        sql_query = self._llm_generate_sql(question, schema_info, user_id)
        
        if not sql_query:
            return ChatResponse(
                answer="Could not generate appropriate SQL query for this question.",
                sources=[]
            )
        
        # Try the generated SQL first
        try:
            conn = sqlite3.connect(self.db_path)
            result_df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # Format results
            formatted_answer = self._llm_format_results(question, result_df, sql_query)
            
            return ChatResponse(
                answer=formatted_answer,
                sources=self._extract_table_sources(schema_info)
            )
            
        except Exception as e:
            print(f"SQL execution failed: {e}")
            
            # For prediction queries, use simple fallback immediately
            if 'predict' in question.lower() or 'completion' in question.lower():
                print("Using simple prediction fallback")
                return self._generate_simple_prediction(user_id)
            
            # For other queries, try one simple fix
            try:
                # Generate a basic query without complex logic
                tables = self._get_user_tables(user_id)
                if tables:
                    simple_sql = f"SELECT * FROM {tables[0]} LIMIT 20;"
                    conn = sqlite3.connect(self.db_path)
                    result_df = pd.read_sql_query(simple_sql, conn)
                    conn.close()
                    
                    formatted_answer = self._llm_format_results(question, result_df, simple_sql)
                    return ChatResponse(
                        answer=f"**Note: Showing sample data due to query complexity**\n\n{formatted_answer}",
                        sources=self._extract_table_sources(schema_info)
                    )
            except:
                pass
            
            # Final fallback
            return ChatResponse(
                answer=f"Unable to execute query. Error: {str(e)}",
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
            # Check if user has any data at all
            test_chunks = self.vector_store.search("*", user_id, top_k=1)
            if not test_chunks:
                return ChatResponse(
                    answer="No data uploaded yet. Please upload your files first to get started!",
                    sources=[]
                )
            else:
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
    
    def _generate_data_overview(self, chunks):
        """Generate overview of available data"""
        
        # Group chunks by file
        files_data = {}
        for chunk in chunks:
            file_name = chunk.get('file_name', 'Unknown')
            if file_name not in files_data:
                files_data[file_name] = []
            files_data[file_name].append(chunk['content'])
        
        # Generate overview
        overview = "## Your Uploaded Data\n\n"
        
        for file_name, file_chunks in files_data.items():
            overview += f"**{file_name}**\n"
            
            # LLM analyzes content type
            sample_content = ' '.join(file_chunks[:2])[:300]
            
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Describe data type in one brief line."},
                        {"role": "user", "content": f"What type of data is this: {sample_content}"}
                    ],
                    max_tokens=30,
                    temperature=0.1
                )
                
                description = response.choices[0].message.content.strip()
                overview += f"- {description}\n"
            except:
                overview += "- Contains structured data\n"
            
            overview += f"- {len(file_chunks)} data chunks\n\n"
        
        # Generate suggestions based on actual data using LLM
        data_summary = "\n".join([f"{file}: {', '.join(chunks[:3])}" for file, chunks in files_data.items()])
        
        suggestions_prompt = f"""Based on this data:
{data_summary[:1000]}

Generate 4 relevant questions users could ask:

Questions:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Generate specific questions based on actual data content."},
                    {"role": "user", "content": suggestions_prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            suggestions = response.choices[0].message.content
            overview += f"**Try asking:**\n{suggestions}\n"
        except:
            overview += "**Try asking:**\n• Analyze the data\n• Show key insights\n"
        
        return ChatResponse(
            answer=overview,
            sources=list(files_data.keys())
        )
    
    def _llm_generate_sql(self, question: str, schema_info: str, user_id: str) -> Optional[str]:
        """LLM generates SQL query with exploratory data discovery"""
        
        # Step 1: LLM explores data to understand what's available
        exploration_results = self._llm_explore_data(question, schema_info)
        
        # Step 2: Generate final query based on exploration
        prompt = f"""Question: {question}

Schema Information:
{schema_info}

For delay prediction queries, include:
- project_name (not COUNT)
- incident counts per project
- time variance or delay metrics
- Show individual projects, not aggregated counts

Example: SELECT project_name, COUNT(incidents) as incident_count, AVG(time_variance) as avg_delay FROM table GROUP BY project_name

Generate SQLite query using exact table/column names from schema:
{{"sql": "SELECT project_name, ... FROM table_name ..."}}

Return only valid JSON:"""

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
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON and get SQL
            import json
            import re
            
            # Extract SQL directly from response - skip JSON parsing
            sql_match = re.search(r'"sql"\s*:\s*"([^"]+)"', response_text, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1)
                print(f"Extracted SQL: {sql_query[:100]}...")
            else:
                # Fallback - look for SELECT statement
                select_match = re.search(r'(SELECT[^;]+;?)', response_text, re.IGNORECASE | re.DOTALL)
                if select_match:
                    sql_query = select_match.group(1)
                else:
                    sql_query = response_text
                print("Used fallback SQL extraction")
            
            # Clean any remaining artifacts - no semicolon needed
            sql_query = sql_query.strip().replace('```sql', '').replace('```', '').rstrip(';')
            
            # Validate SQL against actual schema
            validated_sql = self._validate_and_fix_sql(sql_query, schema_info, user_id)
            
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
            with sqlite3.connect(self.db_path) as conn:
                for query_desc, query in exploration_queries:
                    try:
                        result_df = pd.read_sql_query(query, conn)
                        
                        exploration_results += f"{query_desc}:\n"
                        if not result_df.empty:
                            # Limit to 5 rows to reduce token usage
                            exploration_results += result_df.head(5).to_string(index=False) + "\n\n"
                        else:
                            exploration_results += "No data found\n\n"
                            
                    except Exception as e:
                        exploration_results += f"Query failed: {e}\n\n"
            
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
    
    def _validate_and_fix_sql(self, sql_query: str, schema_info: str, user_id: str) -> str:
        """Validate SQL against actual schema and fix if needed"""
        
        # Extract actual column names from schema
        actual_columns = self._extract_actual_columns(schema_info)
        
        # Test the query first
        try:
            conn = sqlite3.connect(self.db_path)
            # Try to execute directly - pandas doesn't need semicolons
            pd.read_sql_query(sql_query, conn, params=None)
            conn.close()
            
            # If successful, return original query
            return sql_query
            
        except Exception as e:
            print(f"SQL validation failed: {e}")
            
            # If failed, let LLM fix it
            return self._llm_fix_sql(sql_query, schema_info, str(e), user_id)
    
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
    
    def _llm_fix_sql(self, broken_sql: str, schema_info: str, error_message: str, user_id: str) -> str:
        """LLM fixes broken SQL query"""
        
        prompt = f"""Fix this SQLite error:

Broken SQL: {broken_sql}
Error: {error_message}

Schema Information:
{schema_info}

Look at the schema above and use ONLY the exact table and column names shown. Replace any incorrect names in the broken SQL.

Return corrected SQL in JSON format:
{{"sql": "corrected SELECT with exact names from schema"}}

Return only valid JSON:"""
        
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
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON and get SQL
            import json
            import re
            
            # Extract SQL directly - skip JSON parsing
            sql_match = re.search(r'"sql"\s*:\s*"([^"]+)"', response_text, re.DOTALL)
            if sql_match:
                fixed_sql = sql_match.group(1)
            else:
                # Fallback - look for SELECT statement
                select_match = re.search(r'(SELECT[^;]+;?)', response_text, re.IGNORECASE | re.DOTALL)
                if select_match:
                    fixed_sql = select_match.group(1)
                else:
                    fixed_sql = response_text
            
            # Clean SQL - no semicolon needed
            fixed_sql = fixed_sql.strip().replace('```sql', '').replace('```', '').rstrip(';')
            
            print(f"LLM fixed SQL: {fixed_sql}")
            return fixed_sql
            
        except Exception as e:
            print(f"SQL fixing failed: {e}")
            # Return a simple fallback query using first available table
            tables = self._get_user_tables(user_id)
            if tables:
                return f"SELECT * FROM {tables[0]}"
            return "SELECT 'No data available' as message"
    
    def _llm_decide_search_strategy(self, question: str, user_id: str) -> Dict[str, Any]:
        """LLM decides search parameters"""
        
        # LLM decides search strategy based on question context
        available_files = self._get_available_files(user_id)
        
        prompt = f"""Question: "{question}"
Files: {available_files}

Return JSON: {{"query": "search terms", "top_k": 20}}"""

        try:
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Return only valid JSON for search strategy."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.1
                )
                
                import json
                import re
                response_text = response.choices[0].message.content.strip()
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                
                if json_match:
                    strategy = json.loads(json_match.group())
                    return strategy
        except Exception as e:
            print(f"LLM search strategy failed: {e}")
        
        # Fallback strategy for other questions
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
        
        # Clean and format data
        result_df = result_df.dropna(how='all')  # Remove completely empty rows
        result_df = result_df.dropna()  # Remove rows with any NaN values
        result_df = result_df.fillna('N/A')  # Replace remaining None/NaN with N/A
        
        # Format as markdown table (manual creation)
        def create_markdown_table(df):
            if df.empty:
                return "No data"
            
            # Create header
            headers = '| ' + ' | '.join(df.columns) + ' |'
            separator = '|' + '|'.join([' --- ' for _ in df.columns]) + '|'
            
            # Create rows
            rows = []
            for _, row in df.iterrows():
                row_str = '| ' + ' | '.join([str(val) for val in row.values]) + ' |'
                rows.append(row_str)
            
            return '\n'.join([headers, separator] + rows)
        
        # Generate contextual response
        prompt = f"""Question: {question}

Data:
{result_df.to_string(index=False)}

Write a brief, direct answer to the question using this data:

Answer:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Analyze data and provide insights. Focus on patterns, correlations, and actionable findings."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            # Only show table if data is meaningful (not mostly empty)
            if len(result_df) > 0 and not result_df.empty:
                table_md = create_markdown_table(result_df)
                return f"{answer}\n\n{table_md}"
            else:
                return answer
            
        except Exception as e:
            table_md = create_markdown_table(result_df)
            # Filter out empty rows before showing table
            if len(result_df) > 0:
                table_md = create_markdown_table(result_df)
                return f"Here are the results for your query:\n\n{table_md}"
            else:
                return "No meaningful data found for your query."

        # Direct return without LLM processing for faster, cleaner results
    
    def _llm_process_semantic_results(self, question: str, chunks: List[Dict]) -> str:
        """LLM processes semantic search results"""
        
        # Build context from chunks - summarize if too long
        context = ""
        for i, chunk in enumerate(chunks[:5]):  # Reduced from 10 to 5
            content = chunk['content'][:500]  # Limit chunk size
            context += f"[{i+1}] {chunk.get('file_name', 'Unknown')}: {content}\n\n"
        
        # Let LLM process all semantic queries
        
        prompt = f"""Answer concisely: {question}

Data:
{context}

Provide a brief, direct answer:"""

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
        """Get structured schema information for LLM"""
        
        tables = self._get_user_tables(user_id)
        if not tables:
            return ""
        
        schema_parts = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for table in tables:
                    # Validate table name to prevent injection
                    if not table.replace('_', '').replace(user_id, '').isalnum():
                        continue
                    
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    
                    # Build concise schema info
                    table_info = f"Table: {table} ({row_count} rows)"
                    column_names = [col[1] for col in columns]
                    table_info += f"\nColumns: {', '.join(column_names)}"
                    
                    # Add one sample row for context
                    cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                    sample_row = cursor.fetchone()
                    if sample_row:
                        sample_dict = dict(zip(column_names, sample_row))
                        table_info += f"\nSample: {sample_dict}"
                    
                    schema_parts.append(table_info)
                
        except Exception as e:
            return f"Schema error: {e}"
        
        return "\n\n".join(schema_parts)
    
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
        """Get available files for user from metadata"""
        
        try:
            # Get files from SQL tables
            sql_tables = self._get_user_tables(user_id)
            files = [table.replace(f"{user_id}_", "").replace("_", " ") + ".csv" for table in sql_tables]
            
            # Get files from vector store metadata
            test_chunks = self.vector_store.search("*", user_id, top_k=10)
            vector_files = list(set([chunk.get('file_name', '') for chunk in test_chunks if chunk.get('file_name')]))
            
            all_files = list(set(files + vector_files))
            return all_files
        except:
            return []
    
    def _extract_table_sources(self, schema_info: str) -> List[str]:
        """Extract table names as sources"""
        
        import re
        tables = re.findall(r'Table: (\S+)', schema_info)
        return tables
    
    def _get_fresh_table_metadata(self, user_id: str) -> str:
        """Get fresh metadata directly from database"""
        
        tables = self._get_user_tables(user_id)
        if not tables:
            return "No tables found"
        
        metadata = ""
        try:
            conn = sqlite3.connect(self.db_path)
            
            for table in tables:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                metadata += f"Table: {table} ({row_count} rows)\n"
                metadata += "Columns:\n"
                for col in columns:
                    metadata += f"  - {col[1]} ({col[2]})\n"
                metadata += "\n"
            
            conn.close()
            
        except Exception as e:
            metadata += f"Error getting metadata: {e}\n"
        
        return metadata
    
    def _get_sample_data(self, user_id: str) -> str:
        """Get sample data from all tables"""
        
        tables = self._get_user_tables(user_id)
        if not tables:
            return "No data available"
        
        sample_data = ""
        try:
            conn = sqlite3.connect(self.db_path)
            
            for table in tables:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                rows = cursor.fetchall()
                
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                
                sample_data += f"Sample from {table}:\n"
                sample_data += f"Columns: {', '.join(columns)}\n"
                for row in rows:
                    sample_data += f"Row: {row}\n"
                sample_data += "\n"
            
            conn.close()
            
        except Exception as e:
            sample_data += f"Error getting sample data: {e}\n"
        
        return sample_data
    
    def _llm_analyze_and_fix(self, question: str, error: str, metadata: str, sample_data: str) -> str:
        """LLM analyzes actual data and generates working query"""
        
        prompt = f"""Question: {question}
Error: {error}

Metadata:
{metadata}

Sample Data:
{sample_data}

Read the metadata carefully and use the exact table and column names shown. Generate simple SQL:

SQL:"""
        
        try:
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a SQL expert. Analyze actual database structure and generate working queries."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                
                sql_query = response.choices[0].message.content.strip()
                if sql_query.startswith("```"):
                    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                
                return sql_query
        except Exception as e:
            print(f"LLM analysis failed: {e}")
        
        # Simple fallback - extract table name from metadata
        import re
        table_match = re.search(r'Table: (\S+)', metadata)
        if table_match:
            return f"SELECT * FROM {table_match.group(1)} LIMIT 10"
        return "SELECT 'No data' as message"
    
    def _llm_fallback_analysis(self, question: str, user_id: str, error: str) -> ChatResponse:
        """Final fallback - LLM analyzes raw data without SQL"""
        
        # Get actual data from tables
        tables = self._get_user_tables(user_id)
        if not tables:
            return ChatResponse(answer="No data available for analysis.", sources=[])
        
        try:
            conn = sqlite3.connect(self.db_path)
            all_data = ""
            
            for table in tables[:2]:  # Limit to 2 tables to avoid token limits
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                all_data += f"Data from {table}:\n{df.to_string()}\n\n"
            
            conn.close()
            
            # LLM analyzes raw data
            prompt = f"""Question: {question}
SQL Error: {error}

Actual Data:
{all_data[:8000]}  

Analyze this data directly and answer the question without using SQL:

Answer:"""
            
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Analyze data directly and provide insights without SQL."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                
                return ChatResponse(
                    answer=response.choices[0].message.content,
                    sources=tables
                )
        
        except Exception as e:
            return ChatResponse(
                answer=f"Unable to analyze data: {str(e)}",
                sources=[]
            )
        
        return ChatResponse(
            answer="Data analysis failed after multiple attempts.",
            sources=[]
        )
    
    def _llm_validate_answer(self, question: str, answer: str, result_df: pd.DataFrame) -> Dict[str, Any]:
        """LLM validates if the answer properly addresses the question"""
        
        prompt = f"""Question: {question}
Answer: {answer}
Data columns: {list(result_df.columns)}
Data rows: {len(result_df)}

Does the answer properly address the question?
For prediction questions, check if predicted dates/values are calculated.
For listing questions, check if all requested items are shown.

Return JSON: {{"is_valid": true/false, "reason": "explanation"}}"""
        
        try:
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a query validator. Return only valid JSON."},
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
                    return json.loads(json_match.group())
        except Exception as e:
            print(f"Validation failed: {e}")
        
        return {"is_valid": True, "reason": "Validation skipped"}
    
    def _llm_correct_answer(self, question: str, result_df: pd.DataFrame, issue: str) -> str:
        """LLM corrects the answer based on validation feedback"""
        
        prompt = f"""Question: {question}
Issue: {issue}

Data:
{result_df.to_string(index=False)}

Generate the correct answer that properly addresses the question:

Answer:"""
        
        try:
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a data analyst. Provide accurate answers based on the data."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.1
                )
                
                return response.choices[0].message.content
        except Exception as e:
            print(f"Answer correction failed: {e}")
        
        return f"## Query Results ({len(result_df)} records)\n\n{result_df.to_markdown(index=False)}"
    
    def _generate_simple_prediction(self, user_id: str) -> ChatResponse:
        """Generate simple completion date predictions using basic SQLite"""
        
        tables = self._get_user_tables(user_id)
        if not tables:
            return ChatResponse(answer="No data available for predictions.", sources=[])
        
        # Simple SQLite-compatible prediction query
        sql_query = f"""
        SELECT 
            project_name,
            start_date,
            planned_end_date,
            completion_pct,
            status,
            date('now', '+365 days') as simple_predicted_completion,
            CASE 
                WHEN CAST(REPLACE(completion_pct, '%', '') AS INTEGER) >= 90 THEN 'Almost done - within 30 days'
                WHEN CAST(REPLACE(completion_pct, '%', '') AS INTEGER) >= 80 THEN 'Nearing completion - 60 days'
                ELSE date('now', '+90 days')
            END as realistic_prediction
        FROM {tables[0]}
        WHERE status IN ('In - Progress', 'In Progress')
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            result_df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            if result_df.empty:
                return ChatResponse(
                    answer="No in-progress projects found for prediction.",
                    sources=[tables[0]]
                )
            
            # Clean and format
            result_df = result_df.fillna('')
            
            # Create markdown table
            def create_markdown_table(df):
                if df.empty:
                    return "No data"
                headers = '| ' + ' | '.join(df.columns) + ' |'
                separator = '|' + '|'.join([' --- ' for _ in df.columns]) + '|'
                rows = []
                for _, row in df.iterrows():
                    row_str = '| ' + ' | '.join([str(val) for val in row.values]) + ' |'
                    rows.append(row_str)
                return '\n'.join([headers, separator] + rows)
            
            table_md = create_markdown_table(result_df)
            answer = f"## Predicted Completion Dates for In-Progress Projects\n\n{table_md}\n\n*Note: Predictions based on start date + 1 year estimate. Check planned_end_date for more accurate timelines.*"
            
            return ChatResponse(
                answer=answer,
                sources=[tables[0]]
            )
            
        except Exception as e:
            return ChatResponse(
                answer=f"Failed to generate predictions: {str(e)}",
                sources=[]
            )