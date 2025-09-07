import sqlite3
import pandas as pd
import hashlib
import json
import os
from typing import Dict, List, Any, Optional
from groq import Groq
from local_vector_store import LocalVectorStore
from models import ChatResponse, DataChunk

class ThreePhaseSQLService:
    def __init__(self):
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
        self.vector_store = LocalVectorStore()
        self.db_path = "local_data/structured_data.db"
        self.cache_file = "local_data/sql_query_cache.json"
        self.metadata_cache = {}
        self.query_cache = {}
        
        os.makedirs("local_data", exist_ok=True)
        self._load_caches()
    
    def process_sql_query(self, question: str, user_id: str) -> ChatResponse:
        """3-Phase SQL processing: Metadata Search → SQL Generation → Execution"""
        
        # Check query cache first
        cache_key = self._get_cache_key(question, user_id)
        if cache_key in self.query_cache:
            print("*** USING CACHED SQL RESULT ***")
            cached_result = self.query_cache[cache_key]
            return ChatResponse(answer=cached_result['answer'], sources=cached_result['sources'])
        
        print(f"*** 3-PHASE SQL PROCESSING: {question} ***")
        
        # Phase 1: Semantic search for relevant tables/columns
        relevant_metadata = self._phase1_find_relevant_metadata(question, user_id)
        
        if not relevant_metadata:
            return ChatResponse(
                answer="No relevant tables found for your query. Please upload data first.",
                sources=[]
            )
        
        # Phase 2: Generate SQL using relevant schema
        sql_query = self._phase2_generate_sql(question, relevant_metadata)
        
        if not sql_query:
            return ChatResponse(
                answer="Could not generate SQL query for this question.",
                sources=[]
            )
        
        # Phase 3: Execute SQL and format results
        result = self._phase3_execute_and_format(question, sql_query, relevant_metadata)
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        return result
    
    def ingest_table_metadata(self, df: pd.DataFrame, table_name: str, user_id: str):
        """Phase 1: Ingest table & column metadata into vector store"""
        
        print(f"Phase 1: Ingesting metadata for {table_name}")
        
        metadata_chunks = []
        
        # Create table-level metadata
        table_info = f"""
        Table: {table_name}
        User: {user_id}
        Total Rows: {len(df)}
        Total Columns: {len(df.columns)}
        Description: Data table containing {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}
        """
        
        table_chunk = DataChunk(
            content=table_info.strip(),
            user_id=user_id,
            file_name=f"METADATA_TABLE_{table_name}",
            chunk_id=f"table_{table_name}",
            metadata={
                "type": "table_metadata",
                "table_name": table_name,
                "user_id": user_id,
                "row_count": len(df),
                "column_count": len(df.columns)
            }
        )
        metadata_chunks.append(table_chunk)
        
        # Create column-level metadata
        for col in df.columns:
            # Analyze column
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            unique_count = df[col].nunique()
            
            # Get sample values
            sample_values = df[col].dropna().astype(str).head(5).tolist()
            
            # Determine column purpose
            col_purpose = self._determine_column_purpose(col, sample_values, dtype)
            
            column_info = f"""
            Column: {col}
            Table: {table_name}
            Data Type: {dtype}
            Purpose: {col_purpose}
            Non-null Count: {non_null_count}/{len(df)}
            Unique Values: {unique_count}
            Sample Values: {', '.join(sample_values)}
            
            Queries this column can answer:
            - Questions about {col.lower().replace('_', ' ')}
            - Filtering by {col_purpose}
            - Counting/grouping {col.lower().replace('_', ' ')}
            """
            
            column_chunk = DataChunk(
                content=column_info.strip(),
                user_id=user_id,
                file_name=f"METADATA_COLUMN_{table_name}_{col}",
                chunk_id=f"column_{table_name}_{col}",
                metadata={
                    "type": "column_metadata",
                    "table_name": table_name,
                    "column_name": col,
                    "user_id": user_id,
                    "data_type": dtype,
                    "purpose": col_purpose,
                    "sample_values": sample_values[:3]
                }
            )
            metadata_chunks.append(column_chunk)
        
        # Store metadata in vector store
        self.vector_store.store_chunks(metadata_chunks)
        
        print(f"Stored {len(metadata_chunks)} metadata chunks for {table_name}")
    
    def _phase1_find_relevant_metadata(self, question: str, user_id: str) -> List[Dict]:
        """Phase 1: Semantic search for relevant tables/columns"""
        
        print(f"Phase 1: Searching metadata for: {question}")
        
        # Search for relevant metadata
        metadata_results = self.vector_store.search(question, user_id, top_k=20)
        
        # Filter to only metadata chunks
        relevant_metadata = []
        for result in metadata_results:
            if result.get('file_name', '').startswith('METADATA_'):
                relevant_metadata.append(result)
        
        print(f"Found {len(relevant_metadata)} metadata chunks")
        
        # If no metadata found, fallback to all user tables
        if not relevant_metadata:
            print("No metadata chunks found, using fallback approach")
            return self._fallback_get_all_tables(user_id)
        
        # Group by table
        tables_found = {}
        for result in relevant_metadata:
            # Extract table name from metadata or file name
            table_name = None
            
            if hasattr(result, 'metadata') and result.metadata:
                table_name = result.metadata.get('table_name')
            elif 'table_name' in result:
                table_name = result['table_name']
            else:
                # Extract from file name: METADATA_TABLE_tablename or METADATA_COLUMN_tablename_col
                file_name = result.get('file_name', '')
                if 'METADATA_TABLE_' in file_name:
                    table_name = file_name.replace('METADATA_TABLE_', '')
                elif 'METADATA_COLUMN_' in file_name:
                    parts = file_name.replace('METADATA_COLUMN_', '').split('_')
                    if len(parts) >= 2:
                        table_name = '_'.join(parts[:-1])  # Everything except last part (column name)
            
            if table_name:
                if table_name not in tables_found:
                    tables_found[table_name] = {
                        'table_name': table_name,
                        'columns': [],
                        'metadata': []
                    }
                
                # Extract column info if this is a column metadata
                if 'METADATA_COLUMN_' in result.get('file_name', ''):
                    # Parse column info from content or metadata
                    column_info = self._extract_column_info(result)
                    if column_info:
                        tables_found[table_name]['columns'].append(column_info)
                
                tables_found[table_name]['metadata'].append(result)
        
        relevant_tables = list(tables_found.values())
        
        print(f"Phase 1: Found {len(relevant_tables)} relevant tables with {sum(len(t['columns']) for t in relevant_tables)} columns")
        
        return relevant_tables
    
    def _extract_column_info(self, result: Dict) -> Optional[Dict]:
        """Extract column information from metadata result"""
        
        try:
            content = result.get('content', '')
            
            # Parse column info from content
            column_name = None
            data_type = None
            purpose = None
            sample_values = []
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Column:'):
                    column_name = line.replace('Column:', '').strip()
                elif line.startswith('Data Type:'):
                    data_type = line.replace('Data Type:', '').strip()
                elif line.startswith('Purpose:'):
                    purpose = line.replace('Purpose:', '').strip()
                elif line.startswith('Sample Values:'):
                    values_str = line.replace('Sample Values:', '').strip()
                    sample_values = [v.strip() for v in values_str.split(',') if v.strip()]
            
            if column_name:
                return {
                    'column_name': column_name,
                    'data_type': data_type or 'unknown',
                    'purpose': purpose or 'data field',
                    'sample_values': sample_values
                }
        
        except Exception as e:
            print(f"Error extracting column info: {e}")
        
        return None
    
    def _fallback_get_all_tables(self, user_id: str) -> List[Dict]:
        """Fallback: Get all tables for user from database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables for this user
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (f"{user_id}_%",))
            table_names = [row[0] for row in cursor.fetchall()]
            
            tables_info = []
            for table_name in table_names:
                if not table_name.endswith('_mapping'):  # Skip mapping tables
                    # Get column info
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns_info = cursor.fetchall()
                    
                    columns = []
                    for col_info in columns_info:
                        columns.append({
                            'column_name': col_info[1],
                            'data_type': col_info[2],
                            'purpose': 'data field',
                            'sample_values': []
                        })
                    
                    tables_info.append({
                        'table_name': table_name,
                        'columns': columns,
                        'metadata': []
                    })
            
            conn.close()
            print(f"Fallback found {len(tables_info)} tables")
            return tables_info
            
        except Exception as e:
            print(f"Fallback table discovery failed: {e}")
            return []
    
    def _phase2_generate_sql(self, question: str, relevant_metadata: List[Dict]) -> Optional[str]:
        """Phase 2: Generate SQL using relevant schema info"""
        
        print(f"Phase 2: Generating SQL for: {question}")
        
        # Build schema context from relevant metadata
        schema_context = "Available tables and columns:\n\n"
        
        for table_info in relevant_metadata:
            table_name = table_info['table_name']
            columns = table_info['columns']
            
            schema_context += f"Table: {table_name}\n"
            schema_context += "Columns:\n"
            
            for col in columns:
                col_name = col['column_name']
                data_type = col['data_type']
                purpose = col['purpose']
                samples = col.get('sample_values', [])
                
                schema_context += f"  - {col_name} ({data_type}): {purpose}\n"
                if samples:
                    schema_context += f"    Sample values: {', '.join(samples)}\n"
            
            schema_context += "\n"
        
        prompt = f"""Generate a SQL query to answer this question: {question}

{schema_context}

Rules:
1. Return ONLY the SQL query, no explanations
2. Use proper SQL syntax for SQLite
3. Use table names and column names exactly as shown
4. Based on the sample values, choose appropriate WHERE conditions
5. For questions about "open" items, look for status columns with values like 'open', 'Open', etc.
6. For questions about "tickets" or "issues", look for number/id columns
7. Only add LIMIT if user asks for "top N" or "first N"

SQL Query:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Generate accurate SQLite queries based on the provided schema and sample data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the query
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            print(f"Phase 2: Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            print(f"Phase 2: SQL generation failed: {e}")
            return None
    
    def _phase3_execute_and_format(self, question: str, sql_query: str, relevant_metadata: List[Dict]) -> ChatResponse:
        """Phase 3: Execute SQL and format results"""
        
        print(f"Phase 3: Executing SQL query")
        
        try:
            conn = sqlite3.connect(self.db_path)
            result_df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            if result_df.empty:
                return ChatResponse(
                    answer="No results found for your query.",
                    sources=[t['table_name'] for t in relevant_metadata]
                )
            
            # Format results
            formatted_answer = self._format_sql_results(question, result_df, sql_query)
            
            return ChatResponse(
                answer=formatted_answer,
                sources=[t['table_name'] for t in relevant_metadata]
            )
            
        except Exception as e:
            print(f"Phase 3: SQL execution failed: {e}")
            return ChatResponse(
                answer=f"Error executing query: {str(e)}",
                sources=[]
            )
    
    def _determine_column_purpose(self, col_name: str, sample_values: List[str], dtype: str) -> str:
        """Determine the purpose/meaning of a column"""
        
        col_lower = col_name.lower()
        
        # ID columns
        if any(word in col_lower for word in ['id', 'number', 'ticket', 'incident']):
            return "identifier or reference number"
        
        # Name columns
        elif any(word in col_lower for word in ['name', 'title', 'description']):
            return "name or descriptive text"
        
        # Status columns
        elif 'status' in col_lower:
            return "status or state information"
        
        # Date columns
        elif any(word in col_lower for word in ['date', 'time', 'created', 'updated']):
            return "date or timestamp"
        
        # Priority/Severity
        elif any(word in col_lower for word in ['priority', 'severity', 'level']):
            return "priority or severity level"
        
        # Financial
        elif any(word in col_lower for word in ['cost', 'budget', 'amount', 'price']):
            return "financial amount"
        
        # Percentage
        elif any(word in col_lower for word in ['percent', '%', 'completion']):
            return "percentage value"
        
        # Based on sample values
        elif sample_values:
            first_sample = sample_values[0].lower()
            if first_sample in ['open', 'closed', 'pending', 'resolved']:
                return "status information"
            elif first_sample in ['high', 'medium', 'low']:
                return "priority or level"
        
        return f"data field ({dtype})"
    
    def _format_sql_results(self, question: str, result_df: pd.DataFrame, sql_query: str) -> str:
        """Format SQL results with proper tables and spacing"""
        
        # Create markdown table format
        if len(result_df) > 50:
            display_df = result_df.head(50)
            truncated = True
        else:
            display_df = result_df
            truncated = False
        
        # Convert to markdown table
        if len(display_df.columns) <= 6:  # Use table format for reasonable column count
            table_str = self._df_to_markdown_table(display_df)
            
            answer = f"## 📊 Query Results\n\n"
            answer += f"**Found {len(result_df)} results**\n\n"
            
            answer += table_str
            
            if truncated:
                answer += f"\n*Showing first 50 of {len(result_df)} results*\n"
        else:
            # Too many columns, use list format
            answer = f"## 📋 Query Results\n\n"
            answer += f"**Found {len(result_df)} results**\n\n"
            
            for i, (_, row) in enumerate(display_df.iterrows(), 1):
                answer += f"**{i}.** "
                row_items = []
                for col, val in row.items():
                    if pd.notna(val) and str(val).strip():
                        row_items.append(f"{col}: {val}")
                answer += " • ".join(row_items) + "\n\n"
            
            if truncated:
                answer += f"*Showing first 50 of {len(result_df)} results*\n\n"
        
        # Add query info in collapsed section
        answer += f"\n<details>\n<summary>🔍 SQL Query Used</summary>\n\n```sql\n{sql_query}\n```\n</details>"
        
        return answer
    
    def _df_to_markdown_table(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to markdown table"""
        
        if df.empty:
            return "*No data to display*\n"
        
        # Create header
        headers = list(df.columns)
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        # Add rows
        for _, row in df.iterrows():
            row_values = []
            for val in row:
                if pd.isna(val):
                    row_values.append("-")
                else:
                    # Clean and truncate long values
                    val_str = str(val).strip()
                    if len(val_str) > 50:
                        val_str = val_str[:47] + "..."
                    # Escape pipe characters
                    val_str = val_str.replace("|", "\\|")
                    row_values.append(val_str)
            
            table += "| " + " | ".join(row_values) + " |\n"
        
        return table + "\n"
    
    def _get_cache_key(self, question: str, user_id: str) -> str:
        """Generate cache key for query"""
        cache_input = f"{question.lower().strip()}_{user_id}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: ChatResponse):
        """Cache query result"""
        self.query_cache[cache_key] = {
            "answer": result.answer,
            "sources": result.sources
        }
        self._save_query_cache()
    
    def _load_caches(self):
        """Load caches from files"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.query_cache = json.load(f)
        except Exception as e:
            print(f"Cache load error: {e}")
            self.query_cache = {}
    
    def _save_query_cache(self):
        """Save query cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.query_cache, f)
        except Exception as e:
            print(f"Cache save error: {e}")
    
    def clear_cache(self):
        """Clear all caches"""
        self.query_cache = {}
        self.metadata_cache = {}
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
        except Exception as e:
            print(f"Cache clear error: {e}")