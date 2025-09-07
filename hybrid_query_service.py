import sqlite3
import pandas as pd
import hashlib
import json
import os
from typing import Dict, List, Any, Optional
from groq import Groq
from local_vector_store import LocalVectorStore
from query_router import QueryRouter
from schema_mapper import SchemaMapper
from three_phase_sql_service import ThreePhaseSQLService
from models import ChatResponse

class HybridQueryService:
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
        self.query_router = QueryRouter()
        self.schema_mapper = SchemaMapper()
        self.three_phase_sql = ThreePhaseSQLService()
        self.cache = {}
        self.db_path = "local_data/structured_data.db"
        self.cache_file = "local_data/query_cache.json"
        self.data_version = 1
        
        # Ensure directories exist
        os.makedirs("local_data", exist_ok=True)
        self._load_cache()
    
    def process_query(self, question: str, user_id: str) -> ChatResponse:
        """Route query to SQL or Vector Store based on query type"""
        
        # Check cache first
        cache_key = self._get_cache_key(question, user_id)
        if cache_key in self.cache:
            print("*** USING CACHED RESULT ***")
            cached_result = self.cache[cache_key]
            return ChatResponse(answer=cached_result['answer'], sources=cached_result['sources'])
        
        # Determine query type using LLM and metadata
        query_type = self._classify_query(question, user_id)
        print(f"*** HYBRID QUERY TYPE: {query_type} ***")
        
        if query_type == "structured":
            # Use 3-phase SQL processing
            result = self.three_phase_sql.process_sql_query(question, user_id)
        else:
            # Use vector store for semantic queries
            result = self._process_semantic_query(question, user_id)
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        return result
    
    def store_csv_data(self, file_content: bytes, file_name: str, user_id: str):
        """Store CSV data with LLM-assisted schema mapping"""
        
        try:
            import io
            df = pd.read_csv(io.BytesIO(file_content))
            
            # Determine domain for schema mapping
            domain = self._determine_domain(file_name)
            
            # Apply LLM-assisted schema mapping
            print(f"Applying LLM schema mapping for {file_name} (domain: {domain})")
            mapping = self.schema_mapper.map_file_to_canonical(df, file_name, domain)
            
            # Transform DataFrame to canonical schema
            df_canonical = self.schema_mapper.apply_mapping(df, mapping)
            
            # Store canonical data in SQL database
            table_name = self._store_in_sql(df_canonical, file_name, user_id, mapping)
            
            # Phase 1: Ingest table metadata into vector store
            self.three_phase_sql.ingest_table_metadata(df_canonical, table_name, user_id)
            
            # Also store original data in vector store for semantic queries
            self._store_in_vector(df, file_name, user_id)
            
            # Increment data version to invalidate cache
            self.data_version += 1
            self._clear_cache()
            
            print(f"Stored {file_name} with canonical mapping (confidence: {mapping.get('confidence', 'unknown')})")
            
        except Exception as e:
            print(f"Error storing {file_name}: {e}")
    
    def _classify_query(self, question: str, user_id: str) -> str:
        """Use LLM to classify query based on available metadata"""
        
        # Get available tables and their metadata
        available_tables = self._get_user_tables(user_id)
        
        if not available_tables:
            return "semantic"  # No structured data available
        
        # Get sample of available data structure
        metadata_context = self._get_classification_context(available_tables)
        
        prompt = f"""Analyze this user question and determine the best approach.

User Question: "{question}"

Available Data Structure:
{metadata_context}

CRITICAL ANALYSIS:

1. Is the user asking for a SPECIFIC DOCUMENT/CONTENT by name?
   - Examples: "Project Charter - ABC", "Report XYZ", "Document ABC"
   - If YES → Use SEMANTIC search to find the actual document content

2. Is the user asking for DATA ANALYSIS/CALCULATIONS?
   - Examples: "How many projects", "Total cost", "List all projects with status X"
   - If YES → Use STRUCTURED SQL queries

For the question "{question}":
- Does it mention a specific document name? 
- Is it asking for the content/text of that document?
- Or is it asking for data analysis/calculations?

IMPORTANT: If the question contains a specific document title or name (like "Project Charter - ProjectName"), this is a SEMANTIC request for document content, NOT a structured data query.

Return only: STRUCTURED or SEMANTIC"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a query routing expert. Analyze the question and data to choose the best approach."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            classification = response.choices[0].message.content.strip().upper()
            print(f"LLM Classification Decision: {classification}")
            
            if "STRUCTURED" in classification:
                return "structured"
            elif "SEMANTIC" in classification:
                return "semantic"
            else:
                # Fallback to semantic if unclear
                print("LLM classification unclear, defaulting to semantic")
                return "semantic"
                
        except Exception as e:
            print(f"LLM classification failed: {e}")
            # Fallback to simple heuristic
            question_lower = question.lower()
            
            # Check for document title patterns first
            if any(pattern in question_lower for pattern in ["charter", "document", "report"]) and \
               any(pattern in question_lower for pattern in ["project", "-", "–"]):
                print("Fallback: Detected document title pattern -> semantic")
                return "semantic"
            elif any(word in question_lower for word in ["count", "total", "sum", "average", "list all"]):
                return "structured"
            else:
                return "semantic"
    
    def _process_sql_query(self, question: str, user_id: str) -> ChatResponse:
        """Process structured queries using SQL"""
        
        # Get available tables for this user
        tables = self._get_user_tables(user_id)
        
        if not tables:
            return ChatResponse(
                answer="No structured data available. Please upload CSV files first.",
                sources=[]
            )
        
        # Generate SQL query using LLM
        sql_query = self._generate_sql(question, tables, user_id)
        
        if not sql_query:
            return ChatResponse(
                answer="Could not generate SQL query for this question.",
                sources=[]
            )
        
        # Execute SQL query
        try:
            conn = sqlite3.connect(self.db_path)
            result_df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # Format results using LLM
            formatted_answer = self._format_sql_results(question, result_df, sql_query)
            
            return ChatResponse(
                answer=formatted_answer,
                sources=tables
            )
            
        except Exception as e:
            print(f"SQL execution error: {e}")
            return ChatResponse(
                answer=f"Error executing query: {str(e)}",
                sources=[]
            )
    
    def _process_semantic_query(self, question: str, user_id: str) -> ChatResponse:
        """Process semantic queries using vector store"""
        
        # Use existing vector store logic
        chunks = self.vector_store.search(question, user_id, top_k=10)
        
        if not chunks:
            return ChatResponse(
                answer="No relevant information found in the vector store.",
                sources=[]
            )
        
        # Build context and generate answer
        context = self._build_context(chunks)
        
        # Check if this is a specific document request
        if any(word in question.lower() for word in ["charter", "document", "report", "content"]):
            prompt = f"""The user is asking for a specific document or content: {question}

Available data:
{context}

If you find the requested document or content in the data, provide it in full. If not found, clearly state that the document is not available in the uploaded data.

Response:"""
        else:
            prompt = f"""Based on the following data, answer this question: {question}

Data Context:
{context}

Provide a clear, concise answer based only on the provided data."""

        try:
            # Adjust max tokens for document requests
            max_tokens = 1000 if "charter" in question.lower() or "document" in question.lower() else 500
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a document retrieval assistant. Provide exact content when requested, or analyze data when asked."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            raw_answer = response.choices[0].message.content
            
            # Format the answer with better structure
            formatted_answer = f"## 🤖 AI Analysis\n\n{raw_answer}\n\n"
            formatted_answer += f"**📁 Sources:** {', '.join(list(set([chunk.get('file_name', 'Unknown') for chunk in chunks])))}"
            
            sources = list(set([chunk.get('file_name', 'Unknown') for chunk in chunks]))
            
            return ChatResponse(answer=formatted_answer, sources=sources)
            
        except Exception as e:
            print(f"Semantic query error: {e}")
            return ChatResponse(
                answer=f"Error processing semantic query: {str(e)}",
                sources=[]
            )
    
    def _store_in_sql(self, df: pd.DataFrame, file_name: str, user_id: str, mapping: Dict = None) -> str:
        """Store DataFrame in SQL database with canonical schema"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Create table name from file name and user ID
        table_name = f"{user_id}_{file_name.replace('.csv', '').replace(' ', '_').replace('-', '_')}"
        
        # Store DataFrame with canonical columns
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Store mapping metadata in a separate table
        if mapping:
            mapping_table = f"{table_name}_mapping"
            mapping_df = pd.DataFrame([{
                'file_name': file_name,
                'domain': mapping.get('domain'),
                'confidence': mapping.get('confidence'),
                'mapping_json': json.dumps(mapping)
            }])
            mapping_df.to_sql(mapping_table, conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"Stored {file_name} as canonical table {table_name}")
        
        return table_name
    
    def _get_classification_context(self, tables: List[str]) -> str:
        """Get context about available data for LLM classification"""
        
        context = "Available Tables:\n"
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            for table in tables[:3]:  # Limit to first 3 tables
                cursor = conn.cursor()
                
                # Get column info
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                context += f"\n{table} ({row_count} rows):\n"
                context += "  Columns: " + ", ".join([col[1] for col in columns]) + "\n"
            
            conn.close()
            
        except Exception as e:
            context += f"Error getting table info: {e}\n"
        
        return context
    
    def _store_in_vector(self, df: pd.DataFrame, file_name: str, user_id: str):
        """Store DataFrame in vector store for semantic search"""
        
        from models import DataChunk
        
        chunks = []
        for idx, row in df.iterrows():
            # Create text representation of the row
            row_text = f"From {file_name}:\n"
            for col, val in row.items():
                if pd.notna(val):
                    row_text += f"{col}: {val}\n"
            
            chunk = DataChunk(
                content=row_text.strip(),
                user_id=user_id,
                file_name=file_name,
                chunk_id=f"{file_name}_{idx}",
                metadata={"source_file": file_name, "row_index": idx}
            )
            chunks.append(chunk)
        
        self.vector_store.store_chunks(chunks)
    
    def _generate_sql(self, question: str, tables: List[str], user_id: str) -> Optional[str]:
        """Generate SQL query using LLM"""
        
        # Get table schemas
        schemas = self._get_table_schemas(tables)
        
        prompt = f"""Generate a SQL query to answer this question: {question}

Available tables and their schemas:
{schemas}

Rules:
1. Return ONLY the SQL query, no explanations
2. Use proper SQL syntax for SQLite
3. Use table names exactly as shown
4. Use column names exactly as they appear in the schema
5. Look at the sample data to understand what columns contain
6. Only add LIMIT if the user specifically asks for "top N" or "first N"
7. If looking for issues/tickets, check the incident table
8. If looking for projects, check the project management table

SQL Query:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Generate only valid SQLite queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the query
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            print(f"Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            print(f"SQL generation error: {e}")
            return None
    
    def _get_user_tables(self, user_id: str) -> List[str]:
        """Get list of tables for a user"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (f"{user_id}_%",))
            tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return tables
            
        except Exception as e:
            print(f"Error getting tables: {e}")
            return []
    
    def _get_table_schemas(self, tables: List[str]) -> str:
        """Get detailed schema information with sample data for tables"""
        
        schemas = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            for table in tables:
                cursor = conn.cursor()
                
                # Get column info
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                # Get sample data
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample_rows = cursor.fetchall()
                
                schema = f"Table: {table}\n"
                schema += "Columns: " + ", ".join([f"{col[1]} ({col[2]})" for col in columns]) + "\n"
                
                if sample_rows:
                    schema += "Sample data:\n"
                    column_names = [col[1] for col in columns]
                    for i, row in enumerate(sample_rows):
                        schema += f"Row {i+1}: "
                        for j, value in enumerate(row):
                            if j < len(column_names):
                                schema += f"{column_names[j]}='{value}', "
                        schema = schema.rstrip(", ") + "\n"
                
                schemas.append(schema)
            
            conn.close()
            
        except Exception as e:
            print(f"Error getting schemas: {e}")
        
        return "\n\n".join(schemas)
    
    def _determine_domain(self, file_name: str) -> str:
        """Determine domain from file name"""
        
        file_lower = file_name.lower()
        
        if any(word in file_lower for word in ['project', 'management', 'milestone']):
            return "projects"
        elif any(word in file_lower for word in ['employee', 'staff', 'hr', 'team']):
            return "employees"
        elif any(word in file_lower for word in ['incident', 'issue', 'ticket', 'bug']):
            return "incidents"
        else:
            return "general"
    
    def _format_sql_results(self, question: str, result_df: pd.DataFrame, sql_query: str) -> str:
        """Format SQL results using LLM"""
        
        if result_df.empty:
            return "No results found for your query."
        
        # Convert DataFrame to string representation
        if len(result_df) > 50:
            data_str = result_df.head(50).to_string(index=False)
            data_str += f"\n\n... and {len(result_df) - 50} more rows (showing first 50)"
        else:
            data_str = result_df.to_string(index=False)
        
        prompt = f"""Format these SQL query results to answer the user's question.

User Question: {question}
SQL Query: {sql_query}

Results:
{data_str}

Provide a clear, formatted answer with the key findings."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Format SQL results clearly and professionally."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Result formatting error: {e}")
            return f"Query returned {len(result_df)} results:\n\n{data_str}"
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from vector store chunks"""
        
        context_parts = []
        for chunk in chunks[:5]:  # Limit to 5 chunks
            context_parts.append(f"From {chunk.get('file_name', 'Unknown')}:\n{chunk['content']}")
        
        return "\n\n".join(context_parts)
    
    def _get_cache_key(self, question: str, user_id: str) -> str:
        """Generate cache key for query"""
        
        cache_input = f"{question}_{user_id}_{self.data_version}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: ChatResponse):
        """Cache query result"""
        
        self.cache[cache_key] = {
            "answer": result.answer,
            "sources": result.sources
        }
        
        # Save cache to file
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Cache save error: {e}")
    
    def _load_cache(self):
        """Load cache from file"""
        
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
        except Exception as e:
            print(f"Cache load error: {e}")
            self.cache = {}
    
    def _clear_cache(self):
        """Clear cache when data changes"""
        
        self.cache = {}
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
        except Exception as e:
            print(f"Cache clear error: {e}")