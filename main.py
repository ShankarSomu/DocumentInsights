from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from fastapi.staticfiles import StaticFiles
import os
from secure_config import SecureConfig

# Load configuration from secure SQLite storage
secure_config = SecureConfig()
try:
    config = secure_config.get_api_keys()
    if not config.get('GROQ_API_KEY'):
        raise Exception("No GROQ API key found in secure storage")
except Exception as e:
    print(f"Failed to load secure config: {e}")
    print("Please run 'python run.py' first to set up API keys")
    config = {}
from models import ChatRequest, ChatResponse, CSVUpload
from document_processor import DocumentProcessor
from local_vector_store import LocalVectorStore
from rag_service import RAGService
from schema_analyzer import SchemaAnalyzer
from session_manager import SessionManager
from metadata_service import MetadataService
from two_stage_query_service import TwoStageQueryService
from llm_driven_service import LLMDrivenService
import uuid

app = FastAPI(title="DocuSight", description="AI-Powered Document Analysis Assistant")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
doc_processor = DocumentProcessor()
vector_store = LocalVectorStore()
rag_service = RAGService()
schema_analyzer = SchemaAnalyzer()
session_manager = SessionManager()
metadata_service = MetadataService()
two_stage_service = TwoStageQueryService()
llm_service = LLMDrivenService()

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse('static/index.html')

@app.post("/upload-files")
async def upload_files(files: List[UploadFile] = File(...), user_id: str = None):
    if not user_id:
        user_id = str(uuid.uuid4())
    
    supported_types = ['csv', 'json', 'xml', 'txt', 'doc', 'docx', 'ppt', 'pptx', 'pdf', 'xls', 'xlsx']
    results = []
    total_chunks = 0
    
    for file in files:
        file_ext = file.filename.lower().split('.')[-1]
        
        if file_ext not in supported_types:
            results.append({"file": file.filename, "status": "error", "message": f"Unsupported file type: {file_ext}"})
            continue
        
        try:
            content = await file.read()
            chunks = doc_processor.process_file(content, user_id, file.filename)
            vector_store.store_chunks(chunks)
            
            results.append({"file": file.filename, "status": "success", "chunks": len(chunks)})
            total_chunks += len(chunks)
        except Exception as e:
            results.append({"file": file.filename, "status": "error", "message": str(e)})
    
    return {
        "message": f"Processed {len(files)} files",
        "total_chunks_created": total_chunks,
        "results": results,
        "user_id": user_id
    }

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), user_id: str = Form(None)):
    if not user_id:
        user_id = str(uuid.uuid4())
    
    print(f"Processing file {file.filename} for user_id: {user_id}")
    
    # Clean up old session data if this is a new session
    if session_manager.is_new_session(user_id):
        print(f"New session detected, cleaning up old data...")
        session_manager.cleanup_old_sessions(user_id)
    
    supported_types = ['csv', 'json', 'xml', 'txt', 'doc', 'docx', 'ppt', 'pptx', 'pdf', 'xls', 'xlsx']
    file_ext = file.filename.lower().split('.')[-1]
    
    if file_ext not in supported_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {', '.join(supported_types)}")
    
    try:
        # Read file content
        content = await file.read()
        print(f"Read {len(content)} bytes from {file.filename}")
        
        # Process file into chunks
        chunks = doc_processor.process_file(content, user_id, file.filename)
        print(f"Created {len(chunks)} chunks from {file.filename}")
        
        # Import DataChunk for SQL knowledge storage
        from models import DataChunk
        
        # Store in vector database
        try:
            vector_store.store_chunks(chunks)
            print(f"Stored chunks in vector store for {file.filename}")
        except Exception as e:
            print(f"Vector storage failed for {file.filename}: {e}")
            # Continue processing
        
        # Store CSV data in SQL for structured queries
        if file.filename.endswith('.csv'):
            try:
                import pandas as pd
                import io
                import os
                
                # Ensure directory exists
                os.makedirs("local_data", exist_ok=True)
                
                df = pd.read_csv(io.BytesIO(content))
                print(f"Parsed CSV with {len(df)} rows and {len(df.columns)} columns")
                
                # Normalize column names for consistent SQL
                original_columns = df.columns.tolist()
                df.columns = df.columns.str.strip()  # Remove spaces
                df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores
                df.columns = df.columns.str.replace('-', '_')  # Replace hyphens
                df.columns = df.columns.str.replace('%', '_pct')  # Replace % with _pct
                df.columns = df.columns.str.replace('(', '').str.replace(')', '')  # Remove parentheses
                df.columns = df.columns.str.replace('/', '_')  # Replace slashes
                df.columns = df.columns.str.replace('#', 'num_')  # Replace # with num_
                df.columns = df.columns.str.lower()  # Convert to lowercase
                normalized_columns = df.columns.tolist()
                
                print(f"Column normalization: {dict(zip(original_columns, normalized_columns))}")
                
                # Store in SQL database
                import sqlite3
                conn = sqlite3.connect("local_data/structured_data.db")
                table_name = f"{user_id}_{file.filename.replace('.csv', '').replace(' ', '_').replace('-', '_')}"
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
                # Store column mapping for LLM reference
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS column_mappings (
                        table_name TEXT,
                        original_column TEXT,
                        normalized_column TEXT,
                        column_type TEXT,
                        sample_values TEXT
                    )
                """)
                
                # Clear old mappings for this table
                cursor.execute("DELETE FROM column_mappings WHERE table_name = ?", (table_name,))
                
                # Store new mappings
                for orig, norm in zip(original_columns, normalized_columns):
                    sample_vals = df[norm].dropna().head(3).astype(str).tolist()
                    col_type = str(df[norm].dtype)
                    cursor.execute(
                        "INSERT INTO column_mappings VALUES (?, ?, ?, ?, ?)",
                        (table_name, orig, norm, col_type, str(sample_vals))
                    )
                
                conn.commit()
                conn.close()
                
                # Store SQL knowledge in vector store
                sql_examples = [
                    f"SELECT * FROM {table_name}",
                    f"SELECT COUNT(*) FROM {table_name}",
                    f"SELECT {', '.join(normalized_columns[:5])} FROM {table_name}"
                ]
                
                for sql in sql_examples:
                    sql_chunk = DataChunk(
                        content=f"SQL Query: {sql}\nTable: {table_name}\nColumns: {', '.join(normalized_columns)}",
                        user_id=user_id,
                        metadata={
                            "file_name": f"{file.filename}_sql_knowledge",
                            "type": "sql_reference",
                            "table_name": table_name,
                            "sql_query": sql
                        }
                    )
                    chunks.append(sql_chunk)
                
                print(f"Stored {file.filename} as SQL table {table_name}")
                
            except Exception as e:
                print(f"SQL storage failed for {file.filename}: {e}")
                import traceback
                traceback.print_exc()
                # Continue processing
        
        # Analyze and store column metadata
        try:
            metadata_service.analyze_and_store_metadata(content, file.filename, user_id)
            print(f"Metadata analysis completed for {file.filename}")
        except Exception as e:
            print(f"Metadata analysis failed for {file.filename}: {e}")
            import traceback
            traceback.print_exc()
            # Continue processing
        
        # Analyze file schema for context awareness
        try:
            sample_content = chunks[0].content if chunks else ""
            schema = schema_analyzer.analyze_file_schema(file.filename, sample_content, user_id)
            print(f"Schema analysis completed for {file.filename}")
        except Exception as e:
            print(f"Schema analysis failed for {file.filename}: {e}")
            schema = "unknown"
            # Continue processing
        
        print(f"Successfully completed processing {file.filename}")
        
        return {
            "message": f"Successfully processed {file.filename}",
            "chunks_created": len(chunks),
            "user_id": user_id,
            "schema": schema
        }
    
    except Exception as e:
        print(f"Critical error processing {file.filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Use LLM-driven service for intelligent query handling
        response = llm_service.process_query(request.question, request.user_id)
        return response
    except Exception as e:
        # Fallback to regular RAG service
        print(f"LLM service failed, using fallback: {e}")
        response = rag_service.generate_answer(request.question, request.user_id)
        return response

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)