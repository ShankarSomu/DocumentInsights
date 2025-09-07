# ProjectIQ - AI-Powered Project Management Assistant

An intelligent chat application that allows teams to interact with project management data using natural language queries. Features a fully LLM-driven architecture with intelligent query routing, parallel file processing, and adaptive data analysis.

## 🚀 Key Features

### **LLM-Driven Intelligence**
- **Zero Hardcoding**: LLM makes all decisions based on actual data structure
- **Adaptive Query Routing**: Automatically chooses SQL vs semantic search
- **Self-Healing Queries**: Auto-corrects SQL errors with multiple retry attempts
- **Context-Aware Processing**: Understands your specific data and schema

### **Advanced Data Processing**
- **Multi-format Support**: CSV, JSON, XML, DOC, PPT, PDF, XLS
- **Parallel Upload**: Multiple files processed simultaneously with progress tracking
- **Dual Storage**: SQLite for analytics + Local FAISS for semantic search
- **Text Normalization**: Handles variations in document titles and content
- **Session Management**: Clean data isolation per user session

### **Intelligent Query Processing**
- **Exploratory SQL**: LLM discovers data structure before generating queries
- **Silent Error Handling**: Seamless fallback without exposing technical errors
- **Schema Mapping**: Canonical data transformation with caching
- **Real-time Analysis**: Instant processing of uploaded files

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API key:
   # GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run the Server**
   ```bash
   python run.py
   ```

4. **Open Browser**
   ```
   http://localhost:8000
   ```

5. **Start Querying**
   - Upload your files (parallel processing with progress)
   - Ask questions in natural language
   - Get intelligent, context-aware responses

## 💬 Query Examples

### **Structured Analytics** (→ Auto-detected SQL)
```
"How many projects are at high risk?"
"List all employees with productivity > 85%"
"Show top 10 projects by budget"
"Count incidents by severity level"
"Which projects are behind schedule?"
```

### **Document Retrieval** (→ Auto-detected Semantic)
```
"Project Charter – Project Rhinestone"
"Show me the incident report for ticket #1234"
"Find the employee handbook section on policies"
"Get the project plan for Alpha initiative"
```

### **Semantic Understanding** (→ Auto-detected Semantic)
```
"Explain the main risks in our projects"
"What skills are most common in our team?"
"Describe the incident resolution process"
"Find projects similar to Project Alpha"
```

## 🔧 API Endpoints

### Upload Files
```bash
POST /upload-csv
Content-Type: multipart/form-data

# Form data:
# file: File to upload (CSV, Excel, PDF, etc.)
# user_id: User identifier (auto-generated if not provided)

# Response:
{
  "message": "Successfully processed filename.csv",
  "chunks_created": 150,
  "user_id": "user_abc123",
  "schema": "projects"
}
```

### Query Data
```bash
POST /chat
Content-Type: application/json

{
  "user_id": "user_abc123",
  "question": "Project Charter – Project Rhinestone"
}

# Response:
{
  "answer": "## Project Charter - Project Rhinestone\n\n**Project Manager:** John Doe\n**Status:** Active\n**Budget:** $2.5M...",
  "sources": ["Project Management Dataset.csv"]
}
```

### Health Check
```bash
GET /health
# Returns: {"status": "healthy"}
```

## 📁 Project Structure

```
DocumentInsights/
├── static/
│   └── index.html              # Frontend with parallel upload & progress
├── main.py                     # FastAPI application
├── run.py                      # Server runner
├── models.py                   # Data models
├── llm_driven_service.py       # LLM-driven query processing
├── text_normalizer.py          # Intelligent text normalization
├── local_vector_store.py       # Local FAISS vector storage
├── document_processor.py       # Multi-format file processing
├── session_manager.py          # User session management
├── metadata_service.py         # Column-level metadata analysis
├── schema_analyzer.py          # Schema analysis and mapping
├── requirements.txt            # Dependencies
├── local_data/                 # Local storage
│   ├── structured_data.db      # SQLite database
│   ├── text_mappings.json      # Text normalization cache
│   └── *_index.faiss          # Vector indexes per user
└── .env.example               # Environment template
```

## 🏗️ Architecture

### **LLM-Driven Query Processing**
```
User Query → LLM Analysis → Data Discovery → Query Generation
                        ↓
            SQL Engine (Analytics) ← LLM Router → Vector Engine (Semantics)
                        ↓
            Error Detection → Auto-Correction → Result Formatting
```

### **Technology Stack**
- **Frontend**: Vanilla JavaScript with parallel upload progress
- **Backend**: FastAPI with comprehensive error handling
- **LLM Engine**: Groq Llama-3.1-8B for all decision making
- **SQL Engine**: SQLite with exploratory data discovery
- **Vector Engine**: Local FAISS with text normalization
- **Processing**: Pandas + multi-format document parsing
- **Intelligence**: Zero hardcoding - LLM adapts to any data structure

## 🔑 API Keys Required

- **Groq API Key**: For LLM-driven intelligence (free tier: 14,400 requests/day)
- **No Vector DB Keys**: Uses local FAISS storage
- **No OpenAI Keys**: Uses local sentence-transformers
- **Fully Local**: Embeddings and data storage run locally

## 🔧 Advanced Features

### **LLM Intelligence**
- **Data Structure Discovery**: LLM explores your data before querying
- **Adaptive Schema Mapping**: Learns column meanings and relationships
- **Query Self-Correction**: Fixes SQL errors automatically with retries
- **Context-Aware Routing**: Routes based on actual available data

### **Parallel Processing**
- **Multi-file Upload**: Process multiple files simultaneously
- **Real-time Progress**: Visual progress tracking per file
- **Error Isolation**: Individual file failures don't stop others
- **Session Management**: Clean separation between user sessions

### **Text Intelligence**
- **Normalization**: Handles "–" vs "-" vs "for" variations automatically
- **Canonical Mapping**: Creates consistent document titles
- **Semantic Equivalents**: Understands related terms contextually
- **Learning Cache**: Builds mapping knowledge from your data

### **Error Resilience**
- **Silent SQL Healing**: Fixes database errors without user exposure
- **Multi-Retry Logic**: Attempts different approaches automatically
- **Graceful Fallbacks**: Switches between SQL and semantic seamlessly
- **Comprehensive Logging**: Detailed backend logs for troubleshooting

## 🚀 Performance Features

### **Smart Caching**
- **Query Result Cache**: Instant responses for repeated questions
- **Text Mapping Cache**: Learns document title variations
- **Schema Cache**: Remembers data structure discoveries
- **Session Isolation**: Clean cache separation per user

### **Optimized Storage**
- **Dual Engine**: SQL for analytics, Vector for understanding
- **Local Processing**: No external API dependencies for data
- **Efficient Indexing**: FAISS for fast similarity search
- **Batch Processing**: Optimized chunk storage and retrieval

## 🔧 Troubleshooting

### **Common Issues**
- **No Groq API Key**: Set `GROQ_API_KEY` in `.env` file
- **Slow First Run**: Downloads embedding model (~90MB)
- **Upload Failures**: Check file format and size limits
- **Query Errors**: System auto-corrects, check logs for details

### **Performance Tips**
- **Upload Related Files**: Keep project data in same session
- **Use Descriptive Names**: Clear file names help LLM routing
- **Parallel Uploads**: Upload multiple files simultaneously
- **Session Management**: Start new session for different projects

### **Debug Information**
Check server console for:
- LLM decision making process
- SQL query generation and correction
- File processing progress
- Error handling and retries

---

**Built with ❤️ for intelligent data interaction**

*Transform your files into an intelligent, queryable knowledge base with zero configuration.*