# ProjectIQ Architecture Document

## Overview

ProjectIQ is an intelligent document analysis and query system that leverages Large Language Models (LLMs) to provide adaptive, context-aware data processing without hardcoded rules. The system automatically routes queries between SQL analytics and semantic search based on user intent and available data structure.

## Key Architectural Principles

### 1. **Zero Hardcoding Philosophy**
- No predefined keywords, patterns, or rules
- LLM makes all routing and processing decisions
- Adaptive to any data structure or domain
- Self-learning from actual data content

### 2. **Dual Storage Strategy**
- **SQLite**: Structured analytics, aggregations, filtering
- **FAISS Vector Store**: Semantic search, document retrieval
- **Automatic Routing**: LLM decides which engine to use

### 3. **Error Resilience**
- Silent error handling with multiple retry attempts
- Automatic SQL query correction
- Graceful fallbacks between engines
- No technical errors exposed to users

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  • Parallel File Upload with Progress Tracking                 │
│  • Real-time Chat Interface                                    │
│  • Session Management                                          │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                           │
├─────────────────────────────────────────────────────────────────┤
│  • File Upload Endpoint (/upload-csv)                         │
│  • Chat Endpoint (/chat)                                      │
│  • Health Check (/health)                                     │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM-Driven Service Layer                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │   Query Router  │    │  Data Explorer  │                   │
│  │   (LLM-based)   │    │   (LLM-based)   │                   │
│  └─────────────────┘    └─────────────────┘                   │
│           │                       │                            │
│           ▼                       ▼                            │
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │ SQL Generator   │    │ Search Strategy │                   │
│  │  (LLM-based)    │    │  (LLM-based)    │                   │
│  └─────────────────┘    └─────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   SQL Engine    │ │  Vector Engine  │ │ Text Normalizer │
│   (SQLite)      │ │    (FAISS)      │ │  (LLM-based)    │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ • Structured    │ │ • Semantic      │ │ • Canonical     │
│   Analytics     │ │   Search        │ │   Mapping       │
│ • Aggregations  │ │ • Document      │ │ • Variation     │
│ • Filtering     │ │   Retrieval     │ │   Handling      │
│ • Joins         │ │ • Similarity    │ │ • Learning      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Detailed Component Flow

### 1. File Upload Process

```
User Uploads Files
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Parallel Processing                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  File 1 ──┐                                                    │
│           │    ┌─────────────────┐    ┌─────────────────┐      │
│  File 2 ──┼───▶│ Document        │───▶│ Chunk Creation  │      │
│           │    │ Processor       │    │                 │      │
│  File N ──┘    └─────────────────┘    └─────────────────┘      │
│                                                │                │
└────────────────────────────────────────────────┼────────────────┘
                                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Dual Storage                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐              ┌─────────────────┐          │
│  │   SQL Storage   │              │ Vector Storage  │          │
│  │                 │              │                 │          │
│  │ • CSV → Table   │              │ • Text Chunks   │          │
│  │ • Schema Map    │              │ • Embeddings    │          │
│  │ • Metadata      │              │ • Metadata      │          │
│  └─────────────────┘              └─────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Text Normalization                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • LLM analyzes text variations                                │
│  • Creates canonical mappings                                  │
│  • Builds semantic equivalents                                 │
│  • Caches learned patterns                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Query Processing Flow

```
User Query: "Project Charter – Project Rhinestone"
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Query Analysis                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ LLM Analyzes:                                               ││
│  │ • User intent and expected output                           ││
│  │ • Available data structure (tables, files)                 ││
│  │ • Query complexity and requirements                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Routing Decision                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Decision: "User wants specific document content" → SEMANTIC    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ STRUCTURED      │ │   SEMANTIC      │ │    HYBRID       │
│ (SQL Path)      │ │ (Vector Path)   │ │   (Both)        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
            │               │               │
            ▼               ▼               ▼
```

### 3. SQL Processing Path (Structured Queries)

```
LLM Decision: STRUCTURED
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Data Exploration                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM Generates Exploration Queries:                            │
│  • SELECT * FROM table LIMIT 5                                 │
│  • SELECT DISTINCT column FROM table                           │
│  • PRAGMA table_info(table)                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SQL Generation                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM Generates Query Based on:                                 │
│  • Exploration results (actual data)                           │
│  • Schema information (column names, types)                    │
│  • User question intent                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Error Handling & Retry                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Attempt 1: Execute generated SQL                              │
│      ↓ (if error)                                             │
│  Attempt 2: LLM fixes the error                               │
│      ↓ (if error)                                             │
│  Attempt 3: LLM tries different approach                      │
│      ↓ (if all fail)                                          │
│  Fallback: Switch to semantic search                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Result Formatting                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM Formats Results:                                          │
│  • Creates tables/lists as appropriate                         │
│  • Adds insights and analysis                                  │
│  • Hides technical SQL details                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Semantic Processing Path (Document Retrieval)

```
LLM Decision: SEMANTIC
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Search Strategy                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM Determines:                                               │
│  • Optimal search query variations                             │
│  • Number of results to retrieve                               │
│  • File filtering (if specific document)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                Text Normalization                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query: "Project Charter – Project Rhinestone"                 │
│  Generates Variations:                                         │
│  • "project charter project rhinestone"                        │
│  • "Project Charter - Project Rhinestone"                      │
│  • "project charter for project rhinestone"                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                Vector Search                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • Search with all query variations                            │
│  • Combine and deduplicate results                             │
│  • Rank by relevance score                                     │
│  • Apply file filtering if specified                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│               Content Processing                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM Processes Retrieved Content:                              │
│  • Extracts relevant information                               │
│  • Formats for user presentation                               │
│  • Provides full document if requested                         │
│  • Cites sources appropriately                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Enhancements & Fixes

### 1. **Eliminated Hardcoding**

**Problem**: System relied on hardcoded keywords and patterns
```python
# OLD: Hardcoded patterns
structured_keywords = ["count", "total", "sum", "average"]
document_keywords = ["charter", "document", "report"]
```

**Solution**: LLM-based decision making
```python
# NEW: LLM analyzes context
def _llm_decide_approach(self, question: str, user_id: str) -> str:
    # LLM analyzes question + available data structure
    # Returns "structured" or "semantic" based on context
```

**Benefits**:
- Adapts to any domain or data type
- No maintenance of keyword lists
- Handles edge cases intelligently

### 2. **Silent Error Handling**

**Problem**: SQL errors exposed to users, breaking user experience
```
Error: no such column: completion
```

**Solution**: Multi-retry system with automatic correction
```python
def _execute_sql_with_retries(self, sql_query: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return execute_sql(sql_query)
        except Exception as e:
            sql_query = self._llm_fix_sql(sql_query, error=e)
    # Fallback to semantic search
```

**Benefits**:
- Users never see technical errors
- Automatic query correction
- Graceful fallbacks maintain functionality

### 3. **Exploratory SQL Generation**

**Problem**: LLM generated SQL with non-existent columns
```sql
-- Generated without knowing actual data
SELECT completion FROM projects WHERE completion > 50
-- Error: column 'completion' doesn't exist
```

**Solution**: Data exploration before query generation
```python
def _llm_explore_data(self, question: str, schema_info: str):
    # LLM generates exploration queries
    exploration_queries = [
        "SELECT * FROM table LIMIT 5",
        "SELECT DISTINCT column_name FROM table"
    ]
    # Provides actual data context to LLM
```

**Benefits**:
- Accurate column names and values
- Context-aware query generation
- Reduced SQL errors

### 4. **Parallel File Upload**

**Problem**: Sequential file uploads were slow with no progress feedback

**Solution**: Parallel processing with real-time progress
```javascript
// Upload files in parallel
const uploadPromises = files.map(file => uploadSingleFile(file));
const results = await Promise.all(uploadPromises);

// Real-time progress updates
function updateFileProgress(index, fileName, status) {
    // Visual progress indicators
}
```

**Benefits**:
- Faster upload times
- Real-time progress feedback
- Better user experience

### 5. **Text Normalization System**

**Problem**: Queries for "Project Charter – Project X" didn't match "Project Charter - Project X"

**Solution**: LLM-driven text normalization
```python
class TextNormalizer:
    def get_canonical_form(self, text: str) -> str:
        # LLM creates canonical mappings
        # Handles punctuation variations
        # Builds semantic equivalents
```

**Benefits**:
- Handles document title variations
- Learns from actual data patterns
- Improves search recall

### 6. **Dual Storage Architecture**

**Problem**: Single storage couldn't handle both analytics and document retrieval efficiently

**Solution**: Specialized storage for different query types
```
CSV Data → SQLite (for analytics: COUNT, SUM, GROUP BY)
         → FAISS (for semantics: document retrieval, similarity)
```

**Benefits**:
- Optimal performance for each query type
- LLM routes to appropriate storage
- Maintains data consistency

## Performance Optimizations

### 1. **Smart Caching**
- Query result caching with versioning
- Text mapping cache for normalization
- Schema discovery cache
- Session-based cache isolation

### 2. **Efficient Processing**
- Parallel file upload and processing
- Batch operations for vector storage
- Lazy loading of user data
- Connection pooling for database

### 3. **Memory Management**
- Streaming file processing
- Chunked data handling
- Automatic cleanup of old sessions
- Efficient embedding storage

## Security & Isolation

### 1. **User Session Management**
- Unique user IDs for each session
- Data isolation between users
- Automatic cleanup of old sessions
- No cross-user data leakage

### 2. **Error Handling**
- Comprehensive exception handling
- Graceful degradation
- No sensitive information in error messages
- Detailed logging for debugging

### 3. **Input Validation**
- File type validation
- Content sanitization
- SQL injection prevention
- Rate limiting capabilities

## Monitoring & Debugging

### 1. **Comprehensive Logging**
- LLM decision tracking
- SQL query generation and execution
- File processing progress
- Error occurrence and resolution

### 2. **Performance Metrics**
- Query response times
- File processing speeds
- Cache hit rates
- Error rates by component

### 3. **Debug Information**
- LLM reasoning traces
- SQL query evolution
- Vector search results
- Normalization mappings

## Future Enhancements

### 1. **Advanced LLM Features**
- Multi-model ensemble for better decisions
- Fine-tuning on domain-specific data
- Advanced reasoning capabilities
- Custom model integration

### 2. **Scalability Improvements**
- Distributed processing
- Cloud storage integration
- Load balancing
- Horizontal scaling

### 3. **Enhanced Analytics**
- Advanced visualization
- Predictive analytics
- Trend analysis
- Custom reporting

---

This architecture provides a robust, intelligent, and adaptive system that can handle diverse data types and query patterns without hardcoded limitations, ensuring excellent user experience and maintainable codebase.