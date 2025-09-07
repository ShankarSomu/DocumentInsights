from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class User(BaseModel):
    user_id: str
    email: str
    created_at: datetime = datetime.now()

class CSVUpload(BaseModel):
    file_name: str
    file_url: Optional[str] = None
    user_id: str

class ChatRequest(BaseModel):
    user_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    timestamp: datetime = datetime.now()

class DataChunk(BaseModel):
    content: str
    metadata: dict
    user_id: str