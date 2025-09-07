import pandas as pd
from typing import List
from models import DataChunk
import io

class CSVProcessor:
    def __init__(self, chunk_size: int = 5):
        self.chunk_size = chunk_size
    
    def process_csv(self, file_content: bytes, user_id: str, file_name: str) -> List[DataChunk]:
        df = pd.read_csv(io.BytesIO(file_content))
        chunks = []
        
        # Process in chunks of rows
        for i in range(0, len(df), self.chunk_size):
            chunk_df = df.iloc[i:i+self.chunk_size]
            
            # Convert chunk to readable text
            content = self._dataframe_to_text(chunk_df)
            
            metadata = {
                "file_name": file_name,
                "chunk_index": i // self.chunk_size,
                "row_start": i,
                "row_end": min(i + self.chunk_size - 1, len(df) - 1),
                "columns": list(df.columns)
            }
            
            chunks.append(DataChunk(
                content=content,
                metadata=metadata,
                user_id=user_id
            ))
        
        return chunks
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        text_parts = []
        for _, row in df.iterrows():
            row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            text_parts.append(row_text)
        return "\n".join(text_parts)