import os
import hashlib
from typing import List, Dict, Any, Optional
import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
from models import DataChunk

class PineconeVectorStore:
    def __init__(self):
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment="us-east-1-aws"  # or your preferred environment
        )
        self.index_name = "projectiq-hybrid"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # Initialize index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize Pinecone index"""
        
        try:
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                print(f"Creating Pinecone index: {self.index_name}")
                
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine'
                )
            
            # Connect to index
            self.index = pinecone.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise
    
    def store_chunks(self, chunks: List[DataChunk]):
        """Store chunks in Pinecone with metadata"""
        
        if not chunks:
            return
        
        vectors_to_upsert = []
        
        for chunk in chunks:
            try:
                # Generate embedding
                embedding = self.get_embedding(chunk.content)
                
                # Create unique ID
                chunk_id = f"{chunk.user_id}_{chunk.chunk_id}"
                
                # Prepare metadata
                metadata = {
                    "user_id": chunk.user_id,
                    "file_name": chunk.file_name,
                    "content": chunk.content[:1000],  # Truncate for metadata limits
                    "chunk_id": chunk.chunk_id
                }
                
                # Add custom metadata if available
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    # Flatten metadata for Pinecone
                    for key, value in chunk.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"meta_{key}"] = value
                        elif isinstance(value, dict):
                            # Convert dict to string for storage
                            metadata[f"meta_{key}"] = str(value)
                
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding.tolist(),
                    "metadata": metadata
                })
                
            except Exception as e:
                print(f"Error processing chunk {chunk.chunk_id}: {e}")
                continue
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error upserting batch {i//batch_size + 1}: {e}")
        
        print(f"Stored {len(vectors_to_upsert)} chunks in Pinecone")
    
    def search(self, query: str, user_id: str, top_k: int = 20, file_filter: str = None) -> List[Dict]:
        """Search Pinecone with optional file filtering"""
        
        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            
            # Build filter
            filter_dict = {"user_id": {"$eq": user_id}}
            
            if file_filter:
                filter_dict["file_name"] = {"$eq": file_filter}
            
            # Search Pinecone
            search_results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=min(top_k, 1000),  # Pinecone limit
                include_metadata=True,
                filter=filter_dict
            )
            
            # Convert to expected format
            results = []
            for match in search_results.matches:
                result = {
                    "content": match.metadata.get("content", ""),
                    "file_name": match.metadata.get("file_name", "Unknown"),
                    "chunk_id": match.metadata.get("chunk_id", ""),
                    "score": match.score
                }
                
                # Add custom metadata
                for key, value in match.metadata.items():
                    if key.startswith("meta_"):
                        result[key[5:]] = value  # Remove "meta_" prefix
                
                results.append(result)
            
            print(f"Pinecone search found {len(results)} results for user {user_id}")
            if file_filter:
                print(f"Filtered to results from {file_filter}")
            
            return results
            
        except Exception as e:
            print(f"Pinecone search error: {e}")
            return []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        
        try:
            # Clean text
            text = str(text).strip()
            if not text:
                text = "empty"
            
            # Generate embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Embedding generation error: {e}")
            # Return zero vector as fallback
            return np.zeros(self.dimension)
    
    def delete_user_data(self, user_id: str):
        """Delete all data for a user"""
        
        try:
            # Pinecone doesn't support delete by metadata filter directly
            # We need to query first, then delete by IDs
            
            # Get all vectors for user (in batches)
            all_ids = []
            
            # Query with dummy vector to get all user data
            dummy_embedding = np.zeros(self.dimension)
            
            results = self.index.query(
                vector=dummy_embedding.tolist(),
                top_k=10000,  # Large number to get all
                include_metadata=True,
                filter={"user_id": {"$eq": user_id}}
            )
            
            all_ids = [match.id for match in results.matches]
            
            # Delete in batches
            batch_size = 1000
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                if batch_ids:
                    self.index.delete(ids=batch_ids)
            
            print(f"Deleted {len(all_ids)} vectors for user {user_id}")
            
        except Exception as e:
            print(f"Error deleting user data: {e}")
    
    def get_index_stats(self) -> Dict:
        """Get Pinecone index statistics"""
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}
    
    def list_user_files(self, user_id: str) -> List[str]:
        """List all files for a user"""
        
        try:
            # Query to get unique file names for user
            dummy_embedding = np.zeros(self.dimension)
            
            results = self.index.query(
                vector=dummy_embedding.tolist(),
                top_k=1000,
                include_metadata=True,
                filter={"user_id": {"$eq": user_id}}
            )
            
            file_names = set()
            for match in results.matches:
                file_name = match.metadata.get("file_name")
                if file_name and file_name != "Unknown":
                    file_names.add(file_name)
            
            return list(file_names)
            
        except Exception as e:
            print(f"Error listing user files: {e}")
            return []
    
    def cleanup_old_sessions(self, active_user_ids: List[str]):
        """Clean up data from inactive sessions"""
        
        try:
            # This is a simplified cleanup - in production you might want
            # more sophisticated session management
            print(f"Cleanup requested for active users: {active_user_ids}")
            
            # For now, we'll just log this - implementing full cleanup
            # would require querying all data and filtering
            
        except Exception as e:
            print(f"Error during cleanup: {e}")