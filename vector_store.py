from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
from models import DataChunk

class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "projectiq-index"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # all-MiniLM-L6-v2 embedding dimension
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
        
        self.index = self.pc.Index(self.index_name)
    
    def get_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def store_chunks(self, chunks: List[DataChunk]):
        vectors = []
        user_id = chunks[0].user_id
        print(f"Storing {len(chunks)} chunks for user_id: {user_id}")
        
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk.content)
            vector_id = f"{chunk.user_id}_{i}_{abs(hash(chunk.content))}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "content": chunk.content,
                    "user_id": chunk.user_id,
                    **chunk.metadata
                }
            })
            print(f"Created vector {vector_id} with content preview: {chunk.content[:100]}...")
        
        self.index.upsert(vectors=vectors, namespace=user_id)
        print(f"Upserted {len(vectors)} vectors to namespace {user_id}")
    
    def search(self, query: str, user_id: str, top_k: int = 20) -> List[Dict]:
        query_embedding = self.get_embedding(query)
        
        # Check if namespace has data
        stats = self.index.describe_index_stats()
        print(f"Index stats: {stats}")
        print(f"Searching for user_id: {user_id}")
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=user_id,
            include_metadata=True
        )
        
        print(f"Search results: {len(results['matches'])} matches found")
        
        # If no matches found, try getting all vectors from namespace
        if not results['matches']:
            print("No vector matches found, trying to fetch all data from namespace")
            try:
                # Query with a very low threshold to get any data
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,  # Use requested top_k
                    namespace=user_id,
                    include_metadata=True,
                    filter={}  # No filter to get all data
                )
                print(f"Fallback search found: {len(results['matches'])} matches")
            except Exception as e:
                print(f"Fallback search failed: {e}")
        
        if results['matches']:
            print(f"First match score: {results['matches'][0]['score']}")
        
        return [match['metadata'] for match in results['matches']]