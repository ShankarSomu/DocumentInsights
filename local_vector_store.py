import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from text_normalizer import TextNormalizer
from typing import List, Dict
from models import DataChunk

class LocalVectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_normalizer = TextNormalizer()
        self.dimension = 384
        self.data_dir = "local_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # User-specific storage
        self.user_indexes = {}
        self.user_metadata = {}
    
    def _get_user_files(self, user_id: str):
        index_file = os.path.join(self.data_dir, f"{user_id}_index.faiss")
        metadata_file = os.path.join(self.data_dir, f"{user_id}_metadata.pkl")
        return index_file, metadata_file
    
    def _load_user_data(self, user_id: str):
        if user_id in self.user_indexes:
            return
        
        index_file, metadata_file = self._get_user_files(user_id)
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            self.user_indexes[user_id] = faiss.read_index(index_file)
            with open(metadata_file, 'rb') as f:
                self.user_metadata[user_id] = pickle.load(f)
        else:
            self.user_indexes[user_id] = faiss.IndexFlatIP(self.dimension)
            self.user_metadata[user_id] = []
    
    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text)
    
    def store_chunks(self, chunks: List[DataChunk]):
        if not chunks:
            return
        
        user_id = chunks[0].user_id
        self._load_user_data(user_id)
        
        embeddings = []
        metadata = []
        
        for chunk in chunks:
            embedding = self.get_embedding(chunk.content)
            embeddings.append(embedding)
            metadata.append({
                "content": chunk.content,
                "user_id": chunk.user_id,
                **chunk.metadata
            })
        
        # Add to FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        self.user_indexes[user_id].add(embeddings_array)
        
        # Store metadata
        self.user_metadata[user_id].extend(metadata)
        
        # Save to disk
        self._save_user_data(user_id)
        print(f"Stored {len(chunks)} chunks locally for user {user_id}")
    
    def _save_user_data(self, user_id: str):
        index_file, metadata_file = self._get_user_files(user_id)
        
        faiss.write_index(self.user_indexes[user_id], index_file)
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.user_metadata[user_id], f)
    
    def search(self, query: str, user_id: str, top_k: int = 20, file_filter: str = None) -> List[Dict]:
        self._load_user_data(user_id)
        
        if user_id not in self.user_indexes or self.user_indexes[user_id].ntotal == 0:
            print(f"No data found for user {user_id}")
            return []
        
        # Special handling for wildcard search to check if user has any data
        if query == "*":
            return self.user_metadata[user_id][:top_k] if self.user_metadata[user_id] else []
        
        # For large requests, return all data directly (faster)
        if top_k > 100:
            print(f"Returning all {len(self.user_metadata[user_id])} results directly")
            all_results = self.user_metadata[user_id].copy()
            
            # Apply file filter if specified
            if file_filter:
                filtered_results = [r for r in all_results if r.get('file_name', '') == file_filter]
                print(f"Filtered to {len(filtered_results)} results from {file_filter}")
                return filtered_results
            
            return all_results
        
        # Get query variations for better matching
        query_variations = self.text_normalizer.get_search_variations(query)
        print(f"Searching with variations: {query_variations[:3]}...")  # Show first 3
        
        all_results = {}
        
        # Search with each variation
        for variation in query_variations:
            query_embedding = self.get_embedding(variation).astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Regular search for smaller requests
            actual_top_k = min(top_k, self.user_indexes[user_id].ntotal)
            scores, indices = self.user_indexes[user_id].search(query_embedding, actual_top_k)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # Valid index
                    result = self.user_metadata[user_id][idx]
                    
                    # Apply file filter if specified
                    if file_filter and result.get('file_name', '') != file_filter:
                        continue
                    
                    # Use index as key to avoid duplicates, keep best score
                    if idx not in all_results or score > all_results[idx]['score']:
                        result_copy = result.copy()
                        result_copy['score'] = score
                        all_results[idx] = result_copy
        
        # Convert to list and sort by score
        results = list(all_results.values())
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Remove score from final results and limit
        final_results = []
        for result in results[:top_k]:
            clean_result = result.copy()
            clean_result.pop('score', None)
            final_results.append(clean_result)
        
        print(f"Local search found {len(final_results)} results for user {user_id}")
        if file_filter:
            print(f"Filtered to results from {file_filter}")
        
        return final_results