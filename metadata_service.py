import json
import os
import pandas as pd
from typing import Dict, List, Any
from groq import Groq
from local_vector_store import LocalVectorStore

class MetadataService:
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
        self.metadata_file = "local_data/column_metadata.json"
        os.makedirs("local_data", exist_ok=True)
    
    def analyze_and_store_metadata(self, file_content: bytes, file_name: str, user_id: str):
        """Extract column metadata and store in vector store for Stage 1 queries"""
        
        try:
            # Parse file to get columns and sample data
            if file_name.endswith('.csv'):
                import io
                df = pd.read_csv(io.BytesIO(file_content))
            else:
                print(f"Unsupported file type for metadata analysis: {file_name}")
                return
            
            # Get column info
            columns_info = []
            for col in df.columns:
                # Get sample values (non-null, unique)
                sample_values = df[col].dropna().unique()[:5].tolist()
                sample_values = [str(val) for val in sample_values if str(val).strip()]
                
                # Get data type
                dtype = str(df[col].dtype)
                
                column_info = {
                    "column_name": str(col),
                    "file_name": str(file_name),
                    "data_type": str(dtype),
                    "sample_values": [str(val) for val in sample_values],
                    "total_rows": int(len(df)),
                    "non_null_count": int(df[col].count())
                }
                columns_info.append(column_info)
            
            # Generate LLM description for each column
            enhanced_metadata = self._enhance_with_llm_descriptions(columns_info, file_name)
            
            # Store metadata in vector store for Stage 1 queries
            self._store_metadata_in_vector_store(enhanced_metadata, user_id)
            
            # Also save to JSON file as backup
            self._save_metadata_to_file(enhanced_metadata, user_id)
            
            print(f"Stored metadata for {len(enhanced_metadata)} columns from {file_name}")
            
        except Exception as e:
            print(f"Error analyzing metadata for {file_name}: {e}")
    
    def _enhance_with_llm_descriptions(self, columns_info: List[Dict], file_name: str) -> List[Dict]:
        """Use LLM to generate descriptions for columns based on names and sample data"""
        
        # Build prompt with all columns
        columns_summary = f"File: {file_name}\n\n"
        for i, col_info in enumerate(columns_info):
            columns_summary += f"Column {i+1}: {col_info['column_name']}\n"
            columns_summary += f"Data Type: {col_info['data_type']}\n"
            columns_summary += f"Sample Values: {', '.join(col_info['sample_values'][:3])}\n\n"
        
        # Skip LLM enhancement for now to avoid JSON parsing issues
        print("Skipping LLM enhancement to avoid JSON parsing issues")
        
        # Use simple rule-based descriptions
        for col_info in columns_info:
            col_name = col_info['column_name'].lower()
            if 'name' in col_name or 'title' in col_name:
                col_info['description'] = f"Names or titles (likely project names)"
                col_info['likely_queries'] = ["list projects", "show names"]
            elif 'id' in col_name or 'number' in col_name:
                col_info['description'] = f"Identifier or reference numbers"
                col_info['likely_queries'] = ["list ids", "show numbers"]
            elif 'cost' in col_name or 'price' in col_name or 'amount' in col_name:
                col_info['description'] = f"Financial amounts or costs"
                col_info['likely_queries'] = ["show costs", "list amounts"]
            elif 'date' in col_name or 'time' in col_name:
                col_info['description'] = f"Date or time information"
                col_info['likely_queries'] = ["show dates", "list times"]
            else:
                col_info['description'] = f"Data in {col_info['column_name']} column"
                col_info['likely_queries'] = [f"show {col_name}", f"list {col_name}"]
        
        return columns_info

        # This is now handled above with rule-based descriptions
    
    def _store_metadata_in_vector_store(self, metadata: List[Dict], user_id: str):
        """Store column metadata in vector store for Stage 1 queries"""
        
        from models import DataChunk
        
        metadata_chunks = []
        for col_meta in metadata:
            # Create searchable text for this column
            searchable_text = f"""
            File: {col_meta['file_name']}
            Column: {col_meta['column_name']}
            Description: {col_meta['description']}
            Data Type: {col_meta['data_type']}
            Sample Values: {', '.join(col_meta['sample_values'])}
            Likely Queries: {', '.join(col_meta['likely_queries'])}
            """
            
            # Create metadata chunk with proper file and column metadata
            chunk_metadata = {
                "source_file": col_meta['file_name'],
                "column_name": col_meta['column_name'],
                "data_type": col_meta['data_type'],
                "is_metadata": True,
                "domain": self._get_file_domain(col_meta['file_name'])
            }
            
            chunk = DataChunk(
                content=searchable_text.strip(),
                user_id=user_id,
                file_name=f"METADATA_{col_meta['file_name']}",
                chunk_id=f"meta_{col_meta['file_name']}_{col_meta['column_name']}",
                metadata=chunk_metadata
            )
            metadata_chunks.append(chunk)
        
        # Store in vector store
        self.vector_store.store_chunks(metadata_chunks)
        print(f"Stored {len(metadata_chunks)} metadata chunks in vector store")
    
    def _save_metadata_to_file(self, metadata: List[Dict], user_id: str):
        """Save metadata to JSON file as backup"""
        
        try:
            # Load existing metadata
            if os.path.exists(self.metadata_file):
                try:
                    with open(self.metadata_file, 'r') as f:
                        all_metadata = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    print("Corrupted metadata file, creating new one")
                    all_metadata = {}
            else:
                all_metadata = {}
            
            # Add new metadata
            if user_id not in all_metadata:
                all_metadata[user_id] = {}
            
            for col_meta in metadata:
                file_name = col_meta['file_name']
                if file_name not in all_metadata[user_id]:
                    all_metadata[user_id][file_name] = []
                all_metadata[user_id][file_name].append(col_meta)
            
            # Save back to file
            with open(self.metadata_file, 'w') as f:
                json.dump(all_metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error saving metadata to file: {e}")
    
    def get_user_metadata(self, user_id: str) -> Dict:
        """Get all metadata for a user"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    all_metadata = json.load(f)
                return all_metadata.get(user_id, {})
        except Exception as e:
            print(f"Error loading metadata: {e}")
        return {}
    
    def _get_file_domain(self, file_name: str) -> str:
        """Determine the domain/type of a file based on its name and content"""
        file_lower = file_name.lower()
        
        if any(word in file_lower for word in ['project', 'management', 'dataset']):
            return "projects"
        elif any(word in file_lower for word in ['employee', 'staff', 'hr', 'pyramid']):
            return "employees"
        elif any(word in file_lower for word in ['incident', 'issue', 'ticket']):
            return "incidents"
        else:
            return "general"