import os
import shutil
from typing import Set

class SessionManager:
    def __init__(self):
        self.data_dir = "local_data"
        self.active_users: Set[str] = set()
    
    def cleanup_old_sessions(self, current_user_id: str):
        """Clean up data from old sessions, keep only current user"""
        if not os.path.exists(self.data_dir):
            return
        
        # Add current user to active list
        self.active_users.add(current_user_id)
        
        # Get all user files
        user_files = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_index.faiss') or filename.endswith('_metadata.pkl'):
                user_id = filename.split('_')[0]
                if user_id != current_user_id:
                    user_files.append(filename)
        
        # Remove old user data files
        for filename in user_files:
            try:
                file_path = os.path.join(self.data_dir, filename)
                os.remove(file_path)
                print(f"Cleaned up old session file: {filename}")
            except Exception as e:
                print(f"Error cleaning up {filename}: {e}")
        
        # Clean up schema file - keep only current user
        schema_file = os.path.join(self.data_dir, "data_schemas.json")
        if os.path.exists(schema_file):
            try:
                import json
                with open(schema_file, 'r') as f:
                    schemas = json.load(f)
                
                # Keep only current user's schemas
                if current_user_id in schemas:
                    new_schemas = {current_user_id: schemas[current_user_id]}
                else:
                    new_schemas = {}
                
                with open(schema_file, 'w') as f:
                    json.dump(new_schemas, f, indent=2)
                
                print(f"Cleaned up schemas, kept only user: {current_user_id}")
            except Exception as e:
                print(f"Error cleaning up schemas: {e}")
    
    def is_new_session(self, user_id: str) -> bool:
        """Check if this is a new session"""
        return user_id not in self.active_users