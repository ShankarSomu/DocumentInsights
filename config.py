import os
import json
from typing import Dict, List, Any

class Config:
    def __init__(self):
        self.config_file = "config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        # Create default configuration
        default_config = {
            "models": {
                "llm_model": "llama-3.1-8b-instant",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384
            },
            "storage": {
                "data_directory": "local_data",
                "index_name": "projectiq-hybrid",
                "chunk_size": 5,
                "cache_file": "query_cache.json",
                "mappings_file": "text_mappings.json",
                "schemas_file": "canonical_schemas.json"
            },
            "limits": {
                "default_top_k": 20,
                "max_results": 1000,
                "display_limit": 50,
                "batch_size": 100,
                "max_tokens_default": 500,
                "max_tokens_document": 1000
            },
            "file_processing": {
                "supported_extensions": [
                    ".csv", ".json", ".xml", ".txt", ".doc", ".docx", 
                    ".ppt", ".pptx", ".pdf", ".xls", ".xlsx"
                ],
                "text_extensions": [".txt", ".csv", ".json", ".xml"],
                "document_extensions": [".doc", ".docx", ".pdf"],
                "presentation_extensions": [".ppt", ".pptx"],
                "spreadsheet_extensions": [".xls", ".xlsx"]
            },
            "domains": {
                "projects": {
                    "keywords": ["project", "management", "milestone", "charter"],
                    "file_patterns": ["project", "management", "milestone"]
                },
                "employees": {
                    "keywords": ["employee", "staff", "hr", "team", "people"],
                    "file_patterns": ["employee", "staff", "hr", "team"]
                },
                "incidents": {
                    "keywords": ["incident", "issue", "ticket", "bug", "problem"],
                    "file_patterns": ["incident", "issue", "ticket", "bug"]
                }
            },
            "text_normalization": {
                "punctuation_patterns": [
                    ["[–—−]", "-"],
                    ["\\s*[-–—−]\\s*", " - "],
                    ["\\s*[/\\\\]\\s*", " / "],
                    ["\\s*:\\s*", ": "],
                    ["\\s+", " "]
                ],
                "semantic_equivalents": {
                    "for": ["–", "—", "-", "of", "regarding"],
                    "project": ["proj", "prj"],
                    "charter": ["document", "doc"],
                    "plan": ["planning", "strategy"]
                }
            },
            "query_classification": {
                "structured_indicators": [
                    "count", "total", "sum", "average", "calculate",
                    "list all", "show all", "top", "highest", "lowest"
                ],
                "semantic_indicators": [
                    "explain", "describe", "what is", "meaning",
                    "charter", "document", "report", "content"
                ],
                "document_patterns": ["charter", "document", "report"]
            },
            "database": {
                "table_prefix_pattern": "{user_id}_{filename}",
                "mapping_table_suffix": "_mapping",
                "metadata_table_prefix": "METADATA_"
            },
            "ui": {
                "typing_speed": 20,
                "max_file_display": 10,
                "session_id_length": 8
            }
        }
        
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'models.llm_model')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
        self._save_config(self.config)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.get('models', {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        return self.get('storage', {})
    
    def get_limits_config(self) -> Dict[str, Any]:
        """Get limits configuration"""
        return self.get('limits', {})
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions"""
        return self.get('file_processing.supported_extensions', [])
    
    def get_domain_config(self, domain: str) -> Dict[str, Any]:
        """Get configuration for a specific domain"""
        return self.get(f'domains.{domain}', {})
    
    def get_normalization_config(self) -> Dict[str, Any]:
        """Get text normalization configuration"""
        return self.get('text_normalization', {})
    
    def update_from_env(self):
        """Update configuration from environment variables"""
        env_mappings = {
            'GROQ_MODEL': 'models.llm_model',
            'EMBEDDING_MODEL': 'models.embedding_model',
            'DATA_DIR': 'storage.data_directory',
            'CHUNK_SIZE': 'storage.chunk_size',
            'DEFAULT_TOP_K': 'limits.default_top_k'
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Convert to appropriate type
                if config_path.endswith(('size', 'top_k', 'dimension')):
                    env_value = int(env_value)
                self.set(config_path, env_value)

# Global configuration instance
config = Config()
config.update_from_env()