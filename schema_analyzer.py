import json
import os
from typing import Dict, List, Any
from groq import Groq

class SchemaAnalyzer:
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
        self.schema_file = "local_data/data_schemas.json"
        os.makedirs("local_data", exist_ok=True)
    
    def analyze_file_schema(self, file_name: str, sample_content: str, user_id: str):
        """Analyze file structure and create schema context"""
        
        # Get schema analysis from LLM
        prompt = f"""Analyze this data file and identify its structure:

File: {file_name}
Sample Data:
{sample_content[:2000]}

Please identify:
1. What type of data this contains (projects, issues, employees, etc.)
2. Key fields/columns and their purposes
3. Sample values for important fields
4. How to query this data (what questions users might ask)

Format as JSON:
{{
    "data_type": "projects|issues|employees|tasks|other",
    "description": "Brief description of what this file contains",
    "key_fields": [
        {{"field": "field_name", "description": "what it contains", "sample_values": ["val1", "val2"]}}
    ],
    "common_queries": ["example question 1", "example question 2"],
    "query_instructions": "How to find specific information in this data"
}}"""

        # Skip LLM schema analysis to avoid JSON parsing issues
        print(f"Using simple schema analysis for {file_name}")
        schema = self._simple_schema_analysis(file_name, sample_content)
        
        # Save schema
        self._save_schema(user_id, file_name, schema)
        return schema
    
    def _simple_schema_analysis(self, file_name: str, content: str) -> Dict:
        """Fallback schema analysis without LLM"""
        
        # Detect data type based on content
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['project_name', 'project', 'initiative']):
            data_type = "projects"
            description = "Project management data"
        elif any(word in content_lower for word in ['issue', 'ticket', 'bug', 'incident']):
            data_type = "issues"
            description = "Issue/ticket tracking data"
        elif any(word in content_lower for word in ['employee', 'staff', 'name', 'email']):
            data_type = "employees"
            description = "Employee/staff data"
        else:
            data_type = "other"
            description = "General data file"
        
        # Extract field names
        import re
        fields = re.findall(r'(\w+):', content)
        key_fields = [{"field": field, "description": f"{field} information", "sample_values": []} 
                     for field in list(set(fields))[:5]]
        
        return {
            "data_type": data_type,
            "description": description,
            "key_fields": key_fields,
            "common_queries": [f"List all {data_type}", f"Show {data_type} details"],
            "query_instructions": f"Look for {data_type} information in the data"
        }
    
    def _save_schema(self, user_id: str, file_name: str, schema: Dict):
        """Save schema to local file"""
        schemas = self._load_schemas()
        
        if user_id not in schemas:
            schemas[user_id] = {}
        
        schemas[user_id][file_name] = schema
        
        with open(self.schema_file, 'w') as f:
            json.dump(schemas, f, indent=2)
        
        print(f"Saved schema for {file_name}: {schema['data_type']}")
    
    def _load_schemas(self) -> Dict:
        """Load existing schemas"""
        if os.path.exists(self.schema_file):
            try:
                with open(self.schema_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def get_user_context(self, user_id: str) -> str:
        """Get context summary for user's data"""
        schemas = self._load_schemas()
        
        if user_id not in schemas:
            return "No data uploaded yet."
        
        context_parts = []
        context_parts.append("## Your Data Context:")
        
        for file_name, schema in schemas[user_id].items():
            context_parts.append(f"\n**{file_name}** ({schema['data_type']})")
            context_parts.append(f"- {schema['description']}")
            
            if schema.get('key_fields'):
                fields = [f['field'] for f in schema['key_fields'][:3]]
                context_parts.append(f"- Key fields: {', '.join(fields)}")
            
            if schema.get('common_queries'):
                context_parts.append(f"- Ask: {schema['common_queries'][0]}")
        
        return "\n".join(context_parts)
    
    def get_query_context(self, user_id: str, question: str) -> str:
        """Get relevant context for a specific query"""
        schemas = self._load_schemas()
        
        if user_id not in schemas:
            return ""
        
        question_lower = question.lower()
        relevant_schemas = []
        
        for file_name, schema in schemas[user_id].items():
            # Check if query matches this data type
            if (schema['data_type'] in question_lower or 
                any(query.lower() in question_lower for query in schema.get('common_queries', []))):
                relevant_schemas.append((file_name, schema))
        
        if not relevant_schemas:
            # Return all schemas if no specific match
            relevant_schemas = list(schemas[user_id].items())
        
        context_parts = []
        for file_name, schema in relevant_schemas[:3]:  # Limit to 3 most relevant
            context_parts.append(f"File: {file_name}")
            context_parts.append(f"Contains: {schema['description']}")
            context_parts.append(f"Query tip: {schema.get('query_instructions', 'Look for relevant data')}")
            context_parts.append("")
        
        return "\n".join(context_parts)