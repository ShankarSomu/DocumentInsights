import json
import os
import hashlib
from typing import Dict, List, Any, Optional
from groq import Groq
import pandas as pd

class SchemaMapper:
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
        self.mapping_cache_file = "local_data/schema_mappings.json"
        self.canonical_schemas_file = "local_data/canonical_schemas.json"
        self.mapping_cache = {}
        self.canonical_schemas = {}
        
        os.makedirs("local_data", exist_ok=True)
        self._load_caches()
        self._initialize_canonical_schemas()
    
    def map_file_to_canonical(self, df: pd.DataFrame, file_name: str, domain: str) -> Dict[str, Any]:
        """Map file columns to canonical schema with LLM assistance"""
        
        # Generate cache key based on columns and sample data
        cache_key = self._generate_cache_key(df, file_name, domain)
        
        # Check if mapping exists in cache
        if cache_key in self.mapping_cache:
            print(f"Using cached mapping for {file_name}")
            return self.mapping_cache[cache_key]
        
        print(f"Generating new LLM mapping for {file_name}")
        
        # Get canonical schema for domain
        canonical_schema = self.canonical_schemas.get(domain, {})
        
        if not canonical_schema:
            print(f"No canonical schema found for domain: {domain}")
            return self._create_direct_mapping(df, file_name)
        
        # Generate mapping using LLM
        mapping = self._generate_llm_mapping(df, file_name, domain, canonical_schema)
        
        # Cache the mapping
        self.mapping_cache[cache_key] = mapping
        self._save_mapping_cache()
        
        return mapping
    
    def _generate_cache_key(self, df: pd.DataFrame, file_name: str, domain: str) -> str:
        """Generate cache key based on file structure"""
        
        # Create signature from columns and sample data
        columns_str = ",".join(sorted(df.columns.tolist()))
        sample_data = ""
        
        for col in df.columns[:5]:  # First 5 columns
            sample_values = df[col].dropna().astype(str).head(3).tolist()
            sample_data += f"{col}:{','.join(sample_values)};"
        
        signature = f"{file_name}_{domain}_{columns_str}_{sample_data}"
        return hashlib.md5(signature.encode()).hexdigest()
    
    def _generate_llm_mapping(self, df: pd.DataFrame, file_name: str, domain: str, canonical_schema: Dict) -> Dict[str, Any]:
        """Use LLM to map file columns to canonical schema"""
        
        # Prepare file analysis
        file_analysis = self._analyze_file_structure(df, file_name)
        canonical_fields = canonical_schema.get('fields', {})
        
        prompt = f"""Map the columns from this file to the canonical schema.

FILE: {file_name}
DOMAIN: {domain}

FILE COLUMNS:
{file_analysis}

CANONICAL SCHEMA:
{json.dumps(canonical_fields, indent=2)}

Create a mapping that transforms the file columns to canonical field names.
For columns that don't match any canonical field, suggest if they should be:
1. Mapped to an existing canonical field (with transformation)
2. Added as a new canonical field
3. Ignored (if not relevant)

Return JSON only:
{{
    "mappings": {{
        "original_column_name": {{
            "canonical_field": "canonical_field_name",
            "transformation": "none|rename|convert_type|split|combine",
            "transformation_rule": "description of transformation if needed"
        }}
    }},
    "new_canonical_fields": {{
        "new_field_name": {{
            "type": "string|integer|float|date|boolean",
            "description": "what this field represents"
        }}
    }},
    "ignored_columns": ["column1", "column2"],
    "confidence": 0.85
}}"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a data schema expert. Map file columns to canonical schemas. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse LLM response
            response_text = response.choices[0].message.content.strip()
            response_text = self._clean_json_response(response_text)
            
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                mapping = json.loads(json_match.group())
                
                # Add metadata
                mapping['file_name'] = file_name
                mapping['domain'] = domain
                mapping['cache_key'] = self._generate_cache_key(df, file_name, domain)
                
                # Update canonical schema if new fields suggested
                if mapping.get('new_canonical_fields'):
                    self._update_canonical_schema(domain, mapping['new_canonical_fields'])
                
                return mapping
            
        except Exception as e:
            print(f"LLM mapping failed for {file_name}: {e}")
        
        # Fallback to direct mapping
        return self._create_direct_mapping(df, file_name)
    
    def _analyze_file_structure(self, df: pd.DataFrame, file_name: str) -> str:
        """Analyze file structure for LLM"""
        
        analysis = f"File: {file_name}\nColumns ({len(df.columns)}):\n\n"
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            total_count = len(df)
            
            # Get sample values
            sample_values = df[col].dropna().astype(str).head(3).tolist()
            
            analysis += f"- {col} ({dtype}): {non_null_count}/{total_count} non-null\n"
            analysis += f"  Sample values: {', '.join(sample_values)}\n\n"
        
        return analysis
    
    def _create_direct_mapping(self, df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
        """Create direct mapping when LLM fails"""
        
        mappings = {}
        for col in df.columns:
            mappings[col] = {
                "canonical_field": col.lower().replace(' ', '_').replace('-', '_'),
                "transformation": "rename",
                "transformation_rule": "Direct mapping with name normalization"
            }
        
        return {
            "mappings": mappings,
            "new_canonical_fields": {},
            "ignored_columns": [],
            "confidence": 0.5,
            "file_name": file_name,
            "fallback": True
        }
    
    def apply_mapping(self, df: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
        """Apply mapping to transform DataFrame to canonical schema"""
        
        if not mapping or mapping.get('fallback'):
            # Simple column name normalization
            df_mapped = df.copy()
            df_mapped.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_mapped.columns]
            return df_mapped
        
        df_mapped = pd.DataFrame()
        mappings = mapping.get('mappings', {})
        
        for original_col, mapping_info in mappings.items():
            if original_col not in df.columns:
                continue
                
            canonical_field = mapping_info.get('canonical_field')
            transformation = mapping_info.get('transformation', 'none')
            
            if transformation == 'none':
                df_mapped[canonical_field] = df[original_col]
            elif transformation == 'rename':
                df_mapped[canonical_field] = df[original_col]
            elif transformation == 'convert_type':
                df_mapped[canonical_field] = self._convert_column_type(df[original_col], mapping_info)
            else:
                # Default: copy as-is
                df_mapped[canonical_field] = df[original_col]
        
        return df_mapped
    
    def _convert_column_type(self, series: pd.Series, mapping_info: Dict) -> pd.Series:
        """Convert column type based on mapping info"""
        
        try:
            transformation_rule = mapping_info.get('transformation_rule', '')
            
            if 'integer' in transformation_rule.lower():
                return pd.to_numeric(series, errors='coerce').astype('Int64')
            elif 'float' in transformation_rule.lower():
                return pd.to_numeric(series, errors='coerce')
            elif 'date' in transformation_rule.lower():
                return pd.to_datetime(series, errors='coerce')
            elif 'boolean' in transformation_rule.lower():
                return series.astype(str).str.lower().isin(['true', '1', 'yes', 'y'])
            else:
                return series.astype(str)
                
        except Exception as e:
            print(f"Type conversion failed: {e}")
            return series
    
    def _initialize_canonical_schemas(self):
        """Initialize canonical schemas for different domains"""
        
        default_schemas = {
            "projects": {
                "fields": {
                    "project_id": {"type": "string", "description": "Unique project identifier"},
                    "project_name": {"type": "string", "description": "Project name or title"},
                    "status": {"type": "string", "description": "Project status (active, completed, etc.)"},
                    "priority": {"type": "string", "description": "Project priority level"},
                    "risk_level": {"type": "string", "description": "Risk assessment level"},
                    "start_date": {"type": "date", "description": "Project start date"},
                    "end_date": {"type": "date", "description": "Project end date"},
                    "budget": {"type": "float", "description": "Project budget amount"},
                    "manager": {"type": "string", "description": "Project manager name"},
                    "department": {"type": "string", "description": "Department or team"},
                    "completion_percentage": {"type": "float", "description": "Completion percentage"}
                }
            },
            "employees": {
                "fields": {
                    "employee_id": {"type": "string", "description": "Unique employee identifier"},
                    "employee_name": {"type": "string", "description": "Employee full name"},
                    "email": {"type": "string", "description": "Employee email address"},
                    "department": {"type": "string", "description": "Department or team"},
                    "role": {"type": "string", "description": "Job title or role"},
                    "manager": {"type": "string", "description": "Manager name"},
                    "skills": {"type": "string", "description": "Employee skills"},
                    "capacity": {"type": "float", "description": "Work capacity percentage"},
                    "productivity": {"type": "float", "description": "Productivity rating"},
                    "location": {"type": "string", "description": "Work location"}
                }
            },
            "incidents": {
                "fields": {
                    "incident_id": {"type": "string", "description": "Unique incident identifier"},
                    "title": {"type": "string", "description": "Incident title or description"},
                    "status": {"type": "string", "description": "Incident status"},
                    "severity": {"type": "string", "description": "Severity level"},
                    "priority": {"type": "string", "description": "Priority level"},
                    "created_date": {"type": "date", "description": "Incident creation date"},
                    "resolved_date": {"type": "date", "description": "Resolution date"},
                    "assignee": {"type": "string", "description": "Assigned person"},
                    "project_id": {"type": "string", "description": "Related project"},
                    "category": {"type": "string", "description": "Incident category"}
                }
            }
        }
        
        # Load existing schemas or use defaults
        if os.path.exists(self.canonical_schemas_file):
            try:
                with open(self.canonical_schemas_file, 'r') as f:
                    self.canonical_schemas = json.load(f)
            except Exception as e:
                print(f"Error loading canonical schemas: {e}")
                self.canonical_schemas = default_schemas
        else:
            self.canonical_schemas = default_schemas
            self._save_canonical_schemas()
    
    def _update_canonical_schema(self, domain: str, new_fields: Dict):
        """Update canonical schema with new fields"""
        
        if domain not in self.canonical_schemas:
            self.canonical_schemas[domain] = {"fields": {}}
        
        self.canonical_schemas[domain]["fields"].update(new_fields)
        self._save_canonical_schemas()
        
        print(f"Updated canonical schema for {domain} with {len(new_fields)} new fields")
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean LLM response for JSON parsing"""
        
        # Remove markdown code blocks
        response_text = response_text.replace('```json', '').replace('```', '')
        
        # Clean control characters
        response_text = ''.join(char for char in response_text if ord(char) >= 32 or char in '\n\r\t')
        
        return response_text.strip()
    
    def _load_caches(self):
        """Load mapping cache from file"""
        
        try:
            if os.path.exists(self.mapping_cache_file):
                with open(self.mapping_cache_file, 'r') as f:
                    self.mapping_cache = json.load(f)
        except Exception as e:
            print(f"Error loading mapping cache: {e}")
            self.mapping_cache = {}
    
    def _save_mapping_cache(self):
        """Save mapping cache to file"""
        
        try:
            with open(self.mapping_cache_file, 'w') as f:
                json.dump(self.mapping_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving mapping cache: {e}")
    
    def _save_canonical_schemas(self):
        """Save canonical schemas to file"""
        
        try:
            with open(self.canonical_schemas_file, 'w') as f:
                json.dump(self.canonical_schemas, f, indent=2)
        except Exception as e:
            print(f"Error saving canonical schemas: {e}")
    
    def get_mapping_info(self, file_name: str, domain: str) -> Optional[Dict]:
        """Get mapping information for a file"""
        
        for mapping in self.mapping_cache.values():
            if mapping.get('file_name') == file_name and mapping.get('domain') == domain:
                return mapping
        
        return None
    
    def invalidate_cache(self, file_name: str = None, domain: str = None):
        """Invalidate cache entries"""
        
        if file_name is None and domain is None:
            # Clear all cache
            self.mapping_cache = {}
        else:
            # Remove specific entries
            keys_to_remove = []
            for key, mapping in self.mapping_cache.items():
                if (file_name and mapping.get('file_name') == file_name) or \
                   (domain and mapping.get('domain') == domain):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.mapping_cache[key]
        
        self._save_mapping_cache()
        print(f"Invalidated cache entries for file: {file_name}, domain: {domain}")