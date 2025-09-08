import os
from typing import Dict, List, Any, Optional
import pandas as pd
import sqlite3
from models import ChatResponse
from local_vector_store import LocalVectorStore
from groq import Groq

class LLMDrivenService:
    def __init__(self):
        try:
            from secure_config import SecureConfig
            config = SecureConfig().get_api_keys()
            groq_key = config.get("GROQ_API_KEY")
            if not groq_key:
                raise Exception("GROQ_API_KEY not found in secure configuration")
            self.groq_client = Groq(api_key=groq_key)
        except Exception as e:
            print(f"Failed to initialize LLM service: {e}")
            self.groq_client = None

        self.vector_store = LocalVectorStore()
        self.db_path = "local_data/structured_data.db"
        os.makedirs("local_data", exist_ok=True)
        
        # Conversation context storage
        self.conversation_history = {}  # {user_id: [{"question": "", "response": "", "columns": []}]}

    def process_query(self, question: str, user_id: str) -> ChatResponse:
        """Process query using simple SQL JOINs + pandas analysis"""
        if not self.groq_client:
            return ChatResponse(answer="LLM service not available. Check API key.", sources=[])

        print(f"*** Processing Query: {question} ***")
        
        # Get conversation context
        context = self._get_conversation_context(user_id, question)
        contextual_question = self._enhance_question_with_context(question, context)
        
        # Step 1: Get relevant columns and filters from LLM (with context)
        relevant_columns = self._llm_determine_columns(contextual_question, user_id, context)
        
        # Step 1.5: Determine if this is aggregation or filtering
        is_aggregation = self._is_aggregation_query(contextual_question)
        if is_aggregation:
            print("*** Detected aggregation query - skipping filters ***")
            filter_conditions = ""
        else:
            filter_conditions = self._llm_determine_filters(contextual_question, user_id)
        
        # Override for specific question types
        print(f"*** Original LLM Selected Columns: {relevant_columns} ***")
        relevant_columns = self._override_columns_if_needed(question, user_id, relevant_columns)
        print(f"*** Final Selected Columns: {relevant_columns} ***")
        
        if not relevant_columns:
            # Emergency fallback - use basic project columns
            print("*** Using emergency fallback columns ***")
            relevant_columns = ['project_name', 'project_manager', 'complexity', 'status']
            print(f"*** Emergency columns: {relevant_columns} ***")
        
        # Step 2: Use previous result set if available, otherwise fetch new data
        if hasattr(self, '_temp_result_df') and self._temp_result_df is not None:
            print("*** Using previous result set for follow-up query ***")
            df = self._temp_result_df
            self._temp_result_df = None  # Clear after use
        else:
            df = self._fetch_joined_data(user_id, relevant_columns, filter_conditions)
            if df.empty and filter_conditions:
                print("*** No data with filters, trying without filters ***")
                # Fetch data without filters
                df_unfiltered = self._fetch_joined_data(user_id, relevant_columns, "")
                if not df_unfiltered.empty:
                    # Let LLM analyze unfiltered data and provide correct filter
                    corrected_analysis = self._llm_correct_filter_and_analyze(contextual_question, filter_conditions, df_unfiltered, user_id)
                    return ChatResponse(answer=corrected_analysis, sources=self._get_user_tables(user_id))
                else:
                    return ChatResponse(answer="No data available in the table.", sources=self._get_user_tables(user_id))
            elif df.empty:
                return ChatResponse(answer="No data available for analysis.", sources=self._get_user_tables(user_id))
        
        # Step 3: Pandas analysis (with context)
        analysis_result = self._pandas_analysis(contextual_question, df, context)
        
        # Store in conversation history with result set
        self._store_conversation(user_id, question, analysis_result, relevant_columns, df)
        
        return ChatResponse(answer=analysis_result, sources=self._get_user_tables(user_id))

    def _llm_determine_columns(self, question: str, user_id: str, context: str = "") -> List[str]:
        """Ask LLM to choose which columns are relevant for the analysis"""
        tables = self._get_user_tables(user_id)
        if not tables:
            return []
        schema_info = self._get_detailed_schema(user_id)
        print(f"*** Available Tables: {tables} ***")
        print(f"*** Schema Info: {schema_info} ***")
        
        context_prompt = f"\nContext from previous conversation:\n{context}" if context else ""
        
        prompt = f"""Question: "{question}"{context_prompt}

Available Tables and Columns:
{schema_info}

Analyze this question and select the minimum essential columns needed to answer it. 

Examples:
- "Which managers handle complex projects?" → ["project_name", "project_manager", "complexity"]
- "Predict delays based on incidents" → ["project_name", "time_variance_days", "status"]
- "Average cost by region" → ["project_name", "project_cost", "region"]
- "Employee productivity" → ["employee_name", "productivity_rating", "project_name"]

Select 3-6 most relevant columns. Include project_name if joining tables.
Return ONLY a valid JSON array: ["col1", "col2", "col3"]"""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a database analyst. Analyze the question and return ONLY a JSON array of column names. Be precise and minimal. Always include project_name when multiple tables are involved."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            import json, re
            text = response.choices[0].message.content
            print(f"*** Raw LLM Column Response: {text} ***")
            
            # Try multiple patterns to extract JSON array
            patterns = [
                r'\[.*?\]',  # Standard array
                r'```json\s*\[.*?\]\s*```',  # JSON code block
                r'```\s*\[.*?\]\s*```',  # Code block
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    json_str = match.group()
                    # Clean up the JSON string
                    json_str = re.sub(r'```json|```', '', json_str).strip()
                    try:
                        selected_columns = json.loads(json_str)
                        print(f"*** LLM Selected Columns: {selected_columns} ***")
                        return selected_columns
                    except json.JSONDecodeError as e:
                        print(f"*** JSON parse error for pattern {pattern}: {e} ***")
                        continue
            
            # Fallback: use common columns for incident/delay analysis
            print("*** Using fallback columns for incident/delay analysis ***")
            return ['project_name', 'status', 'time_variance_days', 'incident_count']
        except Exception as e:
            print(f"LLM column selection failed: {e}")
        # Smart fallback - extract key terms from question
        question_lower = question.lower()
        fallback_cols = ['project_name']  # Always include project_name
        
        # Add columns based on question keywords
        if any(word in question_lower for word in ['manager', 'lead', 'owner']):
            fallback_cols.append('project_manager')
        if any(word in question_lower for word in ['complex', 'difficulty', 'hard']):
            fallback_cols.append('complexity')
        if any(word in question_lower for word in ['cost', 'budget', 'money', 'expense', 'over budget']):
            fallback_cols.extend(['project_cost', 'actual_cost', 'cost_variance'])
        if any(word in question_lower for word in ['delay', 'time', 'schedule', 'late']):
            fallback_cols.append('time_variance_days')
        if any(word in question_lower for word in ['status', 'progress', 'complete']):
            fallback_cols.extend(['status', 'completion_pct'])
        if any(word in question_lower for word in ['region', 'location', 'area']):
            fallback_cols.append('region')
        if any(word in question_lower for word in ['department', 'team', 'group']):
            fallback_cols.append('department')
        if any(word in question_lower for word in ['employee', 'staff', 'people', 'productivity']):
            fallback_cols.extend(['employee_name', 'productivity_rating'])
        
        # Remove duplicates and limit to 6 columns
        return list(dict.fromkeys(fallback_cols))[:6]
    
    def _override_columns_if_needed(self, question: str, user_id: str, columns: List[str]) -> List[str]:
        """Override column selection for specific question types"""
        q_lower = question.lower()
        
        # For completion date predictions, ensure we have timeline and progress data
        if 'completion' in q_lower and 'date' in q_lower:
            print(f"*** Detected completion date question, checking for override ***")
            schema_info = self._get_detailed_schema(user_id)
            available_cols = []
            for line in schema_info.split('\n'):
                if 'Columns:' in line:
                    cols = line.split('Columns:')[1].strip().split(', ')
                    available_cols.extend(cols)
            
            print(f"*** Available columns in schema: {available_cols} ***")
            
            # Essential columns for completion prediction
            essential = ['project_name', 'status', 'completion_pct', 'planned_end_date', 'actual_end_date', 'start_date', 'end_date', 'time_variance_days']
            override_cols = [col for col in essential if col in available_cols]
            
            print(f"*** Override columns found: {override_cols} ***")
            if override_cols:
                print(f"*** Overriding columns for completion prediction: {override_cols} ***")
                return override_cols
        
        return columns

    def _fetch_joined_data(self, user_id: str, columns: List[str], filters: str = "") -> pd.DataFrame:
        """Fetch data using simple SQL JOIN on project_name"""
        tables = self._get_user_tables(user_id)
        if not tables:
            return pd.DataFrame()
        
        # Map columns to tables
        table_columns = self._map_columns_to_tables(user_id, tables, columns)
        
        # Check if Project_Management_Dataset has all required columns
        project_table = None
        for table_name in table_columns.keys():
            if 'Project_Management_Dataset' in table_name:
                project_table = table_name
                break
        
        # Use only Project_Management_Dataset if it has the main columns needed
        if project_table and len(table_columns[project_table]) >= 3:  # Has essential columns
            cols = ', '.join(table_columns[project_table])
            query = f"SELECT {cols} FROM {project_table}"
            print(f"*** Using single table: {project_table} ***")
        elif len(table_columns) == 1:
            # Single table
            table = list(table_columns.keys())[0]
            cols = ', '.join(table_columns[table])
            query = f"SELECT {cols} FROM {table}"
            print(f"*** Using single table: {table} ***")
        else:
            # Simple JOIN on project_name
            query = self._build_simple_join(table_columns)
        
        # Add WHERE clause if filters provided
        if filters:
            query += f" WHERE {filters}"
            print(f"*** Applied Filters: {filters} ***")
        
        print(f"*** SQL Query: {query} ***")
        print(f"*** Table Columns Mapping: {table_columns} ***")
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            print(f"*** Fetched {len(df)} rows ***")
            print(f"*** Sample data columns: {list(df.columns)} ***")
            if not df.empty:
                print(f"*** Sample row: {df.iloc[0].to_dict()} ***")
            return df
        except Exception as e:
            print(f"SQL fetch failed: {e}")
            return pd.DataFrame()
    
    def _map_columns_to_tables(self, user_id: str, tables: List[str], columns: List[str]) -> Dict[str, List[str]]:
        """Map columns to their respective tables"""
        table_columns = {}
        try:
            conn = sqlite3.connect(self.db_path)
            for table in tables:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table})")
                existing_columns = {col[1] for col in cursor.fetchall()}
                valid_cols = [col for col in columns if col in existing_columns]
                if valid_cols:
                    table_columns[table] = valid_cols
            conn.close()
        except Exception as e:
            print(f"Column mapping failed: {e}")
        return table_columns
    
    def _build_simple_join(self, table_columns: Dict[str, List[str]]) -> str:
        """Build simple JOIN query with incident aggregation"""
        tables = list(table_columns.keys())
        
        # Prioritize Project_Management_Dataset as base table
        base_table = tables[0]
        for table in tables:
            if 'Project_Management_Dataset' in table:
                base_table = table
                break
        
        # Check if incident table is involved
        incident_table = None
        for table in tables:
            if 'incident' in table.lower():
                incident_table = table
                break
        
        # Check if we actually need incident data based on selected columns
        needs_incident_data = any('incident' in col.lower() for table_cols in table_columns.values() for col in table_cols)
        
        if incident_table and needs_incident_data:
            # Only aggregate incidents if incident columns are actually requested
            project_cols = []
            incident_cols = []
            
            for table, columns in table_columns.items():
                if 'incident' in table.lower():
                    incident_cols = [col for col in columns if col != 'project_name']
                else:
                    project_cols = [col for col in columns]
            
            # Build query with incident aggregation
            query = f"SELECT p.{', p.'.join(project_cols)}, COUNT(i.project_name) as incident_count"
            if incident_cols:
                query += f", GROUP_CONCAT(DISTINCT i.{incident_cols[0]}) as {incident_cols[0]}_list"
            
            query += f" FROM {base_table} p LEFT JOIN {incident_table} i ON p.project_name = i.project_name GROUP BY p.project_name"
        else:
            # Use only the base table if no incident data needed
            if len(table_columns) == 1 or not any('incident' in t.lower() for t in tables):
                # Single table or no incident tables needed - use the table with most columns
                best_table = max(table_columns.keys(), key=lambda t: len(table_columns[t]))
                cols = table_columns[best_table]
                query = f"SELECT {', '.join(cols)} FROM {best_table}"
            else:
                # Regular join for non-incident tables
                select_parts = [f"{base_table}.project_name"]
                for table, columns in table_columns.items():
                    for col in columns:
                        if col != 'project_name':
                            select_parts.append(f"{table}.{col}")
                
                query = f"SELECT {', '.join(select_parts)} FROM {base_table}"
                for table in tables[1:]:
                    if 'incident' not in table.lower():  # Skip incident table if not needed
                        query += f" LEFT JOIN {table} ON {base_table}.project_name = {table}.project_name"
        
        return query
    
    def _get_valid_columns(self, user_id: str, table: str, requested_columns: List[str]) -> List[str]:
        """Filter columns to only those that exist in the table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            existing_columns = {col[1] for col in cursor.fetchall()}
            conn.close()
            return [col for col in requested_columns if col in existing_columns]
        except Exception as e:
            print(f"Column validation failed: {e}")
            return []

    def _pandas_analysis(self, question: str, df: pd.DataFrame, context: str = "") -> str:
        """Route between pandas operations and LLM analysis based on query type"""
        try:
            # Check if this is a simple data operation
            if self._is_simple_data_operation(question):
                print("*** Using pandas for simple data operation ***")
                return self._handle_with_pandas(question, df)
            else:
                print("*** Using LLM for complex reasoning ***")
                return self._llm_analyze_combined_data(question, df, context)
        except Exception as e:
            print(f"Pandas analysis failed: {e}")
            return f"Data contains {len(df)} rows and columns: {list(df.columns)}"
    
    def _is_simple_data_operation(self, question: str) -> bool:
        """Determine if query is about existing data vs requiring complex reasoning"""
        q_lower = question.lower()
        
        # Simple data operations (use pandas)
        simple_patterns = [
            'in progress', 'completed', 'cancelled', 'on hold',  # Status filtering
            'top ', 'bottom ', 'highest ', 'lowest ',  # Ranking
            'how many', 'count of', 'number of',  # Counting
            'list ', 'show ', 'display ',  # Listing
            'filter', 'where ', 'with status',  # Filtering
            'group by', 'by region', 'by department',  # Grouping
            'average', 'sum', 'total', 'maximum', 'minimum'  # Aggregations
        ]
        
        # Complex operations (use LLM)
        complex_patterns = [
            'predict', 'forecast', 'estimate',  # Predictions
            'recommend', 'suggest', 'should',  # Recommendations
            'analyze', 'insights', 'trends',  # Analysis
            'compare', 'correlation', 'relationship',  # Comparisons
            'why', 'what if', 'impact of'  # Reasoning
        ]
        
        # Check for simple patterns first
        if any(pattern in q_lower for pattern in simple_patterns):
            return True
        
        # Check for complex patterns
        if any(pattern in q_lower for pattern in complex_patterns):
            return False
        
        # Default: treat as simple if it's about existing data
        return True
    
    def _handle_with_pandas(self, question: str, df: pd.DataFrame) -> str:
        """Handle simple data operations with pandas"""
        q_lower = question.lower()
        
        # Status filtering
        if 'in progress' in q_lower:
            filtered = df[df['status'].str.contains('Progress', case=False, na=False)]
        elif 'completed' in q_lower:
            filtered = df[df['status'].str.contains('Completed', case=False, na=False)]
        elif 'cancelled' in q_lower:
            filtered = df[df['status'].str.contains('Cancelled', case=False, na=False)]
        elif 'on hold' in q_lower:
            filtered = df[df['status'].str.contains('Hold', case=False, na=False)]
        else:
            filtered = df
        
        # Create HTML table
        if filtered.empty:
            return "<p>No data found matching the criteria.</p>"
        
        html = "<table border='1'><tr>"
        for col in filtered.columns:
            html += f"<th>{col.replace('_', ' ').title()}</th>"
        html += "</tr>"
        
        for _, row in filtered.iterrows():
            html += "<tr>"
            for col in filtered.columns:
                value = row[col] if pd.notna(row[col]) else 'N/A'
                html += f"<td>{value}</td>"
            html += "</tr>"
        html += "</table>"
        
        return html
    
    def _combine_relevant_data(self, question: str, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine relevant DataFrames based on question context"""
        q_lower = question.lower()
        
        # For incident/delay analysis, join projects and incidents
        if 'incident' in q_lower and 'delay' in q_lower:
            if 'Project_Management_Dataset' in dataframes and 'incident' in dataframes:
                projects = dataframes['Project_Management_Dataset']
                incidents = dataframes['incident']
                
                # Count incidents per project
                incident_counts = incidents.groupby('project_name').size().reset_index(name='incident_count')
                
                # Merge with project data
                combined = projects.merge(incident_counts, on='project_name', how='left')
                combined['incident_count'] = combined['incident_count'].fillna(0)
                return combined
        
        # Default: return the largest DataFrame
        if dataframes:
            largest_df = max(dataframes.values(), key=len)
            return largest_df
        
        return pd.DataFrame()
    
    def _llm_analyze_combined_data(self, question: str, df: pd.DataFrame, context: str = "") -> str:
        """Use LLM to analyze combined pandas data"""
        try:
            print(f"*** DataFrame shape: {df.shape} ***")
            print(f"*** DataFrame columns: {list(df.columns)} ***")
            print(f"*** Unique projects in data: {df['project_name'].nunique() if 'project_name' in df.columns else 'N/A'} ***")
            
            # Check if this is a document formatting request
            if self._is_document_format_request(question):
                return self._format_as_document(question, df)
            
            summary_csv = df.head(50).to_csv(index=False)
            context_prompt = f"\nPrevious conversation context:\n{context}" if context else ""
            
            # Check if this is a percentage/aggregation query
            if any(word in question.lower() for word in ['percentage', 'percent', '%']):
                prompt = f"""Question: "{question}"{context_prompt}
Data ({len(df)} total rows): {summary_csv}

This is a PERCENTAGE calculation question. You must:
1. Identify what categories/groups are being asked about
2. Count items in each category
3. Calculate percentages (count/total * 100)
4. Show results as Category | Count | Percentage

For "What percentage of projects are in Execution phase":
- Find which phase represents "Execution" in the data
- Count projects in that phase
- Calculate percentage of total

Return ONLY an HTML table with Category, Count, and Percentage columns."""
            else:
                prompt = f"""Question: "{question}"{context_prompt}
Data ({len(df)} total rows): {summary_csv}

IMPORTANT: You must FILTER the data, not show all rows.

1. Analyze what conditions the question is asking for
2. Apply those conditions to filter the data
3. Show ONLY rows that meet ALL the specified criteria
4. Do NOT include rows that don't match the conditions

Create ONE HTML table with ONLY the filtered rows that match the question.
Return ONLY the HTML table."""
            
            # First, get LLM's interpretation
            interpretation = self._get_llm_interpretation(question, df)
            print(f"*** LLM Interpretation: {interpretation} ***")
            
            if 'percentage' in question.lower():
                system_message = "Calculate percentages and show summary statistics. For percentage questions, group data and calculate percentages."
            else:
                system_message = "You MUST filter data based on question criteria. Never show all rows. Only show rows that match the specified conditions. Apply logical filtering."
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            print(f"*** Raw LLM Response Length: {len(content)} characters ***")
            
            # Check if response was truncated
            if len(content) > 1400 and not content.strip().endswith('</table>'):
                print("*** Response appears truncated, using summary approach ***")
                return self._create_summary_table(question, df)
            
            # Aggressive cleaning
            content = content.strip()
            # Remove any text/whitespace before <table>
            if '<table' in content:
                table_start = content.find('<table')
                content = content[table_start:]
            # Remove extra newlines and spaces
            content = content.replace('\n\n', '').replace('\n', '').strip()
            
            print(f"*** Cleaned Response: '{content}' ***")
            
            # Validate response matches query
            if not self._validate_response(question, content):
                print("*** Response validation failed, regenerating ***")
                regenerated = self._regenerate_response(question, df)
                print(f"*** Regenerated Response: {regenerated[:200]}... ***")
                return regenerated
            
            return content
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return f"Data contains {len(df)} rows and columns: {list(df.columns)}"
    
    def _create_summary_table(self, question: str, df: pd.DataFrame) -> str:
        """Create a summary table when full response is too large"""
        try:
            # For large datasets, create aggregated summary
            if 'count' in question.lower() or 'how many' in question.lower():
                # Group by first non-project_name column and count
                group_col = None
                for col in df.columns:
                    if col != 'project_name' and df[col].dtype == 'object':
                        group_col = col
                        break
                
                if group_col:
                    summary = df.groupby(group_col).size().reset_index(name='Count')
                    html = f"<table border='1'><tr><th>{group_col.replace('_', ' ').title()}</th><th>Count</th></tr>"
                    for _, row in summary.iterrows():
                        html += f"<tr><td>{row[group_col]}</td><td>{row['Count']}</td></tr>"
                    html += "</table>"
                    return html
            
            # Fallback: show first 20 rows
            sample_df = df.head(20)
            html = "<table border='1'><tr>"
            for col in sample_df.columns:
                html += f"<th>{col.replace('_', ' ').title()}</th>"
            html += "</tr>"
            
            for _, row in sample_df.iterrows():
                html += "<tr>"
                for col in sample_df.columns:
                    value = row[col] if pd.notna(row[col]) else 'N/A'
                    html += f"<td>{value}</td>"
                html += "</tr>"
            
            if len(df) > 20:
                html += f"<tr><td colspan='{len(sample_df.columns)}' style='text-align:center; font-style:italic;'>... and {len(df)-20} more rows</td></tr>"
            
            html += "</table>"
            return html
        except Exception as e:
            print(f"Summary table creation failed: {e}")
            return f"Large dataset with {len(df)} rows. Please ask a more specific question."
    
    def _validate_response(self, question: str, response: str) -> bool:
        """Validate if response addresses the question using LLM"""
        try:
            validation_prompt = f"""Question: "{question}"
Response: "{response}"

Does this response properly answer the question? Consider:
1. Are the relevant data columns included?
2. Does it address the specific analysis requested?
3. Is the data sufficient for the type of analysis asked?

Return only: YES or NO"""
            
            validation_response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a data analyst validator. Return only YES or NO."},
                    {"role": "user", "content": validation_prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            result = validation_response.choices[0].message.content.strip().upper()
            print(f"*** Validation Question: {question} ***")
            print(f"*** Validation Response Preview: {response[:200]}... ***")
            print(f"*** Validation Result: {result} ***")
            return result == "YES"
        except Exception as e:
            print(f"Validation failed: {e}")
            return True  # Default to valid if validation fails
    
    def _regenerate_response(self, question: str, df: pd.DataFrame) -> str:
        """Regenerate response with LLM guidance on required columns"""
        try:
            summary_csv = df.head(20).to_csv(index=False)
            available_columns = list(df.columns)
            
            prompt = f"""Question: "{question}"
Available data: {summary_csv}
Available columns: {available_columns}

Analyze what columns are needed to properly answer this question. Then create an HTML table with those specific columns. Include all relevant data for the analysis requested.

Return ONLY the HTML table."""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Analyze the question, identify required columns, and return only HTML table with relevant data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            if '<table' in content:
                content = content[content.find('<table'):]
            return content.replace('\n', '').strip()
        except Exception as e:
            print(f"Regeneration failed: {e}")
            return "Unable to generate proper response"

    def _is_document_format_request(self, question: str) -> bool:
        """Detect if user wants a formatted document (charter, report, etc.)"""
        q_lower = question.lower()
        document_keywords = [
            'charter', 'project charter', 'document', 'report', 'summary report',
            'provide charter', 'generate charter', 'create charter', 'format as charter'
        ]
        return any(keyword in q_lower for keyword in document_keywords)
    
    def _format_as_document(self, question: str, df: pd.DataFrame) -> str:
        """Format data as a proper document (project charter, etc.)"""
        try:
            if df.empty:
                return "No data available to create document."
            
            # Get the project data (assuming first row is the target project)
            project_data = df.iloc[0].to_dict()
            
            q_lower = question.lower()
            if 'charter' in q_lower:
                return self._create_project_charter(project_data)
            else:
                return self._create_generic_document(question, project_data)
                
        except Exception as e:
            print(f"Document formatting failed: {e}")
            return f"Unable to format document: {e}"
    
    def _create_project_charter(self, project_data: dict) -> str:
        """Create a properly formatted project charter"""
        try:
            charter = f"""
# PROJECT CHARTER

## PROJECT IDENTIFICATION
**Project Name:** {project_data.get('project_name', 'N/A')}  
**Project Manager:** {project_data.get('project_manager', 'N/A')}  
**Current Status:** {project_data.get('status', 'N/A')}  
**Completion:** {project_data.get('completion_pct', 'N/A')}  
**Current Phase:** {project_data.get('phase', 'N/A')}  

## PROJECT DESCRIPTION
{project_data.get('project_description', 'Project description not available.')}

## PROJECT SCOPE
**Project Type:** {project_data.get('project_type', 'Not specified')}  
**Complexity Level:** {project_data.get('complexity', 'Not specified')}  
**Department:** {project_data.get('department', 'Not specified')}  
**Region:** {project_data.get('region', 'Not specified')}  

## FINANCIAL INFORMATION
**Project Budget:** ${project_data.get('project_cost', 'N/A'):,} if project_data.get('project_cost') else 'N/A'  
**Actual Cost:** ${project_data.get('actual_cost', 'N/A'):,} if project_data.get('actual_cost') else 'N/A'  
**Cost Variance:** {project_data.get('cost_variance', 'N/A')}  

## PROJECT TIMELINE
**Start Date:** {project_data.get('start_date', 'N/A')}  
**Planned End Date:** {project_data.get('planned_end_date', 'N/A')}  
**Actual End Date:** {project_data.get('actual_end_date', 'N/A')}  
**Time Variance:** {project_data.get('time_variance_days', 'N/A')} days  

## RISK ASSESSMENT
**Priority Level:** {project_data.get('priority_level', 'Not specified')}  
**Risk Level:** {project_data.get('risk_level', 'Not specified')}  

## PROJECT AUTHORIZATION
This project charter authorizes the project team to proceed with project activities under the direction of the assigned Project Manager.

**Charter Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  
**Status:** {project_data.get('status', 'N/A')}  
"""
            return charter
        except Exception as e:
            return f"Error creating project charter: {e}"
    
    def _create_generic_document(self, question: str, project_data: dict) -> str:
        """Create a generic formatted document"""
        try:
            doc = f"""
# PROJECT DOCUMENT

## Project Information
**Name:** {project_data.get('project_name', 'N/A')}  
**Manager:** {project_data.get('project_manager', 'N/A')}  
**Status:** {project_data.get('status', 'N/A')}  

## Details
{project_data.get('project_description', 'No description available.')}

## Key Metrics
- **Completion:** {project_data.get('completion_pct', 'N/A')}
- **Phase:** {project_data.get('phase', 'N/A')}
- **Budget:** ${project_data.get('project_cost', 'N/A'):,} if project_data.get('project_cost') else 'N/A'
"""
            return doc
        except Exception as e:
            return f"Error creating document: {e}"
    
    def _direct_pandas_analysis(self, question: str, df: pd.DataFrame) -> Optional[str]:
        """Deprecated - using _handle_with_pandas instead"""
        return None

    def _get_user_tables(self, user_id: str) -> List[str]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (f"{user_id}_%",))
            tables = [row[0] for row in cursor.fetchall() if not row[0].endswith('_mapping')]
            conn.close()
            return tables
        except Exception as e:
            print(f"Error getting tables: {e}")
            return []

    def _get_detailed_schema(self, user_id: str) -> str:
        tables = self._get_user_tables(user_id)
        if not tables:
            return ""
        schema_parts = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                for table in tables:
                    cursor = conn.cursor()
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    col_names = [col[1] for col in columns]
                    schema_parts.append(f"Table: {table}\nColumns: {', '.join(col_names)}")
        except Exception as e:
            print(f"Schema error: {e}")
        return "\n\n".join(schema_parts)

    
    def _get_relevance_score(self, current_question: str, context: str, user_id: str) -> int:
        """Get LLM relevance score (0-100) for using previous result set"""
        try:
            prompt = f"""Current question: "{current_question}"
Previous conversation: {context}

How relevant (0-100%) is the current question to the previous result set?

Examples:
- Previous: "Show projects in progress" → Current: "Add project manager" → 95% (same projects, add column)
- Previous: "Show employees" → Current: "Filter only managers" → 90% (filter previous results)
- Previous: "Show projects" → Current: "Show incidents" → 10% (completely different data)
- Previous: "Budget analysis" → Current: "What's the weather?" → 0% (unrelated)

Return only the percentage number (0-100):"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Return only a number between 0-100 representing relevance percentage."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            # Extract number from response
            import re
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = int(numbers[0])
                return min(100, max(0, score))  # Clamp between 0-100
            return 0
            
        except Exception as e:
            print(f"Relevance scoring failed: {e}")
            return 0  # Default to no relevance on error
    
    def _llm_determine_filters(self, question: str, user_id: str) -> str:
        """Ask LLM to determine SQL WHERE conditions with data exploration"""
        try:
            schema_info = self._get_detailed_schema(user_id)
            
            # Step 1: LLM decides what columns to explore
            exploration_queries = self._llm_generate_exploration_queries(question, user_id, schema_info)
            
            # Step 2: Execute exploration queries
            exploration_results = self._execute_exploration_queries(exploration_queries, user_id)
            
            # Step 3: LLM generates filters based on exploration results
            return self._llm_generate_final_filters(question, schema_info, exploration_results)
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Return ONLY SQL WHERE conditions. No explanations, no examples, no text. Just the conditions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            conditions = response.choices[0].message.content.strip()
            print(f"*** LLM Generated Filters: {conditions} ***")
            return conditions
            
        except Exception as e:
            print(f"Filter generation failed: {e}")
            return ""
    
    def _get_llm_interpretation(self, question: str, df: pd.DataFrame) -> str:
        """Get LLM's interpretation of what the question is asking"""
        try:
            prompt = f"""Question: "{question}"
Data columns: {list(df.columns)}
Sample data: {df.head(2).to_dict()}

Explain in one sentence what filtering conditions this question requires:"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Explain what filtering conditions the question requires in one clear sentence."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Interpretation failed: {e}"
    
    def _is_aggregation_query(self, question: str) -> bool:
        """Use LLM to detect if query requires aggregation (GROUP BY) instead of filtering"""
        try:
            prompt = f"""Question: "{question}"

Does this question require data aggregation (GROUP BY, COUNT, SUM, AVG) or just filtering existing rows?

Aggregation examples:
- "Which employees are spread across multiple projects" (GROUP BY employee, COUNT DISTINCT projects > 1)
- "Which employees work on multiple projects" (GROUP BY employee, COUNT projects > 1)
- "Average cost by region" (GROUP BY region, AVG cost)
- "Top 5 projects by budget" (ORDER BY, LIMIT)

Filtering examples:
- "Show projects with status = completed" (WHERE status = 'completed')
- "List employees in Marketing department" (WHERE department = 'Marketing')

Key indicators for AGGREGATION:
- "multiple", "across", "spread", "count", "average", "sum", "top", "most", "least"

Return only: AGGREGATION or FILTERING"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Determine if question needs aggregation or filtering. Return only AGGREGATION or FILTERING."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().upper()
            is_aggregation = result == "AGGREGATION"
            print(f"*** LLM Aggregation Detection: {result} -> {is_aggregation} ***")
            return is_aggregation
            
        except Exception as e:
            print(f"LLM aggregation detection failed: {e}")
            # Fallback to simple keyword detection
            q_lower = question.lower()
            return any(word in q_lower for word in ['multiple', 'spread', 'across', 'average', 'count', 'sum', 'total', 'top', 'highest', 'lowest'])
    
    def _get_sample_data(self, user_id: str) -> str:
        """Get sample data values to make LLM data-aware"""
        try:
            tables = self._get_user_tables(user_id)
            if not tables:
                return "No sample data available"
            
            main_table = None
            for table in tables:
                if 'Project_Management_Dataset' in table:
                    main_table = table
                    break
            
            if not main_table:
                main_table = tables[0]
            
            conn = sqlite3.connect(self.db_path)
            sample_df = pd.read_sql_query(f"SELECT * FROM {main_table} LIMIT 3", conn)
            conn.close()
            
            # Show unique values for key columns
            sample_info = []
            key_columns = ['status', 'priority_level', 'complexity', 'phase']
            
            for col in key_columns:
                if col in sample_df.columns:
                    unique_vals = sample_df[col].unique()[:5]  # First 5 unique values
                    sample_info.append(f"{col}: {list(unique_vals)}")
            
            return "\n".join(sample_info)
            
        except Exception as e:
            return f"Sample data error: {e}"
    
    def _llm_analyze_empty_result(self, question: str, filters: str, user_id: str) -> str:
        """LLM analyzes why no data was found and suggests alternatives"""
        try:
            # Let LLM run diagnostic queries
            diagnostic_queries = self._llm_generate_diagnostic_queries(question, filters, user_id)
            diagnostic_results = self._execute_diagnostic_queries(diagnostic_queries, user_id)
            
            prompt = f"""Question: "{question}"
Filters applied: {filters}
Diagnostic results: {diagnostic_results}

No data was found. Analyze why and suggest what the user should ask instead.
Provide a helpful explanation and alternative query suggestions."""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Analyze why no data was found and provide helpful suggestions for alternative queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"No data found for the specified criteria. Error in analysis: {e}"
    
    def _llm_generate_diagnostic_queries(self, question: str, filters: str, user_id: str) -> List[str]:
        """LLM generates diagnostic queries to understand why no data was found"""
        try:
            schema_info = self._get_detailed_schema(user_id)
            
            prompt = f"""Question: "{question}"
Filters that returned no data: {filters}
Schema: {schema_info}

Generate 2-3 diagnostic SQL queries to understand why no data was found:
1. Check if the filtered values exist in the data
2. Show what values are actually available
3. Count total rows

Return only SQL queries, one per line:"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Generate diagnostic SQL queries to investigate empty results. Return only SQL statements."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            queries = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
            print(f"*** LLM Generated Diagnostic Queries: {queries} ***")
            return queries[:3]
            
        except Exception as e:
            print(f"Diagnostic query generation failed: {e}")
            return []
    
    def _execute_diagnostic_queries(self, queries: List[str], user_id: str) -> Dict[str, Any]:
        """Execute diagnostic queries to understand empty results"""
        results = {}
        try:
            conn = sqlite3.connect(self.db_path)
            for i, query in enumerate(queries):
                try:
                    result = pd.read_sql_query(query, conn)
                    results[f"diagnostic_{i+1}"] = result.to_dict('records')
                    print(f"*** Diagnostic Query {i+1}: {query} ***")
                    print(f"*** Result: {result.to_dict('records')} ***")
                except Exception as e:
                    print(f"*** Diagnostic Query {i+1} failed: {e} ***")
                    results[f"diagnostic_{i+1}"] = f"Error: {e}"
            conn.close()
        except Exception as e:
            print(f"Diagnostic execution failed: {e}")
        return results
    
    def _llm_generate_exploration_queries(self, question: str, user_id: str, schema_info: str) -> List[str]:
        """LLM generates SQL queries to explore relevant column values"""
        try:
            prompt = f"""Question: "{question}"
Schema: {schema_info}

To answer this question properly, what column values should I explore?
Generate 2-3 SQL queries to get unique values from relevant columns.

Examples:
- For "incomplete projects": SELECT DISTINCT status FROM table_name
- For "high priority tasks": SELECT DISTINCT priority_level FROM table_name
- For "budget analysis": SELECT MIN(cost_variance), MAX(cost_variance) FROM table_name

Return only the SQL queries, one per line:"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Generate SQL exploration queries. Return only SQL statements, one per line."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            queries = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
            print(f"*** LLM Generated Exploration Queries: {queries} ***")
            return queries[:3]  # Limit to 3 queries
            
        except Exception as e:
            print(f"Exploration query generation failed: {e}")
            return []
    
    def _execute_exploration_queries(self, queries: List[str], user_id: str) -> Dict[str, Any]:
        """Execute exploration queries and return results"""
        results = {}
        try:
            conn = sqlite3.connect(self.db_path)
            for i, query in enumerate(queries):
                try:
                    result = pd.read_sql_query(query, conn)
                    results[f"query_{i+1}"] = result.to_dict('records')
                    print(f"*** Exploration Query {i+1}: {query} ***")
                    print(f"*** Result: {result.to_dict('records')} ***")
                except Exception as e:
                    print(f"*** Exploration Query {i+1} failed: {e} ***")
                    results[f"query_{i+1}"] = f"Error: {e}"
            conn.close()
        except Exception as e:
            print(f"Exploration execution failed: {e}")
        return results
    
    def _llm_generate_final_filters(self, question: str, schema_info: str, exploration_results: Dict[str, Any]) -> str:
        """LLM generates final filters based on exploration results"""
        try:
            prompt = f"""Question: "{question}"
Exploration Results: {exploration_results}

Analyze what the question is asking for and generate appropriate SQL WHERE conditions.

Examples:
- "most complex projects" → complexity = 'High'
- "incomplete projects" → status IN ('In - Progress', 'On - Hold')
- "over budget projects" → cost_variance < 0
- "high priority tasks" → priority_level = 'High'

Focus on the filtering criteria, not listing all possible values.
Return ONLY the conditions:"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Return ONLY SQL conditions. No explanations, no code blocks, no WHERE keyword. Just the conditions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            conditions = response.choices[0].message.content.strip()
            print(f"*** Final LLM Generated Filters: {conditions} ***")
            
            # Clean up the conditions - handle multiple lines intelligently
            if '\n' in conditions:
                lines = [line.strip() for line in conditions.split('\n') if line.strip()]
                # For questions about "most X", usually we want just the filter criteria
                if 'most' in question.lower() or 'highest' in question.lower():
                    # Keep only the main filtering condition (usually the last meaningful line)
                    conditions = lines[-1] if lines else conditions
                else:
                    # Remove duplicate conditions and join with AND
                    unique_conditions = list(dict.fromkeys(lines))
                    conditions = ' AND '.join(unique_conditions)
                print(f"*** Cleaned Filters: {conditions} ***")
            
            return conditions
            
        except Exception as e:
            print(f"Final filter generation failed: {e}")
            return ""
    
    def _get_previous_result_set(self, user_id: str) -> Optional[pd.DataFrame]:
        """Get the result set from the most recent interaction"""
        if user_id not in self.conversation_history or not self.conversation_history[user_id]:
            return None
        
        last_interaction = self.conversation_history[user_id][-1]
        return last_interaction.get('result_df')
    
    def _get_conversation_context(self, user_id: str, current_question: str) -> str:
        """Get relevant conversation context for the current question"""
        if user_id not in self.conversation_history:
            return ""
        
        history = self.conversation_history[user_id]
        if not history:
            return ""
        
        # Get last 2 interactions for context
        recent_history = history[-2:]
        context_parts = []
        
        for i, interaction in enumerate(recent_history, 1):
            context_parts.append(f"Previous Q{i}: {interaction['question']}")
            # Include brief response summary
            response_preview = interaction['response'][:200] + "..." if len(interaction['response']) > 200 else interaction['response']
            context_parts.append(f"Previous A{i}: {response_preview}")
        
        return "\n".join(context_parts)
    
    def _enhance_question_with_context(self, question: str, context: str) -> str:
        """Let LLM decide if question is related to previous context"""
        if not context or not self.groq_client:
            return question
        
        try:
            prompt = f"""Current question: "{question}"
Previous conversation: {context}

Is the current question related to or building upon the previous conversation? 
Return only: YES or NO"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Determine if the current question relates to previous conversation. Return only YES or NO."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0.1
            )
            
            is_related = response.choices[0].message.content.strip().upper() == "YES"
            print(f"*** LLM Context Decision: {'RELATED' if is_related else 'NEW QUESTION'} ***")
            
            if is_related:
                # Check relevance percentage for using previous result set
                previous_df = self._get_previous_result_set(user_id)
                if previous_df is not None:
                    relevance_score = self._get_relevance_score(question, context, user_id)
                    print(f"*** Relevance score: {relevance_score}% ***")
                    
                    if relevance_score >= 70:  # High relevance threshold
                        print(f"*** Using previous result set ({len(previous_df)} rows) for follow-up ***")
                        self._temp_result_df = previous_df
                    else:
                        print("*** Low relevance, fetching fresh data ***")
                        self._temp_result_df = None
                else:
                    self._temp_result_df = None
                return f"{question} (Context: {context})"
            else:
                print("*** Treating as new independent question ***")
                self._temp_result_df = None
                return question
                
        except Exception as e:
            print(f"Context decision failed: {e}")
            return question
    
    def _store_conversation(self, user_id: str, question: str, response: str, columns: List[str], result_df: pd.DataFrame = None):
        """Store conversation interaction with result set"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        interaction = {
            "question": question,
            "response": response,
            "columns": columns,
            "result_df": result_df.copy() if result_df is not None and not result_df.empty else None,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        self.conversation_history[user_id].append(interaction)
        
        # Keep only last 3 interactions to avoid memory bloat
        if len(self.conversation_history[user_id]) > 3:
            self.conversation_history[user_id] = self.conversation_history[user_id][-3:]
        
        print(f"*** Stored conversation with result set ({len(result_df) if result_df is not None else 0} rows) for {user_id} ***")
    
    def _llm_correct_filter_and_analyze(self, question: str, failed_filters: str, df: pd.DataFrame, user_id: str) -> str:
        """LLM analyzes unfiltered data, corrects filters, and provides analysis"""
        try:
            # Show LLM the actual data to understand what's available
            sample_data = df.head(10).to_csv(index=False)
            unique_values = {}
            
            # Get unique values for key columns
            for col in df.columns:
                if df[col].dtype == 'object' and col != 'project_name':
                    unique_vals = df[col].unique()[:10]  # First 10 unique values
                    unique_values[col] = list(unique_vals)
            
            prompt = f"""Question: "{question}"
Failed filters: {failed_filters}

Actual data available:
{sample_data}

Unique values in key columns:
{unique_values}

The original filters didn't match any data. Create a clean HTML table with ONLY the rows that match the question. 

IMPORTANT:
- Include ONLY data rows, no empty rows
- Use proper table structure with <th> for headers and <td> for data
- No nested tables or extra formatting
- Return ONLY the <table> element

Example format:
<table border='1'>
<tr><th>Column1</th><th>Column2</th></tr>
<tr><td>Value1</td><td>Value2</td></tr>
</table>"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Create clean HTML table with filtered data. No empty rows, no nested tables, no explanations. Return only <table> element."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Extract only the HTML table part
            import re
            
            # Find the table in the response
            table_match = re.search(r'<table[^>]*>.*?</table>', llm_response, re.DOTALL | re.IGNORECASE)
            
            if table_match:
                clean_table = table_match.group()
                print(f"*** Extracted table length: {len(clean_table)} ***")
                return clean_table
            else:
                print(f"*** No table found in response, returning raw response ***")
                return llm_response.strip()
            
        except Exception as e:
            print(f"Filter correction failed: {e}")
            return f"Filter correction failed. Original filters '{failed_filters}' didn't match any data. Please check your criteria."
