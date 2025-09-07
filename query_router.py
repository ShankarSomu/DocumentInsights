from typing import Dict, List, Optional
from groq import Groq
import os

class QueryRouter:
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
    
    def route_query(self, question: str, available_files: List[str]) -> Dict:
        """Route query to appropriate file(s) and determine query type"""
        
        question_lower = question.lower()
        
        # Rule-based routing (fast and reliable)
        routing_result = self._rule_based_routing(question_lower, available_files)
        
        # If rule-based routing is uncertain, use LLM
        if routing_result["confidence"] < 0.8:
            llm_result = self._llm_routing(question, available_files)
            if llm_result:
                routing_result = llm_result
        
        return routing_result
    
    def _rule_based_routing(self, question_lower: str, available_files: List[str]) -> Dict:
        """Fast rule-based query routing"""
        
        # Define keyword mappings
        file_keywords = {
            "projects": ["project", "risk", "milestone", "deliverable", "phase", "portfolio", "program"],
            "employees": ["employee", "staff", "team", "skill", "capacity", "manager", "hr", "people"],
            "incidents": ["incident", "issue", "ticket", "bug", "problem", "severity", "resolution"]
        }
        
        # Score each domain
        domain_scores = {}
        for domain, keywords in file_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Find matching files
        target_files = []
        target_domain = None
        confidence = 0.5
        
        if domain_scores:
            # Get highest scoring domain
            target_domain = max(domain_scores, key=domain_scores.get)
            confidence = min(0.9, 0.6 + (domain_scores[target_domain] * 0.1))
            
            # Map domain to actual files
            domain_file_mapping = {
                "projects": ["Project Management Dataset.csv"],
                "employees": ["employee_data_pyramid.csv"],
                "incidents": ["incident.csv"]
            }
            
            potential_files = domain_file_mapping.get(target_domain, [])
            target_files = [f for f in potential_files if f in available_files]
        
        # Determine query type
        query_type = self._determine_query_type(question_lower)
        
        return {
            "target_files": target_files,
            "target_domain": target_domain,
            "query_type": query_type,
            "confidence": confidence,
            "reasoning": f"Rule-based: Found {len(target_files)} files for domain '{target_domain}'"
        }
    
    def _determine_query_type(self, question_lower: str) -> str:
        """Determine the type of query being asked"""
        
        if any(word in question_lower for word in ["list", "show", "all", "every"]):
            return "list_all"
        elif any(word in question_lower for word in ["count", "how many", "total"]):
            return "count"
        elif any(word in question_lower for word in ["risk", "problem", "issue"]):
            return "risk_analysis"
        elif any(word in question_lower for word in ["status", "progress"]):
            return "status_check"
        else:
            return "general_query"
    
    def _llm_routing(self, question: str, available_files: List[str]) -> Optional[Dict]:
        """Use LLM for complex routing decisions"""
        
        files_list = ", ".join(available_files)
        
        prompt = f"""Route this query to the most appropriate data source(s):

Query: {question}

Available files: {files_list}

File descriptions:
- Project Management Dataset.csv: Contains project information, risks, milestones, phases
- employee_data_pyramid.csv: Contains employee data, skills, capacity, assignments  
- incident.csv: Contains incident reports, issues, tickets, resolutions

Return JSON only:
{{
    "target_files": ["filename.csv"],
    "reasoning": "why this file was chosen",
    "confidence": 0.85
}}"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a query router. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            import json
            import re
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                result["query_type"] = self._determine_query_type(question.lower())
                return result
                
        except Exception as e:
            print(f"LLM routing failed: {e}")
        
        return None