from typing import List, Dict

def generate_followup_suggestions(question: str, chunks: List[Dict]) -> List[str]:
    """Generate contextual follow-up questions"""
    
    question_lower = question.lower()
    
    # Get data types from chunks
    has_projects = any('project' in chunk.get('content', '').lower() for chunk in chunks)
    has_employees = any('employee' in chunk.get('content', '').lower() for chunk in chunks)
    has_incidents = any('incident' in chunk.get('content', '').lower() for chunk in chunks)
    
    followups = []
    
    # Employee-related followups
    if 'employee' in question_lower or 'staff' in question_lower:
        followups.extend([
            "Which employees have the highest productivity scores?",
            "Show me employees by department",
            "What skills are most common in our team?"
        ])
    
    # Project-related followups
    elif 'project' in question_lower:
        followups.extend([
            "Which projects are behind schedule?",
            "Show me high-risk projects",
            "What's the total budget across all projects?"
        ])
    
    # Risk-related followups
    elif 'risk' in question_lower:
        followups.extend([
            "Show me projects by risk level",
            "What are the most common risk types?",
            "Which project managers handle high-risk projects?"
        ])
    
    # General data exploration
    else:
        if has_projects:
            followups.append("List all active projects")
        if has_employees:
            followups.append("Show me employee distribution by role")
        if has_incidents:
            followups.append("What are the recent incidents?")
    
    return followups[:3]  # Return max 3 suggestions