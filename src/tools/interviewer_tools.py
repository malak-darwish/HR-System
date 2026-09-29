"""
Interviewer Agent Tools (Person B)
Two LangChain tools used by the Interviewer Agent:
  1. get_cv_section_tool  — retrieves a specific section from the parsed CV
  2. question_bank_tool   — returns question templates filtered by category and keyword
"""

from langchain_core.tools import tool
from typing import List


# ──────────────────────────────────────────────
# Tool 1: Get CV Section
# ──────────────────────────────────────────────
@tool
def get_cv_section_tool(section: str, cv_parsed: dict) -> str:
    """
    Retrieve a specific section from the candidate's parsed CV.
    
    Args:
        section: The CV section to retrieve. 
                 Valid sections: 'experience', 'skills', 'education', 'projects', 'certifications', 'contact'
        cv_parsed: The parsed CV dictionary from state.
        
    Returns:
        The content of that CV section as a string, or a message if the section is not found.
    """
    # Normalize the section name to lowercase
    section = section.strip().lower()
    
    if section in cv_parsed:
        content = cv_parsed[section]
        # Handle both string and list/dict content
        if isinstance(content, list):
            return "\n".join(f"- {item}" for item in content)
        elif isinstance(content, dict):
            return "\n".join(f"{k}: {v}" for k, v in content.items())
        else:
            return str(content)
    else:
        available = ", ".join(cv_parsed.keys()) if cv_parsed else "none"
        return f"Section '{section}' not found in CV. Available sections: {available}"


# ──────────────────────────────────────────────
# Tool 2: Question Bank
# ──────────────────────────────────────────────

# Hardcoded question templates organized by category
QUESTION_TEMPLATES = {
    "technical": [
        "Can you explain how you implemented {keyword} in your projects? Walk me through the technical approach.",
        "What challenges did you face when working with {keyword}, and how did you solve them?",
        "How would you rate your proficiency in {keyword} on a scale of 1–10, and what's the most complex thing you've built with it?",
        "Describe the architecture of a system you built using {keyword}.",
        "What best practices do you follow when working with {keyword}?",
    ],
    "behavioral": [
        "Describe a time you faced a significant challenge related to {keyword}. How did you handle it?",
        "Tell me about a time you had to learn {keyword} quickly under pressure. What was your approach?",
        "Give an example of a conflict you had with a teammate regarding {keyword}. How did you resolve it?",
        "Describe a situation where you had to make a difficult decision involving {keyword}.",
        "Tell me about a time you failed at something related to {keyword}. What did you learn?",
    ],
    "experience": [
        "Walk me through your role at {keyword} and your key contributions there.",
        "What was the most impactful project you worked on at {keyword}?",
        "How did your experience at {keyword} prepare you for this position?",
        "What did you learn during your time working with {keyword} that you still apply today?",
        "Describe your day-to-day responsibilities when you were working on {keyword}.",
    ],
}


@tool
def question_bank_tool(category: str, keyword: str = "") -> List[str]:
    """
    Get interview question templates filtered by category and keyword.
    
    Args:
        category: The type of questions to retrieve. 
                  Options: 'technical', 'behavioral', 'experience'
        keyword: Optional keyword to fill into the question templates 
                 (e.g. a skill name, company name, or project name).
                 
    Returns:
        A list of 2-3 personalized question templates.
    """
    category = category.strip().lower()
    
    if category not in QUESTION_TEMPLATES:
        available = ", ".join(QUESTION_TEMPLATES.keys())
        return [f"Unknown category '{category}'. Available categories: {available}"]
    
    templates = QUESTION_TEMPLATES[category]
    
    # Pick 2-3 templates
    selected = templates[:3]
    
    # Fill in keyword if provided
    if keyword:
        selected = [q.format(keyword=keyword) for q in selected]
    else:
        # Replace {keyword} with a generic placeholder
        selected = [q.format(keyword="your relevant experience") for q in selected]
    
    return selected