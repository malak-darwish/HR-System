"""
screener_tools.py — Tools for the Screener Agent (Person A)
Tools:
    1. parse_cv_tool    — extracts structured CV data (ParsedCV) AND atomic claims 
                          (List[Claim]) in one LLM call using Gemini SOM
    2. match_score_tool — computes similarity score between parsed CV and JD 
                          using Google embeddings + cosine similarity
"""

import os
import sys
from dotenv import load_dotenv

# ── Load .env FIRST before anything else ─────────────────
load_dotenv()

from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import our state models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state import ParsedCV, Claim

# ── LLM Setup ────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

# ── Combined Output Schema for parse_cv_tool ─────────────
class ParsedCVWithClaims(BaseModel):
    """
    Combined output of parse_cv_tool.
    Returns both the structured CV AND atomic claims in one LLM call.
    """
    parsed_cv: ParsedCV = Field(description="Structured CV data extracted from raw text")
    claims: List[Claim] = Field(
        default_factory=list,
        description="List of atomic, checkable facts extracted from the CV"
    )


# ── Tool 1: parse_cv_tool ─────────────────────────────────
@tool
def parse_cv_tool(cv_text: str) -> dict:
    """
    Parses raw CV text into:
    - A structured ParsedCV object (name, email, skills, experience, etc.)
    - A list of atomic Claim objects (verifiable facts from the CV)
    
    Both are extracted in ONE LLM call using Gemini Structured Output Mode.
    
    Args:
        cv_text: Raw CV text from candidate
        
    Returns:
        dict with keys 'parsed_cv' and 'claims'
    """
    # One LLM call returns BOTH ParsedCV and Claims together
    structured_llm = llm.with_structured_output(ParsedCVWithClaims)

    prompt = f"""
    You are an expert CV analyzer. Given the raw CV text below, do TWO things:

    PART 1 — Extract structured CV data:
    - name: candidate full name
    - email: candidate email
    - phone: candidate phone number
    - education: list each degree/certification as a separate string
    - experience: list each job/role as a separate string (e.g. "3 years at Google as SWE")
    - skills: list each technical skill as a separate string (e.g. "Python", "AWS", "React")
    - projects: list each project as a separate string
    - github_username: extract github username if mentioned, otherwise null
    - years_of_experience: total years of work experience as a float, otherwise null

    PART 2 — Extract atomic checkable claims:
    - Each claim must be ONE specific verifiable fact
    - category must be one of: "experience", "skill", "education", "project"
    - Do NOT include subjective claims (e.g. "hardworking", "team player")
    - Only include claims that CAN be verified

    Examples of good claims:
    - text: "3 years at Google as Software Engineer", category: "experience"
    - text: "Python", category: "skill"
    - text: "BS Computer Science AUB 2020", category: "education"
    - text: "Built a recommendation engine", category: "project"

    CV:
    {cv_text}
    """

    result: ParsedCVWithClaims = structured_llm.invoke(prompt)

    return {
        "parsed_cv": result.parsed_cv.model_dump(),
        "claims": [claim.model_dump() for claim in result.claims]
    }


# ── Tool 2: match_score_tool ──────────────────────────────
@tool
def match_score_tool(parsed_cv: dict, job_description: str) -> float:
    """
    Computes a similarity score (0.0 - 1.0) between a parsed CV and a job description
    using Gemini LLM judgment.

    Args:
        parsed_cv: dict from ParsedCV (structured CV sections)
        job_description: Job description text

    Returns:
        float between 0.0 and 1.0 representing match quality
    """
    from pydantic import BaseModel

    class MatchScore(BaseModel):
        score: float
        reasoning: str

    # Build clean CV summary from relevant sections only
    relevant_cv = f"""
    Skills: {', '.join(parsed_cv.get('skills', []))}
    Experience: {', '.join(parsed_cv.get('experience', []))}
    Education: {', '.join(parsed_cv.get('education', []))}
    Projects: {', '.join(parsed_cv.get('projects', []))}
    """.strip()

    structured_llm = llm.with_structured_output(MatchScore)

    prompt = f"""
    You are an expert HR screener. Compare the candidate's CV to the job description
    and return a match score between 0.0 and 1.0.

    Scoring guide:
    - 0.0 - 0.3: Poor match, missing key requirements
    - 0.4 - 0.6: Partial match, some relevant skills
    - 0.7 - 0.8: Good match, meets most requirements
    - 0.9 - 1.0: Excellent match, meets all requirements

    CV Summary:
    {relevant_cv}

    Job Description:
    {job_description}

    Return a score (float 0.0-1.0) and brief reasoning.
    """

    result: MatchScore = structured_llm.invoke(prompt)

    # Clamp between 0.0 and 1.0 just in case
    score = float(max(0.0, min(1.0, result.score)))

    return score