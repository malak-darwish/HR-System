"""
state.py — Shared LangGraph State for HR Multi-Agent System
Owned by: Person A (Screener Agent)
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class ParsedCV(BaseModel):
    """Structured representation of a parsed CV."""
    name: str = ""
    email: str = ""
    phone: str = ""
    education: List[str] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    github_username: Optional[str] = None
    years_of_experience: Optional[float] = None


class Claim(BaseModel):
    """
    One atomic checkable claim extracted from the CV.
    Seeded by Person A, verified by Person C.
    """
    text: str                         # e.g. "3 years at Company X"
    category: str                     # "experience" | "skill" | "education" | "project"
    verified: Optional[bool] = None   # filled by Person C
    confidence: Optional[float] = None  # 0.0 - 1.0, filled by Person C
    source: Optional[str] = None      # e.g. "github", "interview_answer"


class HRState(BaseModel):
    """
    Full shared state for the HR LangGraph multi-agent system.
    """

    # ── INPUTS (set once before graph runs) ──────────────
    cv_text: str = ""
    job_description: str = ""

                                    
    # ── PERSON A — Screener Agent ─────────────────────────
    parsed_cv: Optional[ParsedCV] = None
    match_score: Optional[float] = None        # float 0.0–1.0
    screening_passed: Optional[bool] = None    # True = go to interview, False = reject
    claims: List[Claim] = Field(default_factory=list)


    # ── PERSON B — Interviewer Agent ──────────────────────
    questions: List[str] = Field(default_factory=list)
    answers: List[str] = Field(default_factory=list)
    interview_scores: List[float] = Field(default_factory=list)
    

    # ── PERSON C — Verification Agent ─────────────────────
    consistency_flags: Dict[str, bool] = Field(default_factory=dict)
    follow_up_needed: Optional[bool] = None
    verification_notes: Optional[str] = None
    
    
    # ── PERSON D — Recruiter Agent ────────────────────────
    overall_score: Optional[float] = None
    final_decision: Optional[str] = None       # "hire" | "reject" | "waitlist"
    decision_reasoning: Optional[str] = None