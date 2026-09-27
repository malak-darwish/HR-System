"""
Shared state schema for the multi-agent HR Interview System.
All four agents (Screener, Interviewer, Verification, Recruiter) read/write to this state.
"""

import operator
from typing import Annotated, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# --- Pydantic model for individual CV claims (used by Screener & Verification) ---
class Claim(BaseModel):
    """A single checkable claim extracted from the candidate's CV."""
    claim_text: str = Field(description="The raw claim, e.g. '3 years at Company X'")
    category: str = Field(description="Category: 'experience', 'skill', 'education', 'project', 'certification'")
    verified: Optional[bool] = Field(default=None, description="True/False after verification, None if unchecked")
    confidence: Optional[float] = Field(default=None, description="Verification confidence 0.0–1.0")
    source: Optional[str] = Field(default=None, description="Where this was verified, e.g. 'github', 'interview_answer'")


# --- Main shared state ---
class HRState(TypedDict):
    """
    The shared state passed between all agents in the LangGraph pipeline.
    
    Annotated[list, operator.add] ensures that when a node returns a list,
    it gets APPENDED to the existing list rather than replacing it.
    This is critical for questions/answers growing over time (Requirement C).
    """

    # --- Input fields (set at the start) ---
    cv_raw: str                                      # raw CV text uploaded by candidate
    job_description: str                             # target job description

    # --- Screener Agent (Person A) writes these ---
    cv_parsed: Dict                                  # structured CV: {"experience": ..., "skills": ..., etc.}
    match_score: float                               # similarity score between CV and JD (0.0–1.0)
    screening_passed: bool                           # True = proceed to interview, False = reject
    claims: List[Claim]                              # atomic claims extracted from CV

    # --- Interviewer Agent (Person B) writes these ---
    questions: Annotated[list, operator.add]          # interview questions — grows each turn
    answers: Annotated[list, operator.add]            # candidate answers — grows each turn
    interview_complete: bool                         # True when interview is finished

    # --- Verification Agent (Person C) writes these ---
    consistency_flags: Dict[str, bool]               # {"claim_text": True/False} — True means inconsistency found
    follow_up_needed: bool                           # True = send candidate back to Interviewer for clarification

    # --- Recruiter Agent (Person D) writes these ---
    overall_score: float                             # weighted combination of all scores
    final_decision: str                              # "hire", "reject", or "waitlist"
    decision_reasoning: str                          # explanation of the decision