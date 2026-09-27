"""Person A's combined CV extraction and LLM-based job matching tools."""

from typing import Literal

from langchain_core.tools import tool
from pydantic import Field

from src.llm import structured_call
from src.state import ParsedCV, Record, Score, Text


class ExtractedClaim(Record):
    text: Text
    category: Literal["experience", "skill", "education", "project"]


class ParsedCVWithClaims(Record):
    parsed_cv: ParsedCV
    claims: list[ExtractedClaim] = Field(default_factory=list)


class MatchScore(Record):
    score: Score
    reasoning: Text


@tool
def parse_cv_tool(cv_text: str) -> dict:
    """Extract structured CV data and atomic, objective claims in one LLM call."""
    result = structured_call(ParsedCVWithClaims,
        "Extract the CV's contact fields, education, experience, skills, projects, GitHub "
        "username, and years of experience. Leave absent fields empty or null. Extract "
        "atomic checkable claims, one fact per claim, categorized as experience, skill, "
        "education, or project. Omit subjective praise and duplicate claims. Preserve "
        "repository URLs when present. Set github_username only for a profile explicitly "
        "identified as the candidate's own; do not infer it from third-party repository "
        "references. Do not assess truth at this extraction stage.",
        {"cv_text": cv_text})
    return result.model_dump()


@tool
def match_score_tool(parsed_cv: dict, job_description: str) -> dict:
    """Return a bounded CV/job match score and a short explanation using Gemini."""
    relevant = {k: parsed_cv.get(k, []) for k in ("skills", "experience", "education", "projects")}
    result = structured_call(MatchScore,
        "Compare the CV to the job requirements. Score 0-0.3 for a poor match, "
        "0.4-0.6 for a partial match, 0.7-0.8 for a good match, 0.9-1 for an excellent "
        "match. Assess only job-relevant qualifications; ignore identity/contact details. "
        "Return a score and concise reasons identifying missing requirements.",
        {"cv": relevant, "job_description": job_description})
    return result.model_dump()
