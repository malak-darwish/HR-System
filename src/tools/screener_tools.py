"""
screener_tools.py — Tools for the Screener Agent (Person A)

Tools:
    1. parse_cv_tool    — one Gemini SOM call returning ParsedCV + atomic claims.
                          claim_ids are assigned in Python, never by the LLM.
    2. match_score_tool — one Gemini SOM call returning a bounded match score
                          and the reasoning behind it.
"""

from typing import Literal

from langchain_core.tools import tool
from pydantic import Field

from llm import structured_call
from state import Claim, ParsedCV, Record, Text


# ── SOM schemas (what the LLM is allowed to fill in) ─────
class ExtractedClaim(Record):
    """Only text + category. IDs and verification fields are not the Screener's job."""
    text: Text
    category: Literal["experience", "skill", "education", "project"]


class ParsedCVWithClaims(Record):
    parsed_cv: ParsedCV
    claims: list[ExtractedClaim] = Field(default_factory=list)


class MatchScore(Record):
    score: float  # clamped to [0, 1] in Python, so a stray 1.1 can't crash the run
    reasoning: Text


# ── Prompts ──────────────────────────────────────────────
PARSE_INSTRUCTION = """
You are an expert CV analyzer. Given the supplied cv_text, do TWO things.

PART 1 — parsed_cv:
- name, email, phone: exactly as written; empty string if absent
- education: each degree/certification as a separate string
- experience: each job/role as a separate string (e.g. "3 years at Google as SWE")
- skills: each technical skill as a separate string (e.g. "Python", "AWS")
- projects: each project as a separate string
- github_username: the bare username only (not a URL), or null if absent
- years_of_experience: total years of work experience as a number, or null if unclear

PART 2 — claims (atomic, checkable facts):
- Each claim is ONE specific verifiable fact
- category is exactly one of: "experience", "skill", "education", "project"
- No subjective claims ("hardworking", "team player")
- Never invent anything that is not in the CV

Examples:
- text: "3 years at Google as Software Engineer", category: "experience"
- text: "Python", category: "skill"
- text: "BS Computer Science AUB 2020", category: "education"
- text: "Built a recommendation engine", category: "project"
""".strip()

MATCH_INSTRUCTION = """
You are an expert HR screener. Compare the supplied parsed CV to the job
description and return a match score between 0.0 and 1.0 with brief reasoning.

Scoring guide:
- 0.0 - 0.3: poor match, missing key requirements
- 0.4 - 0.6: partial match, some relevant skills
- 0.7 - 0.8: good match, meets most requirements
- 0.9 - 1.0: excellent match, meets all requirements

Judge only job-relevant content. Do not reward writing polish or confidence.
""".strip()


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


# ── Tool 1: parse_cv_tool ────────────────────────────────
@tool
def parse_cv_tool(cv_text: str) -> dict:
    """Parse raw CV text into a ParsedCV and a list of atomic Claims with unique IDs.

    Args:
        cv_text: Raw CV text from the candidate.

    Returns:
        dict with keys 'parsed_cv' and 'claims' (both JSON-safe dicts).
    """
    if not cv_text.strip():
        # An empty CV makes the model invent a candidate ("Jane Doe").
        raise ValueError("cv_text is empty; refusing to parse.")

    result = structured_call(ParsedCVWithClaims, PARSE_INSTRUCTION, {"cv_text": cv_text})

    # Deterministic, unique IDs; exact duplicates dropped.
    claims, seen = [], set()
    for item in result.claims:
        key = (item.category, _norm(item.text))
        if key in seen:
            continue
        seen.add(key)
        claims.append(Claim(
            claim_id=f"C{len(claims) + 1}",
            text=item.text,
            category=item.category,
        ))

    return {
        "parsed_cv": result.parsed_cv.model_dump(),
        "claims": [claim.model_dump() for claim in claims],
    }


# ── Tool 2: match_score_tool ─────────────────────────────
@tool
def match_score_tool(parsed_cv: dict, job_description: str) -> dict:
    """Score how well a parsed CV matches a job description (0.0 - 1.0).

    Args:
        parsed_cv: dict from ParsedCV.
        job_description: Job description text.

    Returns:
        dict with 'score' (float in [0, 1]) and 'reasoning' (str).
    """
    if not job_description.strip():
        raise ValueError("job_description is empty; cannot compute a match score.")

    relevant_cv = {k: parsed_cv.get(k, []) for k in ("skills", "experience", "education", "projects")}
    result = structured_call(MatchScore, MATCH_INSTRUCTION,
                             {"parsed_cv": relevant_cv, "job_description": job_description})

    return {
        "score": float(max(0.0, min(1.0, result.score))),
        "reasoning": result.reasoning,
    }