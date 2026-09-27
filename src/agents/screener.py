"""Person A's Screener, adapted to the shared state contract."""

from src.github_refs import repository_urls
from src.state import Claim, HRState
from src.tools.screener_tools import match_score_tool, parse_cv_tool

SCREENING_THRESHOLD = 0.60


def screener_node(state: HRState) -> dict:
    if not state.cv_text.strip() or not state.job_description.strip():
        raise ValueError("Both CV text and job description are required.")
    parsed = parse_cv_tool.invoke({"cv_text": state.cv_text})
    match = match_score_tool.invoke({
        "parsed_cv": parsed["parsed_cv"], "job_description": state.job_description,
    })
    claims = [Claim(claim_id=f"C{i}", **claim).model_dump()
              for i, claim in enumerate(parsed["claims"], 1)]
    return {
        "parsed_cv": parsed["parsed_cv"], "claims": claims,
        "github_repository_urls": repository_urls(state.cv_text),
        "match_score": match["score"], "screening_reasoning": match["reasoning"],
        "screening_passed": match["score"] >= SCREENING_THRESHOLD,
    }
