"""Provisional decision rules for Person D's Recruiter Agent."""

from typing import Literal, Optional, TypedDict


class DecisionResult(TypedDict):
    final_decision: Literal["hire", "reject", "waitlist"]
    decision_reasoning: str


def decision_policy_tool(
    overall_score: Optional[float],
    screening_passed: Optional[bool],
    *,
    evidence_complete: bool,
    unresolved_contradictions: bool,
    hire_threshold: float = 0.75,
    waitlist_threshold: float = 0.50,
) -> DecisionResult:
    """Return a decision and explanation using rules in priority order.

    1. Failed screening means reject, including when later stages were skipped.
    2. Missing screening, score, or evidence, or unresolved contradictions,
       means waitlist.
    3. Otherwise, use the score thresholds (inclusive lower bounds).

    Use None for an unavailable score or screening result. The caller must
    explicitly report whether all required evidence is available and whether
    contradictions remain. This function cannot infer those from a score.

    Scores and thresholds must be numbers from 0 to 1. Invalid values raise
    ValueError. Thresholds are provisional project choices, not assignment
    requirements. Comparisons use the original score without rounding.
    """
    numeric_inputs = {
        "hire_threshold": hire_threshold,
        "waitlist_threshold": waitlist_threshold,
    }
    if overall_score is not None:
        numeric_inputs["overall_score"] = overall_score

    for name, value in numeric_inputs.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number between 0 and 1.")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number between 0 and 1.")

    if waitlist_threshold >= hire_threshold:
        raise ValueError("waitlist_threshold must be lower than hire_threshold.")
    if screening_passed is not None and not isinstance(screening_passed, bool):
        raise ValueError("screening_passed must be True, False, or None.")
    for name, flag in {
        "evidence_complete": evidence_complete,
        "unresolved_contradictions": unresolved_contradictions,
    }.items():
        if not isinstance(flag, bool):
            raise ValueError(f"{name} must be True or False.")

    if screening_passed is False:
        return {
            "final_decision": "reject",
            "decision_reasoning": "Screening failed; the candidate is rejected regardless of the overall score.",
        }

    review_reasons = []
    if screening_passed is None:
        review_reasons.append("the screening result is missing")
    if overall_score is None:
        review_reasons.append("the overall score is unavailable")
    if not evidence_complete:
        review_reasons.append("required evidence is incomplete")
    if unresolved_contradictions:
        review_reasons.append("unresolved contradictions remain")

    if review_reasons:
        return {
            "final_decision": "waitlist",
            "decision_reasoning": "Further review is required: " + "; ".join(review_reasons) + ".",
        }

    if overall_score >= hire_threshold:
        return {
            "final_decision": "hire",
            "decision_reasoning": (
                f"The overall score meets the hire threshold of {hire_threshold}; "
                "screening passed, required evidence is complete, and no unresolved contradictions remain."
            ),
        }
    if overall_score >= waitlist_threshold:
        return {
            "final_decision": "waitlist",
            "decision_reasoning": (
                f"The overall score meets the waitlist threshold of {waitlist_threshold} "
                f"but is below the hire threshold of {hire_threshold}."
            ),
        }
    return {
        "final_decision": "reject",
        "decision_reasoning": (
            f"The overall score is below the waitlist threshold of {waitlist_threshold}."
        ),
    }