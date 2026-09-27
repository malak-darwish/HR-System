"""Person D's score aggregator. Run this file for a standalone example."""

from math import isclose
from statistics import mean


def _validate_score(value: float, name: str) -> None:
    """Reject missing, nonnumeric, nonfinite, and out-of-range values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number between 0 and 1.")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1.")


def score_aggregator_tool(
    match_score: float,
    interview_scores: list[float],
    claim_confidences: list[float],
    *,
    match_weight: float = 0.30,
    interview_weight: float = 0.40,
    verification_weight: float = 0.30,
) -> float:
    """Combine screening, interview, and verification scores into one score.

    All input scores must be between 0 and 1. Each interview score has equal
    weight within the interview average; each claim has equal weight within
    the verification average. A higher claim confidence must mean stronger
    evidence supporting that claim, rather than confidence in a contradiction.

    The default weights are provisional and must sum to 1 when overridden.
    Missing scores raise ValueError; they are never silently treated as zero.
    The returned score is not rounded and does not itself determine a decision.
    """
    _validate_score(match_score, "match_score")

    if not interview_scores:
        raise ValueError("interview_scores must contain at least one score.")
    if not claim_confidences:
        raise ValueError("claim_confidences must contain at least one score.")

    for index, score in enumerate(interview_scores):
        _validate_score(score, f"interview_scores[{index}]")
    for index, confidence in enumerate(claim_confidences):
        _validate_score(confidence, f"claim_confidences[{index}]")

    weights = {
        "match_weight": match_weight,
        "interview_weight": interview_weight,
        "verification_weight": verification_weight,
    }
    for name, weight in weights.items():
        _validate_score(weight, name)
    if not isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The three weights must sum to 1.")

    return (
        match_weight * match_score
        + interview_weight * mean(interview_scores)
        + verification_weight * mean(claim_confidences)
    )


if __name__ == "__main__":
    # Sample data stands in for the other agents' future outputs.
    sample_match_score = 0.80
    sample_interview_scores = [0.70, 0.90]
    sample_claim_confidences = [0.85, 0.95]

    overall_score = score_aggregator_tool(
        match_score=sample_match_score,
        interview_scores=sample_interview_scores,
        claim_confidences=sample_claim_confidences,
    )

    print(f"Screening score:          {sample_match_score:.2f}")
    print(f"Average interview score: {mean(sample_interview_scores):.2f}")
    print(f"Average claim confidence:{mean(sample_claim_confidences): .2f}")
    print("Weights: screening 30%, interview 40%, verification 30%")
    print(f"Overall score:           {overall_score:.3f} ({overall_score:.1%})")
