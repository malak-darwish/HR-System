"""Person D: fill missing interview scores, apply fixed policy, explain with Gemini."""

from src.llm import structured_call
from src.state import Decision, HRState, Record, Text
from src.tools.recruiter_tools import (
    aggregate_scores, apply_decision_policy, interview_score_tool,
)


class RecruiterExplanation(Record):
    final_decision: Decision
    decision_reasoning: Text


def _compose_reasoning(policy_reason: str, model_reason: str, *, simulated: bool) -> str:
    prefix = "Simulated demo assessment." if simulated else ""
    detail = " ".join(model_reason.split())
    repeated = [phrase for phrase in (prefix, " ".join(policy_reason.split())) if phrase]
    while True:
        phrase = next((phrase for phrase in repeated if detail.casefold().startswith(phrase.casefold())), None)
        if phrase is None:
            break
        detail = detail[len(phrase):].lstrip()
    return " ".join(part for part in (prefix, policy_reason, detail) if part)


def recruiter_node(state: HRState) -> dict:
    scores = list(state.interview_scores)
    reasons = list(state.interview_score_reasoning)
    if state.screening_passed is True and len(scores) < len(state.answers):
        transcript = [
            {"answer_index": i, "question": state.questions[i], "answer": state.answers[i]}
            for i in range(len(scores), len(state.answers))
        ]
        result = interview_score_tool.invoke({
            "job_description": state.job_description, "transcript": transcript,
        })
        for item in sorted(result["evaluations"], key=lambda row: row["answer_index"]):
            scores.append(item["score"])
            reasons.append(item["reasoning"])

    score_inputs_complete = (
        state.match_score is not None and bool(scores)
        and len(scores) == len(state.answers) and bool(state.claims)
        and all(c.confidence is not None for c in state.claims)
    )
    overall_score = None
    if score_inputs_complete:
        overall_score = aggregate_scores.invoke({
            "match_score": state.match_score, "interview_scores": scores,
            "claim_confidences": [c.confidence for c in state.claims],
        })

    evidence_complete = bool(
        state.verification_completed and state.interview_complete
        and score_inputs_complete and set(state.consistency_flags) == {c.claim_id for c in state.claims}
        and all(c.verified is not None and c.evidence and c.source for c in state.claims)
        and not state.follow_up_needed
    )
    unresolved = any(state.consistency_flags.values()) or any(c.verified is False for c in state.claims)
    policy = apply_decision_policy.invoke({
        "overall_score": overall_score, "screening_passed": state.screening_passed,
        "evidence_complete": evidence_complete, "unresolved_contradictions": unresolved,
    })
    answer_context = (
        "The application generated the answers for a simulated demo. Describe the result as a demo assessment. "
        if state.answer_source == "simulated" else
        "The answers were supplied through candidate input. Do not describe them as application-generated or simulated answers. "
        "A fictional CV alone does not mean the application generated the interview responses. "
    )
    explanation = structured_call(RecruiterExplanation,
        answer_context + "Explain the provided deterministic decision without changing it. Keep the exact "
        "final_decision. Explain missing evidence and unresolved contradictions when present; "
        "do not fill them in. Cite relevant claim IDs and assessment limitations. Scores "
        "use 30% screening + 40% mean interview + 30% mean claim support. The thresholds "
        "are 0.75 hire and 0.50 waitlist, with screening/evidence gates taking priority. "
        "Do not claim external verification when only an interview supports a claim. "
        "Use the exact supplied claim-ID lists for missing or interview-only evidence; "
        "do not turn them into a range containing other claims. "
        "When describing unknown claims, use unknown_claims_by_category to keep education, "
        "employment experience, project details, and skills separate. Never label an unknown "
        "project detail as an education or employment claim. "
        "Give additional case-specific reasoning; the application already prints the "
        "deterministic policy reason, so do not repeat that sentence.",
        {"policy": policy, "overall_score": overall_score,
         "screening_passed": state.screening_passed, "match_score": state.match_score,
         "screening_reasoning": state.screening_reasoning,
         "interview_scores": scores, "interview_score_reasoning": reasons,
         "claims": [c.model_dump() for c in state.claims],
         "unknown_claim_ids": [c.claim_id for c in state.claims if c.verified is None],
         "unknown_claims_by_category": {
             category: [{"claim_id": c.claim_id, "text": c.text} for c in state.claims
                        if c.category == category and c.verified is None]
             for category in ("education", "experience", "project", "skill")
         },
         "interview_only_claim_ids": [c.claim_id for c in state.claims if c.source == "interview"],
         "verification_notes": state.verification_notes,
         "evidence_complete": evidence_complete, "unresolved_contradictions": unresolved,
         "answer_source": state.answer_source,
         "follow_ups_used": state.follow_up_count, "follow_up_limit": state.max_follow_ups})
    if explanation.final_decision != policy["final_decision"]:
        raise ValueError("Gemini explanation disagreed with the deterministic decision policy.")
    # Preserve the exact deterministic explanation as well as the LLM's elaboration.
    return {
        "interview_scores": scores, "interview_score_reasoning": reasons,
        "overall_score": overall_score, "final_decision": policy["final_decision"],
        "decision_reasoning": _compose_reasoning(
            policy["decision_reasoning"], explanation.decision_reasoning,
            simulated=state.answer_source == "simulated",
        ),
    }
