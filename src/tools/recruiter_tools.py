"""Person D's interview evaluation, aggregation, and policy tools."""

from langchain_core.tools import tool
from pydantic import Field

from src.llm import structured_call
from src.state import Record, Score, Text
from src.tools.decision_policy import decision_policy_tool
from src.tools.score_aggregator import score_aggregator_tool


class AnswerEvaluation(Record):
    answer_index: int = Field(ge=0)
    score: Score
    reasoning: Text


class InterviewEvaluation(Record):
    evaluations: list[AnswerEvaluation]


@tool
def interview_score_tool(job_description: str, transcript: list[dict]) -> dict:
    """Grade each indexed interview answer with a bounded score and rationale."""
    result = structured_call(InterviewEvaluation,
        "Evaluate each supplied interview answer against its question and the job requirements. "
        "Return exactly one evaluation per answer_index, preserving its index. Use only "
        "job-relevant answer quality. Weight relevance 30%, technical/behavioral soundness "
        "40%, and concrete explanation 30% to produce one score per answer. Anchors: 0=absent "
        "or entirely wrong, 0.25=weak, 0.5=partial, 0.75=solid with small gaps, 1=complete and "
        "well supported. A candid admission of not knowing is not proof of technical mastery. "
        "Do not infer credentials or award points for identity, writing polish, or confidence "
        "alone. Grade at the level required by the job; do not penalize a student for "
        "lacking professional experience when the job does not require it. Provide a "
        "concise rationale for each score. CV truth is assessed separately.",
        {"job_description": job_description, "transcript": transcript})
    expected = {item["answer_index"] for item in transcript}
    actual = [item.answer_index for item in result.evaluations]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Interview evaluator must return every requested answer index exactly once.")
    return result.model_dump()


# Keep the user's deterministic implementations unchanged, exposing LangChain tools too.
aggregate_scores = tool(score_aggregator_tool)
apply_decision_policy = tool(decision_policy_tool)
