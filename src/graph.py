"""Person D's four-agent LangGraph integration with validated partial updates."""

from collections.abc import Callable
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.interviewer import make_interviewer
from src.agents.recruiter import recruiter_node
from src.agents.screener import screener_node
from src.agents.verification import verifier_node
from src.state import HRState

TeamNode = Callable[[HRState], dict[str, Any]]
CONTROL_FIELDS = {"interview_round", "follow_up_count", "max_follow_ups"}


def needs_follow_up(state: HRState) -> bool:
    return (state.follow_up_needed or any(state.consistency_flags.values())
            or any(c.verified is False for c in state.claims))


def after_screening(state: HRState) -> Literal["interviewer", "recruiter"]:
    return "interviewer" if state.screening_passed is True else "recruiter"


def after_verification(state: HRState) -> Literal["interviewer", "recruiter"]:
    if needs_follow_up(state) and state.follow_up_count < state.max_follow_ups:
        return "interviewer"
    return "recruiter"


def build_interview_graph(
    *, simulate: bool = False, answer_provider=None,
    screener: TeamNode = screener_node, interviewer: TeamNode | None = None,
    verifier: TeamNode = verifier_node, recruiter: TeamNode = recruiter_node,
    checkpointer=None, interrupt_before: list[str] | None = None,
):
    """Compile real A/B/C/D nodes; dependency injection is available for tests."""
    interviewer = interviewer or make_interviewer(simulate=simulate, answer_provider=answer_provider)

    def wrap(node: TeamNode, *, interview: bool = False):
        def run(state: HRState) -> dict:
            update = node(state.model_copy(deep=True))
            if not isinstance(update, dict):
                raise TypeError("Each node must return a dictionary of partial state updates.")
            if CONTROL_FIELDS.intersection(update):
                raise ValueError("Only the graph can change round counters or follow-up limits.")
            if interview:
                update = {
                    **update, "interview_round": state.interview_round + 1,
                    "follow_up_count": state.follow_up_count + int(state.interview_round > 0),
                    "verification_completed": False,
                }
            merged = HRState.model_validate({**state.model_dump(), **update})
            return merged.model_dump(mode="json", include=set(update))
        return run

    builder = StateGraph(HRState)
    builder.add_node("screener", wrap(screener))
    builder.add_node("interviewer", wrap(interviewer, interview=True))
    builder.add_node("verification", wrap(verifier))
    builder.add_node("recruiter", wrap(recruiter))
    builder.add_edge(START, "screener")
    builder.add_conditional_edges("screener", after_screening)
    builder.add_edge("interviewer", "verification")
    builder.add_conditional_edges("verification", after_verification)
    builder.add_edge("recruiter", END)
    return builder.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver(),
        interrupt_before=interrupt_before,
    )
