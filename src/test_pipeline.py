"""
Integration test: Screener -> Interviewer -> Verifier (-> Interviewer loop) -> END
Runs inside a real LangGraph so the operator.add reducers are actually applied.
(Calling nodes one by one by hand does NOT apply reducers.)

Run from src/:
    python test_pipeline.py            # normal run
    python test_pipeline.py --lie      # candidate contradicts CV -> exercises loop + cap
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, START, END

from state import HRState, MAX_FOLLOW_UPS
from agents.screener import screener_node          # adjust if Person A named it differently
from agents.verification import verifier_node
import agents.interviewer as interviewer

FORCE_LIE = "--lie" in sys.argv
LIE_ONCE = "--lie-once" in sys.argv

if FORCE_LIE or LIE_ONCE:
    _original = interviewer.simulate_candidate_answer
    _lied = {"done": False}

    def lying_answer(question, cv_parsed):
        answer = _original(question, cv_parsed)
        if LIE_ONCE and _lied["done"]:
            return answer
        _lied["done"] = True
        return answer + (
            " I should be honest though: I dropped out of university after my first year "
            "and never actually finished my Computer Engineering degree."
        )

    interviewer.simulate_candidate_answer = lying_answer


# ── Mini graph (stand-in for Person D's wiring) ──
def route_after_screener(state: HRState):
    return "interviewer" if state.screening_passed else END


def route_after_verifier(state: HRState):
    return "interviewer" if state.follow_up_needed else END


g = StateGraph(HRState)
g.add_node("screener", screener_node)
g.add_node("interviewer", interviewer.interviewer_node)
g.add_node("verifier", verifier_node)

g.add_edge(START, "screener")
g.add_conditional_edges("screener", route_after_screener)
g.add_edge("interviewer", "verifier")
g.add_conditional_edges("verifier", route_after_verifier)

graph = g.compile()


# ── Run ──
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
with open(os.path.join(data_dir, "sample_cv.txt"), encoding="utf-8") as f:
    cv_text = f.read()
with open(os.path.join(data_dir, "sample_jd.txt"), encoding="utf-8") as f:
    jd_text = f.read()

assert cv_text.strip(), "sample_cv.txt is empty (this caused the 'Jane Doe' hallucination before)"

final = graph.invoke(HRState(cv_text=cv_text, job_description=jd_text))  # returns a dict

print("\n================ FINAL STATE ================")
print("screening_passed:", final.get("screening_passed"), "| match_score:", final.get("match_score"))

if not final.get("screening_passed"):
    print("Candidate rejected at screening, interview/verification never ran.")
    sys.exit(0)

qs, ans = final["questions"], final["answers"]
fu = final["follow_up_count"]

print(f"questions: {len(qs)} | answers: {len(ans)} | follow_up_count: {fu}")
for i, (q, a) in enumerate(zip(qs, ans), 1):
    print(f"\nQ{i}: {q}\nA{i}: {a[:200]}...")

print("\nconsistency_flags:", final["consistency_flags"])
print("follow_up_needed:", final["follow_up_needed"])
print("verification_notes:", final["verification_notes"])
for c in final["claims"]:
    print(f"  - {c.text!r} [{c.category}] verified={c.verified} conf={c.confidence} source={c.source}")

# ── Checks ──
assert len(qs) == len(ans), "Questions and answers out of sync"
assert final["interview_complete"] is True
assert fu <= MAX_FOLLOW_UPS, "Loop cap exceeded"
assert len(qs) >= 4 + fu, "Reducer not appending: follow-ups overwrote the original questions"
assert all(c.verified is not None for c in final["claims"]), "Some claims never got verified"
assert final["follow_up_needed"] is False, "Graph ended while still requesting a follow-up"

if FORCE_LIE:
    assert fu == MAX_FOLLOW_UPS, "Persistent lie should hit the cap"
if LIE_ONCE:
    assert fu >= 1, "Lie should trigger at least one follow-up"
    print(f"Follow-ups used: {fu} (1 = resolved by clarification, {MAX_FOLLOW_UPS} = never resolved)")

print("\nALL PIPELINE CHECKS PASSED")