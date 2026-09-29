"""
Unit test for the Verification Agent against Person D's shared contract.
Run from the project root (HR-System):   python -m src.test_verifier
"""

from src.state import HRState, ParsedCV, Claim
from src.agents.verification import verifier_node

EXP_ID, EXP = "c1", "3 years as Backend Developer at Acme Corp"
PY_ID, PY = "c2", "Python"
QUESTION = "Walk me through your role at Acme Corp."


def make_state(answer: str, follow_up_count: int = 0) -> HRState:
    # Built fresh each time so D's validator runs (questions/answers must match).
    return HRState(
        parsed_cv=ParsedCV(name="Test Candidate", skills=[PY], experience=[EXP]),
        claims=[
            Claim(claim_id=EXP_ID, text=EXP, category="experience"),
            Claim(claim_id=PY_ID, text=PY, category="skill"),
        ],
        questions=[QUESTION],
        answers=[answer],
        follow_up_count=follow_up_count,
    )


def show(label, out):
    print(f"\n===== {label} =====")
    print("follow_up_needed:", out["follow_up_needed"])
    print("consistency_flags:", out["consistency_flags"])
    for c in out["claims"]:
        print(f"  - [{c.claim_id}] {c.text!r} verified={c.verified} conf={c.confidence} "
              f"source={c.source} refs={c.evidence_refs}")
        print(f"      evidence: {c.evidence}")
    print("notes:", out["verification_notes"])


def check_contract(state: HRState, out: dict):
    # The merged state must pass D's validator (flags keyed by claim_id, etc.)
    merged = HRState.model_validate({**state.model_dump(), **{
        k: v for k, v in out.items() if k != "claims"
    }, "claims": [c.model_dump() for c in out["claims"]]})
    assert set(merged.consistency_flags) == {EXP_ID, PY_ID}, "Need one flag per claim_id"
    assert out["verification_completed"] is True
    for c in out["claims"]:
        assert c.category, "Category lost in merge-back"
        if c.verified is None:
            assert c.confidence is None, "Unknown claims must not carry a confidence"
        if c.verified is True:
            assert c.evidence_refs, "A verified claim must cite evidence"
        assert all(r.startswith(("interview:", "profile:", "repo:")) for r in c.evidence_refs)


# ── Test 1: contradiction ──
s1 = make_state("Honestly I was only at Acme for a 3-month summer internship, doing frontend work in React.")
out1 = verifier_node(s1)
show("TEST 1: contradiction", out1)
check_contract(s1, out1)
by_id = {c.claim_id: c for c in out1["claims"]}
assert out1["consistency_flags"][EXP_ID] is True, "Contradicted claim must be flagged"
assert by_id[EXP_ID].verified is False
assert "interview:1" in by_id[EXP_ID].evidence_refs, "Contradiction should cite the answer"
assert by_id[PY_ID].verified is not False, "Undiscussed Python must not be marked contradicted"
assert out1["follow_up_needed"] is True
print("PASS: contradiction flagged, cited, follow-up triggered")

# ── Test 2: consistent ──
s2 = make_state("At Acme Corp I spent 3 years as a backend developer, building REST APIs in Python and Django.")
out2 = verifier_node(s2)
show("TEST 2: consistent", out2)
check_contract(s2, out2)
assert not any(out2["consistency_flags"].values())
assert out2["follow_up_needed"] is False
print("PASS: consistent answer not flagged")

# ── Test 3: cap reached ──
s3 = make_state("Honestly I was only at Acme for a 3-month summer internship, doing frontend work in React.",
                follow_up_count=HRState().max_follow_ups)
out3 = verifier_node(s3)
show("TEST 3: cap reached", out3)
check_contract(s3, out3)
assert out3["consistency_flags"][EXP_ID] is True, "Flag still recorded for the Recruiter"
assert out3["follow_up_needed"] is False
print("PASS: loop cap respected")

print("\nALL VERIFIER TESTS PASSED")