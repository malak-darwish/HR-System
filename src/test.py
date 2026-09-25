"""
test_integration.py — Screener → Verification chained end-to-end,
standing in for Person B until the Interviewer node exists.
"""

from dotenv import load_dotenv
load_dotenv()

from state import HRState
from agents.screener import screener_node
from agents.verification import verifier_node

from pathlib import Path

def main():
    cv_text = Path("data/sample_cv.txt").read_text(encoding="utf-8")
    job_description = Path("data/sample_jd.txt").read_text(encoding="utf-8")
    print(f"CV text length: {len(cv_text)} characters")
    print(f"CV text preview: {cv_text[:200]}")
    state = HRState(cv_text=cv_text, job_description=job_description)

    # Person A runs first
    screener_result = screener_node(state)
    state = state.model_copy(update=screener_result)

    print(f"\nscreening_passed = {state.screening_passed}\n")

    if not state.screening_passed:
        print("Candidate rejected at screening — verifier never runs in the real graph.")
        return

    # Stand in for Person B: hand-write answers that respond to whatever
    # claims the Screener actually extracted (check state.claims first,
    # then write plausible answers to those specific claims).
    state = state.model_copy(update={
    "answers": [
        "I completed my BE in Computer Engineering at Lebanese American University.",
        "During my internship at Dar, I worked on AI applications using RAG and custom LLM agents, and also did some full-stack development.",
        "SigmaTutor is an AI tutor I built for Signals and Communication Systems — it uses a custom LLM agent with RAG to pull accurate answers from course material.",
        "For the heartbeat anomaly detection project, I trained a deep learning model on ECG signals to flag abnormal heartbeats.",
    ]
})

    verifier_result = verifier_node(state)
    state = state.model_copy(update=verifier_result)

    print("=== Verifier output ===")
    for claim in state.claims:
        print(f"- {claim.text} (category={claim.category})")
        print(f"    verified={claim.verified}  confidence={claim.confidence}  source={claim.source}")
    print(f"\nfollow_up_needed = {state.follow_up_needed}")
    print(f"verification_notes = {state.verification_notes}")


if __name__ == "__main__":
    main()