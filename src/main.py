"""
Standalone test harness for the Verification Agent, so you can test this
node in isolation before Person D wires the full graph together.

Run from the src/ directory:
    python main.py

Requires GOOGLE_API_KEY in your .env (and GITHUB_TOKEN to avoid GitHub's
unauthenticated rate limit).
"""

from dotenv import load_dotenv
load_dotenv()

from state import HRState, Claim
from agents.verification import verifier_node


def main():
    state = HRState(
        claims=[
            Claim(
                text="Built a project available at github.com/malak-darwish/RAGSystem",
                category="project",
            ),
            Claim(
                text="3 years of experience at Company X",
                category="experience",
            ),
        ],
        answers=[
            "I built a Python-based RAG chatbot for a project.",
            "I worked at Company X for about three years, mostly on backend systems.",
        ],
    )

    result = verifier_node(state)

    print("=== Verifier node output ===")
    for claim in result["claims"]:
        print(f"- {claim.text}  (category={claim.category})")
        print(f"    verified={claim.verified}  confidence={claim.confidence}  source={claim.source}")
    print(f"\nfollow_up_needed = {result['follow_up_needed']}")
    print(f"consistency_flags = {result['consistency_flags']}")
    print(f"verification_notes = {result['verification_notes']}")


if __name__ == "__main__":
    main()