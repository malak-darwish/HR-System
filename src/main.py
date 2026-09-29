"""Run the full A -> B -> C -> D pipeline through Person D's graph.

From the HR-System root:
    python -m src.run_pipeline                      # simulated candidate (default)
    python -m src.run_pipeline --interactive        # you type the answers
    python -m src.run_pipeline --output out/run.json
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

from graph import build_interview_graph
from llm import ProviderRequestError
from state import HRState


def show(state: HRState) -> None:
    line = "=" * 60
    print(f"\n{line}\nSCREENING\n{line}")
    print(f"Passed: {state.screening_passed} | match_score: {state.match_score}")
    print(state.screening_reasoning)

    print(f"\n{line}\nINTERVIEW  ({len(state.questions)} Q&A, "
          f"{state.follow_up_count}/{state.max_follow_ups} follow-ups, source={state.answer_source})\n{line}")
    for i, (q, a) in enumerate(zip(state.questions, state.answers), 1):
        score = state.interview_scores[i - 1] if i <= len(state.interview_scores) else None
        print(f"[interview:{i}] Q: {q}\n   A: {a}\n   score: {score}")

    print(f"\n{line}\nVERIFICATION\n{line}")
    for c in state.claims:
        flag = " <-- UNRESOLVED" if state.consistency_flags.get(c.claim_id) else ""
        print(f"{c.claim_id} [{c.category}] {c.text}")
        print(f"   verified={c.verified} confidence={c.confidence} source={c.source} "
              f"refs={c.evidence_refs}{flag}")
    print(f"Notes: {state.verification_notes}")

    print(f"\n{line}\nREQUIREMENTS\n{line}")
    for r in state.requirement_assessments:
        print(f"{r.requirement_id} [{r.evidence_kind}] supported={r.supported} "
              f"claims={r.claim_ids}: {r.requirement}")
    print(f"Coverage: {state.requirement_coverage} | evidence complete: {state.required_evidence_complete}")
    for item in state.verification_limitations:
        print(f" - {item}")

    print(f"\n{line}\nDECISION\n{line}")
    print(f"overall_score: {state.overall_score} | interview_average: {state.interview_average}")
    print(f"FINAL DECISION: {state.final_decision}")
    print(state.decision_reasoning)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=Path, default=Path("data/sample_cv.txt"))
    parser.add_argument("--jd", type=Path, default=Path("data/sample_jd.txt"))
    parser.add_argument("--interactive", action="store_true", help="type answers yourself")
    parser.add_argument("--max-follow-ups", type=int, default=2)
    parser.add_argument("--output", type=Path, help="save the final state as JSON")
    args = parser.parse_args()

    try:
        app = build_interview_graph(simulate=not args.interactive)
        result = app.invoke(
            {"cv_text": args.cv.read_text(encoding="utf-8"),
             "job_description": args.jd.read_text(encoding="utf-8"),
             "max_follow_ups": args.max_follow_ups},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        state = HRState.model_validate(result)
    except ProviderRequestError as error:
        print(f"Pipeline stopped (Gemini request failed): {error}", file=sys.stderr)
        return 1

    show(state)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"final_state": state.model_dump(mode="json")},
                                          indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())