"""Run the integrated workflow: python -m src.main --simulate."""

import argparse
import json
import sys
import uuid
from pathlib import Path

from src.graph import build_interview_graph
from src.llm import ProviderRequestError
from src.state import HRState

DATA = Path(__file__).resolve().parent / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description="Four-agent HR workflow with live Gemini.")
    parser.add_argument("--cv", type=Path, default=DATA / "demo_cv.txt", help="UTF-8 CV text file")
    parser.add_argument("--jd", type=Path, default=DATA / "demo_jd.txt", help="UTF-8 job description file")
    parser.add_argument("--simulate", action="store_true", help="Generate clearly labelled demo answers with Gemini")
    parser.add_argument("--max-follow-ups", type=int, choices=range(6), default=2, metavar="0..5")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--output", type=Path, help="Save the trace and final state as JSON")
    args = parser.parse_args()
    thread_id = args.thread_id or f"hr-{uuid.uuid4().hex[:8]}"
    try:
        state = HRState(
            cv_text=args.cv.read_text(encoding="utf-8-sig"),
            job_description=args.jd.read_text(encoding="utf-8-sig"),
            max_follow_ups=args.max_follow_ups,
            answer_source="simulated" if args.simulate else "candidate",
        )
        graph = build_interview_graph(simulate=args.simulate)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
        print("All four agents: live Gemini. Answers: "
              + ("SIMULATED DEMO." if args.simulate else "candidate input."), flush=True)
        print(f"Thread: {thread_id}", flush=True)
        print("Initial outputs: screening_passed=None, overall_score=None, final_decision=None", flush=True)
        trace = []
        for event in graph.stream(state.model_dump(), config, stream_mode="updates"):
            for node, update in event.items():
                trace.append({"node": node, "update": update})
                print(f"\n{node} partial update:", flush=True)
                print(json.dumps(update, indent=2, ensure_ascii=False), flush=True)
        final = HRState.model_validate(graph.get_state(config).values).model_dump(mode="json")
        checkpoints = sum(1 for _ in graph.get_state_history(config))
        path = [step["node"] for step in trace]
        print("\nExecution path: " + " -> ".join(path), flush=True)
        print(f"Follow-ups used: {final['follow_up_count']}/{final['max_follow_ups']}", flush=True)
        print(f"MemorySaver checkpoints: {checkpoints}", flush=True)
        print(f"Final decision: {final['final_decision']}", flush=True)
        print(f"Reason: {final['decision_reasoning']}", flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "thread_id": thread_id, "path": path, "checkpoints": checkpoints,
                "trace": trace, "final_state": final,
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Saved run to {args.output}", flush=True)
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nInterview stopped before completion.", file=sys.stderr)
        return 130
    except ProviderRequestError as error:
        print(f"Run stopped: {error}\nNo final assessment was completed.", file=sys.stderr)
        return 1
    except Exception as error:
        # Do not dump provider exceptions/requests that may contain credentials or CV text.
        print(f"Run stopped ({type(error).__name__}). Check input files, the .env key/model, "
              "network access, and provider quota. No final assessment was completed.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
