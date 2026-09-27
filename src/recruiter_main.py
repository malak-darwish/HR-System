"""Run the current recruiter independently on the combined A/B/C state."""

import argparse
import json
import sys
from pathlib import Path

from src.agents.recruiter import recruiter_node
from src.llm import ProviderRequestError
from src.state import HRState


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="UTF-8 JSON HRState, or a run export containing final_state")
    parser.add_argument("--output", type=Path, help="Save the recruiter update and merged state")
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        state = HRState.model_validate(payload.get("final_state", payload))
        print("Recruiter: live Gemini. Upstream A/B/C evidence: supplied JSON.", flush=True)
        if state.answer_source == "simulated":
            print("The supplied interview answers are a simulated demo.", flush=True)
        update = recruiter_node(state)
        final = HRState.model_validate({**state.model_dump(), **update}).model_dump(mode="json")
        print(json.dumps(update, indent=2, ensure_ascii=False), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"recruiter_update": update, "final_state": final},
                                             indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Saved recruiter result to {args.output}", flush=True)
        return 0
    except ProviderRequestError as error:
        print(f"Recruiter stopped: {error}\nNo final assessment was completed.", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("Recruiter stopped before completion.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Recruiter stopped ({type(error).__name__}). Check the input against src/state.py, "
              "the .env configuration, and network access. No final assessment was completed.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
