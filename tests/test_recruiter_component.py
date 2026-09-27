"""Recruiter handoff checks that import no A/B/C agent or graph implementation."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents.recruiter import recruiter_node
from src.recruiter_main import main
from src.state import Claim, HRState


def completed_state(**changes):
    data = dict(
        job_description="Python trainee", match_score=0.8, screening_passed=True,
        questions=["How do you test?"], answers=["Assert expected outputs and exceptions with pytest."],
        interview_complete=True, verification_completed=True,
        claims=[Claim(claim_id="C1", text="Knows pytest", category="skill", verified=True,
                      confidence=0.8, source="interview", evidence="Explained assertions.",
                      evidence_refs=["interview:1"])], consistency_flags={"C1": False},
    )
    return HRState(**{**data, **changes})


class RecruiterComponentTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def respond(self, schema, instruction, payload):
        self.calls.append((schema.__name__, payload))
        if schema.__name__ == "InterviewEvaluation":
            return schema.model_validate({"evaluations": [
                {"answer_index": item["answer_index"], "score": 1.0, "reasoning": "Complete explanation."}
                for item in reversed(payload["transcript"])
            ]})
        if schema.__name__ == "RecruiterExplanation":
            return schema.model_validate({"final_decision": payload["policy"]["final_decision"],
                                          "decision_reasoning": "Explanation from supplied evidence."})
        raise AssertionError(f"Unexpected model call: {schema.__name__}")

    def models(self):
        stack = ExitStack()
        for module in ("src.agents.recruiter", "src.tools.recruiter_tools"):
            stack.enter_context(patch(module + ".structured_call", side_effect=self.respond))
        return stack

    def test_missing_scores_are_graded_then_aggregated(self):
        state = completed_state()
        before = state.model_dump()
        with self.models():
            update = recruiter_node(state)
        self.assertEqual(update["interview_scores"], [1.0])
        self.assertAlmostEqual(update["overall_score"], 0.88)
        self.assertEqual(update["final_decision"], "hire")
        self.assertEqual(state.model_dump(), before)
        HRState.model_validate({**before, **update})

    def test_existing_scores_are_kept_and_only_missing_answer_is_graded(self):
        state = completed_state(questions=["Q1", "Q2"], answers=["A1", "A2"],
                                interview_scores=[0.6], interview_score_reasoning=["B's rubric."])
        with self.models():
            update = recruiter_node(state)
        self.assertEqual(update["interview_scores"], [0.6, 1.0])
        payload = next(p for name, p in self.calls if name == "InterviewEvaluation")
        self.assertEqual([x["answer_index"] for x in payload["transcript"]], [1])

    def test_screening_failure_skips_grading(self):
        with self.models():
            update = recruiter_node(completed_state(screening_passed=False))
        self.assertEqual(update["final_decision"], "reject")
        self.assertNotIn("InterviewEvaluation", [name for name, _ in self.calls])

    def test_unknown_claims_require_review(self):
        state = completed_state(claims=[Claim(claim_id="C1", text="Degree", category="education")])
        with self.models():
            update = recruiter_node(state)
        self.assertEqual(update["final_decision"], "waitlist")
        self.assertIsNone(update["overall_score"])

    def test_contradiction_overrides_high_numeric_score(self):
        with self.models():
            update = recruiter_node(completed_state(consistency_flags={"C1": True}))
        self.assertGreaterEqual(update["overall_score"], 0.75)
        self.assertEqual(update["final_decision"], "waitlist")

    def test_cli_accepts_state_and_full_run_exports(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped), tempfile.TemporaryDirectory() as directory:
                source, target = Path(directory) / "input.json", Path(directory) / "output.json"
                state = completed_state(answer_source="simulated").model_dump(mode="json")
                source.write_text(json.dumps({"final_state": state} if wrapped else state), encoding="utf-8")
                with self.models(), patch("sys.argv", ["recruiter", "--input", str(source), "--output", str(target)]), patch("builtins.print"):
                    self.assertEqual(main(), 0)
                result = json.loads(target.read_text(encoding="utf-8"))
                self.assertEqual(result["final_state"]["final_decision"], "hire")
                self.assertEqual(result["final_state"]["answers"], state["answers"])

    def test_invalid_input_stops_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.json"
            source.write_text('{"match_score": 9}', encoding="utf-8")
            with self.models(), patch("sys.argv", ["recruiter", "--input", str(source)]), patch("builtins.print"):
                self.assertEqual(main(), 1)
        self.assertEqual(self.calls, [])

    def test_provider_failure_does_not_write_a_success_result(self):
        from src.llm import ProviderRequestError
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "input.json", Path(directory) / "output.json"
            source.write_text(completed_state().model_dump_json(), encoding="utf-8")
            with patch("src.recruiter_main.recruiter_node", side_effect=ProviderRequestError("Timed out")), patch(
                "sys.argv", ["recruiter", "--input", str(source), "--output", str(target)],
            ), patch("builtins.print"):
                self.assertEqual(main(), 1)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
