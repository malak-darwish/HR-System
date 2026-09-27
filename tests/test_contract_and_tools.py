"""Regression tests for the cross-branch bugs found during review."""

import unittest
import os
from unittest.mock import Mock, patch

import requests

from src.agents.verification import ClaimVerification, _extract_github_ref, verifier_node
from src.state import Claim, HRState, ParsedCV
from src.llm import ProviderRequestError, structured_call
from src.tools.recruiter_tools import InterviewEvaluation, interview_score_tool
from src.tools.verification_tools import (
    consistency_check_tool, github_profile_scan_tool, github_verify_tool,
)
from tests.test_workflow import Scenario


def response(status, data=None):
    return Mock(status_code=status, json=Mock(return_value=data))


class ContractTests(unittest.TestCase):
    def test_provider_error_names_the_step_and_redacts_credentials(self):
        model = Mock()
        model.with_structured_output.return_value.invoke.side_effect = RuntimeError("key=secret-test-key failed")
        with patch("src.llm.get_model", return_value=model), patch.dict(os.environ, {"GEMINI_API_KEY": "secret-test-key"}):
            with self.assertRaises(ProviderRequestError) as error:
                structured_call(InterviewEvaluation, "Evaluate.", {})
        self.assertIn("InterviewEvaluation", str(error.exception))
        self.assertIn("[REDACTED]", str(error.exception))
        self.assertNotIn("secret-test-key", str(error.exception))

    def test_scores_reject_invalid_types_ranges_and_nonfinite_values(self):
        for value in (-0.1, 1.1, 4.5, float("nan"), float("inf"), True, "0.8"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HRState(match_score=value)
            with self.subTest(claim_confidence=value), self.assertRaises(ValueError):
                Claim(claim_id="C1", text="Python", category="skill", confidence=value)

    def test_duplicate_claim_ids_unknown_flags_and_transcript_mismatches_fail(self):
        claim = Claim(claim_id="C1", text="Python", category="skill")
        for values in (
            {"claims": [claim, claim]}, {"claims": [claim], "consistency_flags": {"wrong": True}},
            {"questions": ["Q"]}, {"interview_scores": [0.5]},
            {"max_follow_ups": 0, "follow_up_count": 1}, {"cv_parsed": {}},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                HRState(**values)

    def test_unknown_and_contradicted_verdicts_have_distinct_confidence_rules(self):
        for verdict, confidence in ((None, 0.5), (False, 0.95), (True, None)):
            with self.subTest(verdict=verdict), self.assertRaises(ValueError):
                ClaimVerification(claim_id="C1", verified=verdict, confidence=confidence,
                                  source="interview", evidence="Evidence.")

    def test_missing_or_duplicate_interview_indices_fail(self):
        transcript = [{"answer_index": 0, "question": "Q", "answer": "A"}]
        for evaluations in ([], [
            {"answer_index": 0, "score": 0.8, "reasoning": "Reason."},
            {"answer_index": 0, "score": 0.8, "reasoning": "Reason."},
        ]):
            with patch("src.tools.recruiter_tools.structured_call",
                       return_value=InterviewEvaluation(evaluations=evaluations)):
                with self.assertRaisesRegex(ValueError, "every requested answer index"):
                    interview_score_tool.invoke({"job_description": "JD", "transcript": transcript})


class GitHubToolsTests(unittest.TestCase):
    def test_no_language_argument_means_no_comparison(self):
        with patch("src.tools.verification_tools.requests.get", side_effect=[
            response(200, {"language": "Python"}), response(200, [{}]), response(404),
        ]):
            result = github_verify_tool.invoke({"username": "demo", "repo": "demo.repo"})
        self.assertEqual(result["status"], "found")
        self.assertIsNone(result["language_matches_claim"])
        self.assertTrue(result["has_commits"])

    def test_rate_limits_server_errors_and_missing_public_access_are_unknown(self):
        for code in (403, 429, 500, 404):
            with self.subTest(code=code), patch(
                "src.tools.verification_tools.requests.get", return_value=response(code),
            ):
                result = github_verify_tool.invoke({"username": "demo", "repo": "project"})
                self.assertIsNone(result["exists"])
                self.assertNotEqual(result["status"], "found")

    def test_timeouts_are_unavailable_instead_of_crashing_the_graph(self):
        with patch("src.tools.verification_tools.requests.get", side_effect=requests.Timeout):
            result = github_profile_scan_tool.invoke({"username": "demo"})
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["exists"])

    def test_failed_repo_listing_is_not_an_empty_success(self):
        with patch("src.tools.verification_tools.requests.get", side_effect=[
            response(200, {"login": "demo"}), response(403),
        ]):
            result = github_profile_scan_tool.invoke({"username": "demo"})
        self.assertEqual(result["status"], "unavailable")
        self.assertTrue(result["exists"])

    def test_dotted_repo_name_and_git_suffix_are_preserved(self):
        self.assertEqual(_extract_github_ref("See https://github.com/demo/my.project.git"),
                         ("demo", "my.project"))

    def test_invalid_path_never_reaches_http(self):
        with patch("src.tools.verification_tools.requests.get") as get:
            result = github_verify_tool.invoke({"username": "demo", "repo": "../private"})
        self.assertEqual(result["status"], "unavailable")
        get.assert_not_called()

    def test_lexical_similarity_never_sets_the_contradiction_flag(self):
        claim = "I built the Python service"
        answer = "I did not build the Python service"
        self.assertGreater(consistency_check_tool.invoke({
            "cv_claim": claim, "interview_answer": answer,
        }), 0.7)
        state = HRState(
            claims=[Claim(claim_id="C1", text=claim, category="project")],
            questions=["Did you build it?"], answers=[answer],
        )
        with Scenario(verdicts=(False,)).patches():
            result = verifier_node(state)
        self.assertTrue(result["consistency_flags"]["C1"])

    def test_github_evidence_is_cached_across_follow_up_rounds(self):
        state = HRState(
            parsed_cv=ParsedCV(github_username="demo"),
            claims=[Claim(claim_id="C1", text="Uses Python", category="skill")],
            questions=["Question?"], answers=["Answer."],
        )
        with Scenario().patches(), patch(
            "src.tools.verification_tools.requests.get",
            side_effect=[response(200, {"login": "demo"}), response(200, [])],
        ) as get:
            first = verifier_node(state)
            state = HRState.model_validate({**state.model_dump(), **first})
            verifier_node(state)
        self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
