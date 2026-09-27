"""Checks driven by the manual Rami interview, without relying on local run files."""

import unittest
from unittest.mock import patch

from src.agents.recruiter import _compose_reasoning
from src.agents.verification import ClaimVerification, VerificationResult, _apply_evidence_rules, verifier_node
from src.state import Claim, HRState


def claim(category="skill"):
    return Claim(claim_id="C1", text="Candidate claim", category=category)


def assessment(**overrides):
    return ClaimVerification.model_validate({
        "claim_id": "C1", "verified": True, "confidence": 1.0,
        "source": "interview", "evidence": "Candidate supplied a detailed answer.",
        "evidence_refs": ["interview:1"], **overrides,
    })


class EvidenceLimitTests(unittest.TestCase):
    def test_interview_support_cannot_be_one_hundred_percent(self):
        original = assessment()
        result, _ = _apply_evidence_rules(claim(), original, {"interview:1"})
        self.assertTrue(result["verified"])
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["source"], "interview")
        self.assertIn("no independent verification", result["evidence"])
        self.assertEqual(original.confidence, 1.0)

    def test_lower_interview_support_is_preserved(self):
        result, _ = _apply_evidence_rules(claim(), assessment(confidence=0.6), {"interview:1"})
        self.assertEqual(result["confidence"], 0.6)

    def test_source_name_cannot_claim_github_when_only_an_answer_was_cited(self):
        result, _ = _apply_evidence_rules(
            claim(), assessment(source="Independently verified on GitHub"), {"interview:1"},
        )
        self.assertEqual(result["source"], "interview")
        self.assertEqual(result["confidence"], 0.8)

    def test_education_and_employment_are_not_verified_by_self_report(self):
        for category in ("education", "experience"):
            with self.subTest(category=category):
                result, _ = _apply_evidence_rules(claim(category), assessment(), {"interview:1"})
                self.assertIsNone(result["verified"])
                self.assertIsNone(result["confidence"])
                self.assertIn("independent documentation", result["evidence"])

    def test_public_repository_metadata_does_not_prove_employment(self):
        result, _ = _apply_evidence_rules(claim("experience"), assessment(
            evidence_refs=["repo:demo/project"], source="github"), {"repo:demo/project"})
        self.assertIsNone(result["verified"])
        self.assertIsNone(result["confidence"])

    def test_cited_contradiction_is_preserved(self):
        result, _ = _apply_evidence_rules(claim("experience"), assessment(
            verified=False, confidence=0.1, evidence="The candidate explicitly denied the claim."),
            {"interview:1"})
        self.assertFalse(result["verified"])
        self.assertEqual(result["confidence"], 0.1)

    def test_missing_citations_make_a_verdict_unknown_not_false(self):
        for verdict, score in ((True, 1.0), (False, 0.1)):
            with self.subTest(verdict=verdict):
                result, _ = _apply_evidence_rules(claim(), assessment(
                    verified=verdict, confidence=score, evidence_refs=[]), {"interview:1"})
                self.assertIsNone(result["verified"])
                self.assertIsNone(result["confidence"])
                self.assertEqual(result["source"], "unavailable")

    def test_unknown_evidence_references_are_rejected(self):
        for ref in ("interview:99", "repo:someone/else", "profile:invented"):
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, "unavailable evidence"):
                _apply_evidence_rules(claim(), assessment(evidence_refs=[ref]), {"interview:1"})

    def test_accessible_public_evidence_has_a_separate_ceiling(self):
        result, _ = _apply_evidence_rules(claim("project"), assessment(
            evidence_refs=["repo:demo/project"], source="github"), {"repo:demo/project"})
        self.assertTrue(result["verified"])
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["source"], "github")

    def test_mixed_sources_are_labelled_and_citations_are_deduplicated(self):
        refs = ["interview:1", "repo:demo/project", "interview:1"]
        result, _ = _apply_evidence_rules(claim(), assessment(evidence_refs=refs), set(refs))
        self.assertEqual(result["source"], "interview + github")
        self.assertEqual(result["evidence_refs"], ["interview:1", "repo:demo/project"])
        self.assertEqual(result["confidence"], 0.95)

    def test_node_routes_normalized_history_claim_to_follow_up(self):
        state = HRState(claims=[claim("education")], questions=["Degree?"], answers=["I have it."])
        result = VerificationResult(claims=[assessment()], verification_notes="All claims verified.")
        with patch("src.agents.verification.structured_call", return_value=result):
            update = verifier_node(state)
        self.assertTrue(update["verification_completed"])
        self.assertTrue(update["follow_up_needed"])
        self.assertFalse(update["consistency_flags"]["C1"])
        self.assertIsNone(update["claims"][0]["confidence"])
        self.assertIn("1 unknown", update["verification_notes"])
        self.assertNotIn("All claims verified", update["verification_notes"])


class ExplanationTests(unittest.TestCase):
    def test_repeated_policy_reason_is_removed_and_case_details_remain(self):
        reason = "Further review is required: the overall score is unavailable; required evidence is incomplete."
        text = reason + " " + reason + " Missing evidence for C1 and C10."
        result = _compose_reasoning(reason, text, simulated=False)
        self.assertEqual(result, reason + " Missing evidence for C1 and C10.")

    def test_simulation_prefix_and_policy_are_not_repeated(self):
        reason = "Screening failed; the candidate is rejected regardless of the overall score."
        result = _compose_reasoning(
            reason, "Simulated demo assessment. " + reason + " The CV lacks the required skills.",
            simulated=True,
        )
        self.assertEqual(result.count("Simulated demo assessment."), 1)
        self.assertEqual(result.count(reason), 1)
        self.assertIn("The CV lacks", result)

    def test_non_repeated_explanation_is_preserved(self):
        self.assertEqual(_compose_reasoning("Policy reason.", "Case-specific details.", simulated=False),
                         "Policy reason. Case-specific details.")


if __name__ == "__main__":
    unittest.main()
