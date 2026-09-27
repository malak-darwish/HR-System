"""Required versus optional evidence: protect both eligible and blocked outcomes."""

from unittest.mock import patch
import unittest

from src.agents.recruiter import recruiter_node
from src.state import Claim, HRState
from src.tools.requirement_tools import assess_job_requirements


def skill(**changes):
    return Claim(**{**dict(claim_id="C1", text="Knows Python testing", category="skill",
                           verified=True, confidence=0.8, source="interview",
                           evidence="Explained assertions and expected exceptions.", evidence_refs=["interview:1"]),
                    **changes})


def state(**changes):
    return HRState(**{**dict(job_description="Required: Python. Degree optional.",
                            match_score=0.8, screening_passed=True, claims=[skill()],
                            questions=["How do you test?"], answers=["Use pytest assertions and raises."],
                            interview_scores=[1.0], interview_score_reasoning=["Complete answer."],
                            interview_complete=True, verification_completed=True,
                            consistency_flags={"C1": False}), **changes})


class RequirementPolicyTests(unittest.TestCase):
    def setUp(self):
        self.requirements = [{"requirement": "Python", "job_description_quote": "Python",
                              "evidence_kind": "capability"}]
        self.matches = [{"requirement_id": "R1", "claim_ids": ["C1"],
                         "evidence_sufficient": True, "reasoning": "Python evidence supplied."}]
        self.requests = []

    def response(self, schema, instruction, payload):
        self.requests.append((schema.__name__, payload))
        if schema.__name__ == "MandatoryRequirements":
            return schema(requirements=self.requirements)
        if schema.__name__ == "RequirementMatches":
            return schema(matches=self.matches)
        if schema.__name__ == "RecruiterExplanation":
            return schema(final_decision=payload["policy"]["final_decision"],
                          decision_reasoning="Case details based on assessed requirements.")
        raise AssertionError(schema.__name__)

    def run_recruiter(self, incoming):
        with patch("src.tools.requirement_tools.structured_call", side_effect=self.response), patch(
            "src.agents.recruiter.structured_call", side_effect=self.response,
        ):
            update = recruiter_node(incoming)
        HRState.model_validate({**incoming.model_dump(), **update})
        return update

    def test_optional_unknown_degree_does_not_block_hire_or_become_verified(self):
        incoming = state(claims=[skill(), Claim(claim_id="C2", text="Degree", category="education")],
                         consistency_flags={"C1": False, "C2": False}, follow_up_needed=True)
        before = incoming.model_dump()
        result = self.run_recruiter(incoming)
        self.assertEqual(result["final_decision"], "hire")
        self.assertAlmostEqual(result["overall_score"], 0.88)
        self.assertEqual(result["scored_claim_ids"], ["C1"])
        self.assertEqual(result["excluded_claim_ids"], ["C2"])
        self.assertEqual(result["requirement_coverage"], 1.0)
        self.assertTrue(result["required_evidence_complete"])
        self.assertTrue(any("still unverified: C2" in x for x in result["verification_limitations"]))
        self.assertNotIn("claims", result)
        self.assertEqual(incoming.model_dump(), before)

    def test_optional_unknown_employment_does_not_block_hire(self):
        incoming = state(claims=[skill(), Claim(claim_id="C2", text="Internship", category="experience")],
                         consistency_flags={"C1": False, "C2": False}, follow_up_needed=True)
        self.assertEqual(self.run_recruiter(incoming)["final_decision"], "hire")

    def test_required_degree_still_blocks_even_if_model_says_sufficient(self):
        self.requirements.append({"requirement": "Degree", "job_description_quote": "Degree",
                                  "evidence_kind": "documentary"})
        self.matches.append({"requirement_id": "R2", "claim_ids": ["C2"],
                             "evidence_sufficient": True, "reasoning": "Candidate says they have it."})
        incoming = state(job_description="Required: Python and Degree.",
                         claims=[skill(), skill(claim_id="C2", text="Degree", category="education")],
                         consistency_flags={"C1": False, "C2": False})
        result = self.run_recruiter(incoming)
        self.assertEqual(result["final_decision"], "waitlist")
        self.assertEqual(result["requirement_coverage"], 0.5)
        self.assertFalse(result["required_evidence_complete"])
        self.assertFalse(result["requirement_assessments"][1]["supported"])

    def test_missing_mandatory_skill_with_no_cv_claim_blocks_high_score(self):
        self.requirements.append({"requirement": "SQL", "job_description_quote": "SQL",
                                  "evidence_kind": "capability"})
        self.matches.append({"requirement_id": "R2", "claim_ids": [],
                             "evidence_sufficient": True, "reasoning": "No matching claim."})
        result = self.run_recruiter(state(job_description="Required: Python and SQL."))
        self.assertEqual(result["final_decision"], "waitlist")
        self.assertAlmostEqual(result["overall_score"], 0.88)
        self.assertEqual(result["requirement_coverage"], 0.5)

    def test_unknown_required_claim_cannot_be_promoted_by_mapper(self):
        result = self.run_recruiter(state(claims=[skill(verified=None, confidence=None)]))
        self.assertEqual(result["final_decision"], "waitlist")
        self.assertIsNone(result["overall_score"])
        self.assertEqual(result["interview_average"], 1.0)
        self.assertEqual(result["requirement_coverage"], 0.0)

    def test_optional_contradiction_still_blocks_hire(self):
        incoming = state(claims=[skill(), skill(claim_id="C2", text="Internship", category="experience",
                                               verified=False, confidence=0.0)],
                         consistency_flags={"C1": False, "C2": True}, follow_up_needed=True)
        result = self.run_recruiter(incoming)
        self.assertEqual(result["final_decision"], "waitlist")
        self.assertAlmostEqual(result["overall_score"], 0.88)
        self.assertIn("unresolved contradictions", result["decision_reasoning"])

    def test_false_verdict_blocks_even_if_flag_incorrectly_false(self):
        incoming = state(claims=[skill(), skill(claim_id="C2", verified=False, confidence=0.0)],
                         consistency_flags={"C1": False, "C2": False})
        self.assertEqual(self.run_recruiter(incoming)["final_decision"], "waitlist")

    def test_partially_supported_requirement_still_blocks(self):
        self.matches[0]["evidence_sufficient"] = False
        result = self.run_recruiter(state())
        self.assertEqual(result["final_decision"], "waitlist")
        self.assertEqual(result["requirement_coverage"], 0.0)

    def test_low_support_for_mandatory_claim_blocks(self):
        result = self.run_recruiter(state(claims=[skill(confidence=0.4)]))
        self.assertFalse(result["required_evidence_complete"])
        self.assertEqual(result["final_decision"], "waitlist")

    def test_unavailable_citations_and_missing_evidence_block(self):
        for changes in ({"evidence_refs": []}, {"evidence_refs": ["interview:99"]},
                        {"evidence_refs": ["repo:other/project"]}, {"source": None}, {"evidence": None}):
            with self.subTest(changes=changes):
                result = self.run_recruiter(state(claims=[skill(**changes)]))
                self.assertEqual(result["final_decision"], "waitlist")
                self.assertIsNone(result["overall_score"])

    def test_incomplete_pipeline_or_flags_block_even_when_requirements_supported(self):
        for changes in ({"verification_completed": False}, {"interview_complete": False},
                        {"consistency_flags": {}}, {"follow_up_needed": True}):
            with self.subTest(changes=changes):
                self.assertEqual(self.run_recruiter(state(**changes))["final_decision"], "waitlist")

    def test_no_requirements_or_empty_job_description_cannot_authorize_hire(self):
        self.requirements = []
        for description in ("Python", ""):
            with self.subTest(description=description):
                result = self.run_recruiter(state(job_description=description))
                self.assertEqual(result["final_decision"], "waitlist")
                self.assertIsNone(result["requirement_coverage"])

    def test_job_requirements_are_extracted_without_candidate_data(self):
        self.run_recruiter(state())
        payload = next(p for name, p in self.requests if name == "MandatoryRequirements")
        self.assertEqual(set(payload), {"job_description"})

    def test_invented_job_description_quote_is_rejected(self):
        self.requirements[0]["job_description_quote"] = "PhD mandatory"
        with self.assertRaisesRegex(ValueError, "quote the supplied job description"):
            self.run_recruiter(state())

    def test_missing_duplicate_and_invented_requirement_ids_are_rejected(self):
        original = self.matches[0]
        for matches in ([], [original, original], [{**original, "requirement_id": "R99"}]):
            with self.subTest(matches=matches), self.assertRaisesRegex(ValueError, "every mandatory requirement"):
                self.matches = matches
                self.run_recruiter(state())

    def test_unknown_claim_id_is_rejected(self):
        self.matches[0]["claim_ids"] = ["C99"]
        with self.assertRaisesRegex(ValueError, "unknown claim ID"):
            self.run_recruiter(state())

    def test_known_optional_claim_does_not_inflate_score(self):
        incoming = state(claims=[skill(), skill(claim_id="C2", text="Optional extra", confidence=1.0)],
                         consistency_flags={"C1": False, "C2": False})
        result = self.run_recruiter(incoming)
        self.assertEqual(result["scored_claim_ids"], ["C1"])
        self.assertAlmostEqual(result["overall_score"], 0.88)

    def test_duplicate_claim_mapping_does_not_double_weight_evidence(self):
        self.requirements.append({**self.requirements[0], "requirement": "Python testing"})
        self.matches.append({**self.matches[0], "requirement_id": "R2", "claim_ids": ["C1", "C1"]})
        result = self.run_recruiter(state())
        self.assertEqual(result["scored_claim_ids"], ["C1"])
        self.assertAlmostEqual(result["overall_score"], 0.88)

    def test_screening_failure_skips_requirement_calls(self):
        result = self.run_recruiter(state(screening_passed=False))
        self.assertEqual(result["final_decision"], "reject")
        self.assertNotIn("MandatoryRequirements", [name for name, _ in self.requests])

    def test_empty_claims_cannot_produce_hire(self):
        result = self.run_recruiter(state(claims=[], consistency_flags={}))
        self.assertEqual(result["final_decision"], "waitlist")
        self.assertFalse(result["required_evidence_complete"])


if __name__ == "__main__":
    unittest.main()
