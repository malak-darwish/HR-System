"""Regression coverage for repository links omitted by CV extraction."""

import unittest
from unittest.mock import patch

from src.agents.screener import screener_node
from src.agents.verification import ClaimVerification, VerificationResult, verifier_node
from src.github_refs import extract_github_repositories, repository_urls
from src.state import Claim, HRState, ParsedCV
from tests.test_workflow import Scenario


URL = "https://github.com/malak-darwish/HR-System"
KEY = "repo:malak-darwish/HR-System"


class LinkParsingTests(unittest.TestCase):
    def test_multiple_links_deduplicate_case_suffix_and_subpaths(self):
        self.assertEqual(repository_urls(
            f"Reference: ({URL}). {URL.lower()}.git {URL}/tree/main",
            "[Other](https://www.github.com/demo/my.project.git?tab=readme)",
            "github.com/demo/my.project/blob/main/app.py",
        ), [URL, "https://github.com/demo/my.project"])

    def test_rejects_profiles_lookalikes_and_invalid_repository_paths(self):
        for text in (
            "https://github.com/demo", "https://evilgithub.com/demo/project",
            "https://github.com.evil.test/demo/project", "https://evil.test/github.com/demo/project",
            "https://github.com@evil.test/demo/project", "https://evil@github.com/demo/project",
            "https://github.com/demo/../private", "https://github.com/demo/bad%20repo",
            "https://github.com:999/demo/project",
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_github_repositories(text), [])


class LinkWorkflowTests(unittest.TestCase):
    def state(self, **changes):
        data = dict(cv_text=f"Third-party reference: {URL}. No ownership claimed.",
                    job_description="Python developer", parsed_cv=ParsedCV(),
                    claims=[Claim(claim_id="C1", text="Uses Python", category="skill")],
                    questions=["Explain Python testing"], answers=["I use pytest."])
        data.update(changes)
        return HRState(**data)

    def test_screener_keeps_raw_link_when_model_omits_it(self):
        with Scenario().patches():
            result = screener_node(self.state())
        self.assertEqual(result["github_repository_urls"], [URL])
        self.assertIsNone(result["parsed_cv"]["github_username"])
        self.assertTrue(all("github" not in c["text"] for c in result["claims"]))

    def test_saved_state_without_link_field_fetches_reference_without_assigning_ownership(self):
        scenario = Scenario()
        with scenario.patches(), patch(
            "src.agents.verification.github_verify_tool", **{"invoke.return_value": {"status": "found"}},
        ) as repo, patch("src.agents.verification.github_profile_scan_tool") as profile:
            result = verifier_node(self.state())
        repo.invoke.assert_called_once_with({"username": "malak-darwish", "repo": "HR-System"})
        profile.invoke.assert_not_called()
        self.assertEqual(result["github_evidence"][KEY]["status"], "found")
        self.assertEqual(result["github_repository_urls"], [URL])
        payload = next(p for name, p in scenario.calls if name == "VerificationResult")
        self.assertIn(KEY, payload["referenced_repositories"])
        self.assertNotIn(KEY, payload["claims"][0]["allowed_evidence_refs"])
        self.assertEqual(result["claims"][0]["source"], "interview")
        self.assertEqual(result["claims"][0]["confidence"], 0.8)

    def test_links_in_answers_projects_and_claims_are_collected(self):
        for changes in (
            {"answers": [f"Reference: {URL}"]},
            {"parsed_cv": ParsedCV(projects=[URL])},
            {"claims": [Claim(claim_id="C1", text=URL, category="project")]},
        ):
            with self.subTest(changes=changes), Scenario().patches(), patch(
                "src.agents.verification.github_verify_tool", **{"invoke.return_value": {"status": "found"}},
            ) as repo:
                result = verifier_node(self.state(cv_text="No link here", **changes))
                repo.invoke.assert_called_once()
                self.assertIn(KEY, result["github_evidence"])

    def test_multiple_claim_links_are_available_without_duplicate_fetches(self):
        scenario = Scenario()
        claim = Claim(claim_id="C1", text=f"Compare {URL} and https://github.com/demo/other", category="project")
        with scenario.patches(), patch(
            "src.agents.verification.github_verify_tool", **{"invoke.return_value": {"status": "found"}},
        ) as repo:
            verifier_node(self.state(claims=[claim]))
        self.assertEqual(repo.invoke.call_count, 2)
        payload = next(p for name, p in scenario.calls if name == "VerificationResult")
        self.assertEqual(set(payload["claims"][0]["allowed_evidence_refs"]),
                         {"interview:1", KEY, "repo:demo/other"})

    def test_failed_and_successful_lookups_are_cached_across_followups(self):
        for status in ("found", "unavailable", "not_found"):
            with self.subTest(status=status), Scenario().patches(), patch(
                "src.agents.verification.github_verify_tool", **{"invoke.return_value": {"status": status}},
            ) as repo:
                state = self.state()
                first = verifier_node(state)
                state = HRState.model_validate({**state.model_dump(), **first,
                                                "answers": [URL.lower() + ".git"]})
                second = verifier_node(state)
                repo.invoke.assert_called_once()
                self.assertEqual(list(second["github_evidence"]), [KEY])

    def test_failed_reference_cannot_be_cited(self):
        scenario = Scenario()
        with scenario.patches(), patch(
            "src.agents.verification.github_verify_tool", **{"invoke.return_value": {"status": "unavailable"}},
        ):
            verifier_node(self.state(claims=[Claim(claim_id="C1", text=URL, category="project")]))
        payload = next(p for name, p in scenario.calls if name == "VerificationResult")
        self.assertNotIn(KEY, payload["claims"][0]["allowed_evidence_refs"])

    def test_unrelated_raw_reference_cannot_inflate_claim_support(self):
        assessment = VerificationResult(claims=[ClaimVerification(
            claim_id="C1", verified=True, confidence=0.95, source="github",
            evidence="Unrelated repository", evidence_refs=[KEY],
        )], verification_notes="Unsupported citation")
        with patch("src.agents.verification.structured_call", return_value=assessment), patch(
            "src.agents.verification.github_verify_tool", **{"invoke.return_value": {"status": "found"}},
        ), self.assertRaisesRegex(ValueError, "unavailable evidence"):
            verifier_node(self.state())

    def test_reference_is_recorded_even_without_extracted_claims(self):
        with patch("src.agents.verification.structured_call") as model, patch(
            "src.agents.verification.github_verify_tool", **{"invoke.return_value": {"status": "found"}},
        ):
            result = verifier_node(self.state(claims=[]))
        model.assert_not_called()
        self.assertIn(KEY, result["github_evidence"])
        self.assertEqual(result["claims"], [])


if __name__ == "__main__":
    unittest.main()
