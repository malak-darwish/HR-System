"""Check policy boundaries, rule precedence, and the scorer integration."""

import unittest
from math import nextafter

from src.tools.decision_policy import decision_policy_tool
from src.tools.score_aggregator import score_aggregator_tool


class DecisionPolicyTests(unittest.TestCase):
    def evaluate(self, score, screening=True, **overrides):
        options = {"evidence_complete": True, "unresolved_contradictions": False}
        options.update(overrides)
        return decision_policy_tool(score, screening, **options)

    def test_sample_scorer_connects_to_policy(self):
        score = score_aggregator_tool(0.80, [0.70, 0.90], [0.85, 0.95])
        result = self.evaluate(score)
        self.assertAlmostEqual(score, 0.83)
        self.assertEqual(set(result), {"final_decision", "decision_reasoning"})
        self.assertEqual(result["final_decision"], "hire")
        self.assertIn("hire threshold of 0.75", result["decision_reasoning"])

    def test_threshold_boundaries_without_rounding(self):
        cases = [
            (0.0, "reject"),
            (nextafter(0.50, 0.0), "reject"),
            (0.50, "waitlist"),
            (nextafter(0.75, 0.0), "waitlist"),
            (0.75, "hire"),
            (1.0, "hire"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(self.evaluate(score)["final_decision"], expected)

    def test_screening_failure_overrides_high_score_and_review_flags(self):
        for score in (1.0, None):
            with self.subTest(score=score):
                result = self.evaluate(
                    score, screening=False,
                    evidence_complete=False, unresolved_contradictions=True,
                )
                self.assertEqual(result["final_decision"], "reject")
                self.assertIn("Screening failed", result["decision_reasoning"])

    def test_missing_screening_or_score_requires_review(self):
        for score, screening in ((1.0, None), (None, True), (None, None)):
            with self.subTest(score=score, screening=screening):
                result = self.evaluate(score, screening=screening)
                self.assertEqual(result["final_decision"], "waitlist")

    def test_review_flags_override_both_high_and_low_scores(self):
        for score in (0.0, 1.0):
            for flag in ({"evidence_complete": False}, {"unresolved_contradictions": True}):
                with self.subTest(score=score, flag=flag):
                    self.assertEqual(self.evaluate(score, **flag)["final_decision"], "waitlist")

    def test_explanation_includes_all_review_reasons(self):
        result = self.evaluate(
            None, screening=None,
            evidence_complete=False, unresolved_contradictions=True,
        )
        for phrase in ("screening result", "score is unavailable", "evidence is incomplete", "contradictions"):
            self.assertIn(phrase, result["decision_reasoning"])

    def test_custom_thresholds(self):
        options = {"hire_threshold": 0.90, "waitlist_threshold": 0.60}
        for score, expected in ((0.59, "reject"), (0.60, "waitlist"), (0.83, "waitlist"), (0.90, "hire")):
            with self.subTest(score=score):
                self.assertEqual(self.evaluate(score, **options)["final_decision"], expected)

    def test_invalid_scores_raise_instead_of_producing_decisions(self):
        for score in (-0.1, 1.1, float("nan"), float("inf"), "0.8", True):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    self.evaluate(score)

    def test_invalid_thresholds_raise(self):
        cases = [
            {"waitlist_threshold": 0.80},
            {"waitlist_threshold": 0.75},
            {"hire_threshold": 1.1},
            {"waitlist_threshold": -0.1},
            {"hire_threshold": float("nan")},
            {"hire_threshold": float("inf")},
            {"hire_threshold": "0.75"},
            {"waitlist_threshold": True},
            {"hire_threshold": None},
        ]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.evaluate(0.83, **options)

    def test_flags_must_be_actual_booleans(self):
        for screening in (0, 1, "false"):
            with self.subTest(screening=screening):
                with self.assertRaises(ValueError):
                    self.evaluate(0.83, screening=screening)
        for name in ("evidence_complete", "unresolved_contradictions"):
            for flag in (None, 0, 1, "false"):
                with self.subTest(name=name, flag=flag):
                    with self.assertRaises(ValueError):
                        self.evaluate(0.83, **{name: flag})


if __name__ == "__main__":
    unittest.main()
