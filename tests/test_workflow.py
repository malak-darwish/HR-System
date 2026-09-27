"""Run real nodes/tools/graph with controlled responses at the model boundary."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

from src.agents.interviewer import make_interviewer
from src.agents.recruiter import recruiter_node
from src.agents.verification import verifier_node
from src.graph import build_interview_graph
from src.state import Claim, HRState, ParsedCV

MODEL_MODULES = (
    "src.tools.screener_tools", "src.agents.interviewer", "src.agents.verification",
    "src.tools.recruiter_tools", "src.agents.recruiter",
    "src.tools.requirement_tools",
)


class Scenario:
    def __init__(self, *, match=0.8, verdicts=(True,), missing_claim=False, bad_decision=False):
        self.match = match
        self.verdicts = verdicts
        self.verifications = 0
        self.calls = []
        self.missing_claim = missing_claim
        self.bad_decision = bad_decision

    def __call__(self, schema, _instruction, payload):
        name = schema.__name__
        self.calls.append((name, payload))
        if name == "ParsedCVWithClaims":
            data = {"parsed_cv": {"skills": ["Python", "SQL"]}, "claims": [
                {"text": "Uses Python", "category": "skill"},
                {"text": "Uses SQL", "category": "skill"},
            ]}
        elif name == "MatchScore":
            data = {"score": self.match, "reasoning": "Controlled job match."}
        elif name == "InterviewQuestions":
            data = {"questions": [
                "How do you test a Python function?", "How do you inspect a slow SQL query?",
                "How did you resolve a disagreement?", "Describe a project lesson.",
            ]}
        elif name == "FollowUpQuestion":
            data = {"question": f"Please clarify your Python claim, round {self.verifications}."}
        elif name == "CandidateAnswer":
            data = {"answer": "Synthetic candidate answer for the controlled test."}
        elif name == "VerificationResult":
            verdict = self.verdicts[min(self.verifications, len(self.verdicts) - 1)]
            self.verifications += 1
            data = {
                "claims": [{
                    "claim_id": c["claim_id"], "verified": verdict,
                    "confidence": 0.9 if verdict is True else 0.1 if verdict is False else None,
                    "source": "interview", "evidence": "Controlled test evidence.",
                    "evidence_refs": ["interview:1"] if payload["interview"] else [],
                } for c in payload["claims"]],
                "verification_notes": "Controlled verification for routing tests.",
            }
            if self.missing_claim:
                data["claims"].pop()
        elif name == "InterviewEvaluation":
            # Reverse order deliberately: D must use the explicit answer indices.
            data = {"evaluations": [
                {"answer_index": row["answer_index"], "score": 0.8,
                 "reasoning": f"Assessment of answer {row['answer_index']}."}
                for row in reversed(payload["transcript"])
            ]}
        elif name == "MandatoryRequirements":
            data = {"requirements": [{"requirement": "Relevant technical knowledge",
                                      "job_description_quote": payload["job_description"],
                                      "evidence_kind": "capability"}]}
        elif name == "RequirementMatches":
            data = {"matches": [{"requirement_id": r["requirement_id"],
                                 "claim_ids": [c["claim_id"] for c in payload["claims"]],
                                 "evidence_sufficient": True, "reasoning": "Controlled mapping."}
                                for r in payload["requirements"]]}
        elif name == "RecruiterExplanation":
            data = {
                "final_decision": "reject" if self.bad_decision else payload["policy"]["final_decision"],
                "decision_reasoning": "Explanation based on the supplied assessment.",
            }
        else:
            raise AssertionError(f"Unexpected schema: {name}")
        return schema.model_validate(data)

    def patches(self):
        stack = ExitStack()
        for module in MODEL_MODULES:
            stack.enter_context(patch(module + ".structured_call", side_effect=self))
        return stack


def initial(**kwargs):
    return HRState(cv_text="Synthetic Python and SQL CV.", job_description="Junior Python developer.", **kwargs)


class WorkflowTests(unittest.TestCase):
    def test_optional_unknown_history_can_hire_after_existing_followup_limit(self):
        class OptionalHistoryScenario(Scenario):
            def __call__(self, schema, instruction, payload):
                result = super().__call__(schema, instruction, payload)
                data = result.model_dump()
                if schema.__name__ == "ParsedCVWithClaims":
                    data["claims"].append({"text": "Has a degree", "category": "education"})
                elif schema.__name__ == "VerificationResult":
                    for claim in data["claims"]:
                        if claim["claim_id"] == "C3":
                            claim.update(verified=None, confidence=None, evidence_refs=[])
                elif schema.__name__ == "RequirementMatches":
                    for requirement in data["matches"]:
                        requirement["claim_ids"] = [cid for cid in requirement["claim_ids"] if cid != "C3"]
                return schema.model_validate(data)

        _, _, events, final = self.run_scenario(OptionalHistoryScenario())
        self.assertEqual(final.final_decision, "hire")
        self.assertAlmostEqual(final.overall_score, 0.80)
        self.assertEqual(final.follow_up_count, 2)
        self.assertEqual([next(iter(e)) for e in events], [
            "screener", "interviewer", "verification", "interviewer", "verification",
            "interviewer", "verification", "recruiter",
        ])
        self.assertIsNone(final.claims[2].verified)
        self.assertIsNone(final.claims[2].confidence)
        self.assertEqual(final.scored_claim_ids, ["C1", "C2"])
        self.assertEqual(final.excluded_claim_ids, ["C3"])
        self.assertTrue(final.required_evidence_complete)

    def run_scenario(self, scenario, *, max_follow_ups=2, simulate=False):
        with scenario.patches():
            graph = build_interview_graph(
                simulate=simulate,
                answer_provider=None if simulate else lambda _q, _s: "Candidate test answer.",
            )
            config = {"configurable": {"thread_id": "test"}, "recursion_limit": 30}
            events = list(graph.stream(initial(max_follow_ups=max_follow_ups).model_dump(), config))
            final = HRState.model_validate(graph.get_state(config).values)
        return graph, config, events, final

    def test_all_real_nodes_pass_and_recruiter_grades_answers(self):
        scenario = Scenario()
        graph, config, events, final = self.run_scenario(scenario)
        self.assertEqual([next(iter(e)) for e in events],
                         ["screener", "interviewer", "verification", "recruiter"])
        self.assertEqual(events[1]["interviewer"]["questions"], final.questions)
        self.assertNotIn("interview_scores", events[1]["interviewer"])
        self.assertEqual(final.interview_scores, [0.8] * 4)
        self.assertAlmostEqual(final.overall_score, 0.80)
        self.assertEqual(final.final_decision, "hire")
        self.assertEqual(final.consistency_flags, {"C1": False, "C2": False})
        self.assertEqual(final.follow_up_count, 0)
        self.assertGreaterEqual(len(list(graph.get_state_history(config))), 6)
        self.assertTrue(all(set(e[next(iter(e))]) != set(HRState.model_fields) for e in events))

    def test_screening_rejection_skips_interview_and_verification(self):
        scenario = Scenario(match=0.2)
        _, _, events, final = self.run_scenario(scenario)
        self.assertEqual([next(iter(e)) for e in events], ["screener", "recruiter"])
        self.assertEqual(final.final_decision, "reject")
        self.assertIsNone(final.overall_score)
        self.assertEqual(final.interview_scores, [])
        self.assertNotIn("InterviewEvaluation", [name for name, _ in scenario.calls])

    def test_follow_up_resolves_and_keeps_entire_transcript(self):
        scenario = Scenario(verdicts=(False, True))
        _, _, events, final = self.run_scenario(scenario)
        self.assertEqual([next(iter(e)) for e in events], [
            "screener", "interviewer", "verification", "interviewer", "verification", "recruiter",
        ])
        self.assertEqual(len(final.questions), 5)
        self.assertEqual(len(final.answers), 5)
        self.assertEqual(len(final.interview_scores), 5)
        self.assertEqual(final.interview_round, 2)
        self.assertEqual(final.follow_up_count, 1)
        self.assertEqual(final.final_decision, "hire")
        follow = next(payload for name, payload in scenario.calls if name == "FollowUpQuestion")
        self.assertEqual(len(follow["previous_interview"]), 4)
        self.assertTrue(follow["unresolved_claims"])

    def test_unresolved_stops_after_two_follow_ups_and_waitlists(self):
        _, _, events, final = self.run_scenario(Scenario(verdicts=(False,)))
        self.assertEqual(len(events), 8)
        self.assertEqual(final.follow_up_count, 2)
        self.assertEqual(len(final.questions), 6)
        self.assertEqual(final.final_decision, "waitlist")
        self.assertIn("contradictions", final.decision_reasoning)

    def test_unknown_evidence_does_not_become_a_contradiction_or_zero(self):
        _, _, _, final = self.run_scenario(Scenario(verdicts=(None,)), max_follow_ups=1)
        self.assertEqual(final.follow_up_count, 1)
        self.assertFalse(any(final.consistency_flags.values()))
        self.assertTrue(all(c.verified is None for c in final.claims))
        self.assertIsNone(final.overall_score)
        self.assertEqual(final.final_decision, "waitlist")

    def test_zero_limit_skips_follow_up(self):
        _, _, _, final = self.run_scenario(Scenario(verdicts=(False,)), max_follow_ups=0)
        self.assertEqual(len(final.questions), 4)
        self.assertEqual(final.follow_up_count, 0)
        self.assertEqual(final.final_decision, "waitlist")

    def test_simulation_is_explicit_and_retains_prior_answers(self):
        scenario = Scenario()
        _, _, _, final = self.run_scenario(scenario, simulate=True)
        self.assertEqual(final.answer_source, "simulated")
        self.assertIn("Simulated demo", final.decision_reasoning)
        calls = [p for name, p in scenario.calls if name == "CandidateAnswer"]
        self.assertEqual([len(p["previous_interview"]) for p in calls], [0, 1, 2, 3])

    def test_checkpoint_can_resume_without_repeating_screening(self):
        scenario = Scenario()
        with scenario.patches():
            graph = build_interview_graph(
                answer_provider=lambda _q, _s: "Candidate answer.",
                interrupt_before=["interviewer"],
            )
            config = {"configurable": {"thread_id": "resume"}}
            graph.invoke(initial().model_dump(), config)
            self.assertEqual(graph.get_state(config).next, ("interviewer",))
            graph.invoke(None, config)
            final = HRState.model_validate(graph.get_state(config).values)
        self.assertEqual(final.final_decision, "hire")
        self.assertEqual(sum(name == "ParsedCVWithClaims" for name, _ in scenario.calls), 1)

    def test_threads_have_isolated_checkpoints(self):
        scenario = Scenario()
        with scenario.patches():
            graph = build_interview_graph(answer_provider=lambda _q, _s: "Candidate answer.")
            good = {"configurable": {"thread_id": "good"}}
            bad = {"configurable": {"thread_id": "bad"}}
            graph.invoke(initial().model_dump(), good)
            scenario.match = 0.1
            graph.invoke(initial().model_dump(), bad)
            self.assertEqual(graph.get_state(good).values["final_decision"], "hire")
            self.assertEqual(graph.get_state(bad).values["final_decision"], "reject")

    def test_missing_verdict_is_rejected_instead_of_preserving_stale_evidence(self):
        with self.assertRaisesRegex(ValueError, "every input claim ID"):
            self.run_scenario(Scenario(missing_claim=True))

    def test_model_cannot_override_policy(self):
        with self.assertRaisesRegex(ValueError, "disagreed"):
            self.run_scenario(Scenario(bad_decision=True))

    def test_graph_rejects_agent_changing_loop_limit(self):
        graph = build_interview_graph(screener=lambda state: {"max_follow_ups": 5})
        with self.assertRaisesRegex(ValueError, "Only the graph"):
            graph.invoke(initial().model_dump(), {"configurable": {"thread_id": "invalid"}})

    def test_graph_validates_last_node_output(self):
        scenario = Scenario()
        with scenario.patches():
            graph = build_interview_graph(
                answer_provider=lambda _q, _s: "Candidate answer.",
                recruiter=lambda state: {"overall_score": 4.5},
            )
            with self.assertRaises(ValueError):
                graph.invoke(initial().model_dump(), {"configurable": {"thread_id": "invalid"}})

    def test_input_is_not_mutated_by_nodes(self):
        state = initial()
        before = state.model_dump()
        with Scenario().patches():
            graph = build_interview_graph(answer_provider=lambda _q, _s: "Candidate answer.")
            graph.invoke(state.model_dump(), {"configurable": {"thread_id": "copy"}})
        self.assertEqual(state.model_dump(), before)

    def test_no_claims_cannot_produce_hire(self):
        state = initial(
            screening_passed=True, match_score=0.9, interview_complete=True,
            questions=["Question?"], answers=["Answer."],
        )
        state = HRState.model_validate({**state.model_dump(), **verifier_node(state)})
        with Scenario().patches():
            update = recruiter_node(state)
        self.assertEqual(update["final_decision"], "waitlist")
        self.assertIsNone(update["overall_score"])

    def test_existing_scores_are_kept_and_only_missing_answers_are_graded(self):
        state = initial(
            screening_passed=True, questions=["Q1", "Q2"], answers=["A1", "A2"],
            interview_scores=[0.6], interview_score_reasoning=["Existing rubric."],
        )
        scenario = Scenario()
        with scenario.patches():
            update = recruiter_node(state)
        self.assertEqual(update["interview_scores"], [0.6, 0.8])
        payload = next(p for name, p in scenario.calls if name == "InterviewEvaluation")
        self.assertEqual([r["answer_index"] for r in payload["transcript"]], [1])

    def test_follow_up_does_not_restart_when_flags_are_empty(self):
        state = initial(
            interview_round=1, follow_up_needed=True, questions=["Previous?"], answers=["Answer."],
            claims=[Claim(claim_id="C1", text="Python", category="skill")],
        )
        scenario = Scenario()
        with scenario.patches():
            update = make_interviewer(answer_provider=lambda _q, _s: "Clarification.")(state)
        self.assertEqual(len(update["questions"]), 2)
        self.assertEqual(update["questions"][0], "Previous?")
        self.assertEqual(scenario.calls[0][0], "FollowUpQuestion")


if __name__ == "__main__":
    unittest.main()
