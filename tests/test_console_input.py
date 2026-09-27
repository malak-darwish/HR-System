"""Real stdin parsing: paragraph pastes must never spill into later answers."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents.interviewer import read_candidate_answer
from src.main import main
from src.state import HRState
from tests.test_workflow import Scenario


class ConsoleInputTests(unittest.TestCase):
    def read(self, text):
        with patch("sys.stdin", io.StringIO(text)), redirect_stdout(io.StringIO()):
            return read_candidate_answer("Question?", HRState())

    def test_multiline_answer_keeps_all_paragraphs_and_code_indentation(self):
        text = "I use pytest.\n\nBoundary cases matter.\n\nwith pytest.raises(ValueError):\n    validate(-1)"
        self.assertEqual(self.read(text + "\n/done\n"), text)

    def test_queued_paragraphs_do_not_answer_the_next_question(self):
        text = "First paragraph.\n\nSecond paragraph.\n/done\nSQL joins answer.\n/done\n"
        with patch("sys.stdin", io.StringIO(text)), redirect_stdout(io.StringIO()):
            first = read_candidate_answer("Python?", HRState())
            second = read_candidate_answer("SQL?", HRState())
        self.assertEqual(first, "First paragraph.\n\nSecond paragraph.")
        self.assertEqual(second, "SQL joins answer.")

    def test_empty_submission_stays_on_current_question(self):
        self.assertEqual(self.read("\n  \n/done\nMy actual answer.\n/done\n"), "My actual answer.")

    def test_clear_discards_only_current_draft(self):
        self.assertEqual(self.read("Wrong answer.\nAnother paragraph.\n/clear\nCorrect answer.\n/done\n"),
                         "Correct answer.")

    def test_submit_command_allows_case_and_surrounding_whitespace(self):
        self.assertEqual(self.read("Answer.\n /DONE \n"), "Answer.")

    def test_command_words_inside_sentences_are_normal_answer_text(self):
        text = "The command /done is a delimiter. I also mention /clear here."
        self.assertEqual(self.read(text + "\n/done\n"), text)

    def test_unicode_answer_text_is_preserved(self):
        text = "Python’s boundary cases: zéro, فارغ."
        self.assertEqual(self.read(text + "\n/done\n"), text)

    def test_eof_without_submit_does_not_accept_a_partial_answer(self):
        with self.assertRaises(EOFError):
            self.read("One paragraph.\n\nAnother paragraph.\n")

    def test_keyboard_interrupt_is_not_swallowed(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                read_candidate_answer("Question?", HRState())

    def test_full_cli_saves_four_correctly_aligned_multiline_answers(self):
        answers = [
            "I use pytest.\n\nI cover boundary cases.\n\nI check expected exceptions.",
            "INNER JOIN returns matches.\n\nLEFT JOIN keeps unmatched left rows.",
            "I reproduce the bug.\n\nThen isolate it and add a regression test.",
            "I practised with Python functions.\n\nI used SQLite for users and tasks.",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv, jd, output = root / "cv.txt", root / "jd.txt", root / "run.json"
            cv.write_text("Synthetic Python/SQL candidate", encoding="utf-8")
            jd.write_text("Junior Python developer.", encoding="utf-8")
            stdin = "".join(answer + "\n/done\n" for answer in answers)
            with Scenario().patches(), patch("sys.stdin", io.StringIO(stdin)), patch(
                "sys.argv", ["interview", "--cv", str(cv), "--jd", str(jd), "--output", str(output)],
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["path"], ["screener", "interviewer", "verification", "recruiter"])
            self.assertEqual(result["final_state"]["answers"], answers)
            self.assertEqual(len(result["final_state"]["questions"]), 4)
            self.assertEqual(result["final_state"]["answer_source"], "candidate")

    def test_cli_eof_mid_answer_stops_without_exporting_a_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv, jd, output = root / "cv.txt", root / "jd.txt", root / "run.json"
            cv.write_text("Synthetic Python candidate", encoding="utf-8")
            jd.write_text("Junior Python developer.", encoding="utf-8")
            with Scenario().patches(), patch("sys.stdin", io.StringIO("Partial answer.\n")), patch(
                "sys.argv", ["interview", "--cv", str(cv), "--jd", str(jd), "--output", str(output)],
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 130)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
