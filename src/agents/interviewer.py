"""Person B's interview flow and question tools, using the shared Pydantic state."""

from collections.abc import Callable

from pydantic import Field

from src.llm import structured_call
from src.state import HRState, Record, Text
from src.tools.interviewer_tools import get_cv_section_tool, question_bank_tool


class InterviewQuestions(Record):
    questions: list[Text] = Field(min_length=4, max_length=4)


class FollowUpQuestion(Record):
    question: Text


class CandidateAnswer(Record):
    answer: Text


def generate_interview_questions(state: HRState) -> list[str]:
    cv = state.parsed_cv.model_dump() if state.parsed_cv else {}
    sections = {section: get_cv_section_tool.invoke({"section": section, "cv_parsed": cv})
                for section in ("experience", "skills", "projects", "education")}
    templates = {
        category: question_bank_tool.invoke({"category": category, "keyword": keyword})
        for category, keyword in (
            ("technical", sections["skills"][:100]),
            ("behavioral", ""), ("experience", sections["experience"][:100]))
    }
    return structured_call(InterviewQuestions,
        "Generate exactly four personalized interview questions: two technical, one "
        "behavioral, one experience question. Use the CV, job requirements, and template "
        "tools. Ask specific, distinct, job-relevant questions. Adapt the templates into "
        "natural questions; do not paste raw CV sections into placeholder sentences. "
        "Coursework is not an employer. Do not assume professional experience when the CV "
        "does not claim it. Ask technical questions that require explanation, not just "
        "repeating a skill name.",
        {"cv_sections": sections, "job_description": state.job_description,
         "templates": templates}).questions


def generate_follow_up_question(state: HRState) -> str:
    targets = [c.model_dump() for c in state.claims if (
        state.consistency_flags.get(c.claim_id, False) or c.verified is not True
        or c.confidence is None or c.confidence < 0.5)]
    return structured_call(FollowUpQuestion,
        "Ask ONE targeted clarifying question about the most important unresolved "
        "claim. Give the candidate a fair opportunity to explain or supply evidence. "
        "Read previous questions and answers; do not repeat a question already asked. "
        "Unknown evidence is not an accusation. Use the verification notes if no specific "
        "claim is listed.",
        {"unresolved_claims": targets, "verification_notes": state.verification_notes,
         "previous_interview": list(zip(state.questions, state.answers)),
         "job_description": state.job_description}).question


def simulate_candidate_answer(question: str, state: HRState) -> str:
    return structured_call(CandidateAnswer,
        "Simulate a candidate's answer for a clearly labelled classroom demo. Answer "
        "in 3-5 sentences, consistently with the supplied CV and previous answers. "
        "Do not invent public repositories, credentials, or employment.",
        {"question": question, "cv": state.parsed_cv.model_dump() if state.parsed_cv else {},
         "previous_interview": list(zip(state.questions, state.answers))}).answer


def read_candidate_answer(question: str, _state: HRState) -> str:
    print(f"\nInterview question: {question}", flush=True)
    print("Type or paste your answer; multiple paragraphs are allowed.\n"
          "To submit, type /done on a new line and press Enter.\n"
          "Type /clear on its own line to start this answer again, or Ctrl+C to stop.", flush=True)
    lines = []
    while True:
        line = input("> ")
        command = line.strip().casefold()
        if command == "/done":
            answer = "\n".join(lines).strip()
            if answer:
                return answer
            print("Please enter an answer before submitting with /done.", flush=True)
        elif command == "/clear":
            lines.clear()
            print("Answer cleared. Enter your replacement answer, then /done.", flush=True)
        else:
            # Blank lines separate paragraphs; only the explicit command submits.
            # EOF/Ctrl+C propagate, so an incomplete paste is never accepted silently.
            lines.append(line)


def make_interviewer(
    *, simulate: bool = False,
    answer_provider: Callable[[str, HRState], str] | None = None,
) -> Callable[[HRState], dict]:
    if simulate and answer_provider is not None:
        raise ValueError("Choose simulated answers or a candidate answer provider.")
    provider = answer_provider or (simulate_candidate_answer if simulate else read_candidate_answer)

    def interviewer_node(state: HRState) -> dict:
        questions = ([generate_follow_up_question(state)] if state.interview_round > 0
                     else generate_interview_questions(state))
        history = state.model_copy(deep=True)
        for question in questions:
            answer = provider(question, history)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("The candidate answer must be non-empty text.")
            history.questions.append(question)
            history.answers.append(answer.strip())
        return {
            # Replacement lists contain the whole history, including earlier rounds.
            "questions": history.questions, "answers": history.answers,
            "interview_complete": True, "follow_up_needed": False,
            "answer_source": "simulated" if simulate else "candidate",
        }

    return interviewer_node


interviewer_node = make_interviewer()
