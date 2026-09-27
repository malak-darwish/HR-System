"""
Interviewer Agent (Person B)

This agent conducts the HR interview by:
  1. Reading the candidate's CV sections using get_cv_section_tool
  2. Generating personalized questions using question_bank_tool + LLM
  3. Collecting candidate answers
  4. Handling follow-up loops when Verification Agent flags inconsistencies

STATE EVOLUTION (Requirement C):
  - state["questions"] grows each turn as new questions are appended
  - state["answers"] grows each turn as candidate answers are appended
  - On follow-up loops, additional clarifying questions/answers are appended

FOLLOW-UP LOOP LOGIC (Requirement F):
  - If state["follow_up_needed"] is True, the Verification Agent found inconsistencies.
  - This agent reads state["consistency_flags"] to see what was flagged.
  - It generates ONE targeted clarifying question about the inconsistency.
  - After the candidate answers, it sets follow_up_needed = False.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from interviewer_tools import get_cv_section_tool, question_bank_tool, QUESTION_TEMPLATES
from state import HRState

load_dotenv()


def get_llm():
    """Initialize the Google Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def parse_response(response):
    """Safely extract text from LLM response (handles string or list of dicts)."""
    raw = response.content
    if isinstance(raw, list):
        parts = []
        for part in raw:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            else:
                parts.append(str(part))
        raw = " ".join(parts)
    return raw.strip()


def generate_interview_questions(cv_parsed, job_description):
    """
    Generate 3-5 personalized interview questions based on the CV and job description.
    Uses get_cv_section_tool and question_bank_tool, then the LLM to personalize.
    """
    llm = get_llm()

    cv_sections = {}
    for section in ["experience", "skills", "projects", "education"]:
        result = get_cv_section_tool.invoke({
            "section": section,
            "cv_parsed": cv_parsed
        })
        cv_sections[section] = result

    skills_text = cv_sections.get("skills", "")
    experience_text = cv_sections.get("experience", "")

    technical_qs = question_bank_tool.invoke({
        "category": "technical",
        "keyword": skills_text[:100] if skills_text else ""
    })
    behavioral_qs = question_bank_tool.invoke({
        "category": "behavioral",
        "keyword": ""
    })
    experience_qs = question_bank_tool.invoke({
        "category": "experience",
        "keyword": experience_text[:100] if experience_text else ""
    })

    system_prompt = """You are an expert HR interviewer. Your job is to create personalized
interview questions for a candidate based on their CV and the job description.

You have the following CV information:
{cv_info}

Job Description:
{jd}

You have these question templates to work with:
Technical: {tech_qs}
Behavioral: {behav_qs}
Experience: {exp_qs}

Generate exactly 4 personalized interview questions:
- 2 technical questions specific to the candidate's skills and the job requirements
- 1 behavioral question relevant to the role
- 1 experience question about their past work

Make each question specific to THIS candidate. Do NOT ask generic questions.

Return ONLY the questions, one per line, numbered 1-4. No extra text."""

    cv_info = "\n".join(f"{k}: {v}" for k, v in cv_sections.items())

    prompt = system_prompt.format(
        cv_info=cv_info,
        jd=job_description,
        tech_qs=technical_qs,
        behav_qs=behavioral_qs,
        exp_qs=experience_qs,
    )

    response = llm.invoke([
        SystemMessage(content="You are an expert HR interviewer."),
        HumanMessage(content=prompt),
    ])

    raw = parse_response(response)

    questions = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and len(line) > 10:
            for prefix in ["1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "4)", "5)"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            questions.append(line)

    return questions[:5] if len(questions) > 5 else questions


def generate_follow_up_question(consistency_flags, cv_parsed):
    """
    Generate ONE targeted clarifying question when Verification Agent flags inconsistencies.

    This is the FOLLOW-UP LOOP logic:
      - Reads consistency_flags to find what was flagged
      - Generates a specific question to address the inconsistency
      - Does NOT repeat previous questions
    """
    llm = get_llm()

    flagged_claims = [claim for claim, is_inconsistent in consistency_flags.items() if is_inconsistent]

    if not flagged_claims:
        return "Could you tell me more about your recent work experience?"

    claims_text = "\n".join(f"- {claim}" for claim in flagged_claims)

    prompt = f"""You are an HR interviewer conducting a follow-up after inconsistencies were found
between the candidate's CV and their interview answers.

The following claims from the CV were flagged as inconsistent:
{claims_text}

Generate exactly ONE specific, professional clarifying question that:
1. Directly addresses the most important inconsistency
2. Gives the candidate a fair chance to explain
3. Is not accusatory but is direct
4. References the specific claim that was flagged

Return ONLY the question, nothing else."""

    response = llm.invoke([
        SystemMessage(content="You are a professional HR interviewer asking a follow-up question."),
        HumanMessage(content=prompt),
    ])

    return parse_response(response)


def simulate_candidate_answer(question, cv_parsed):
    """
    Simulate a candidate's answer for testing purposes.
    In production, this would be replaced by actual candidate input.
    """
    llm = get_llm()

    cv_summary = str(cv_parsed)[:500]

    prompt = f"""You are a job candidate being interviewed. Answer the following interview question
based on your CV information below. Give a realistic, conversational answer (3-5 sentences).

Your CV information:
{cv_summary}

Interview question:
{question}

Answer naturally as if you are in a real interview. Be specific but conversational."""

    response = llm.invoke([
        SystemMessage(content="You are a job candidate in an interview. Answer naturally and specifically."),
        HumanMessage(content=prompt),
    ])

    return parse_response(response)


def interviewer_node(state):
    """
    LangGraph node for the Interviewer Agent.

    NORMAL FLOW:
      1. Reads cv_parsed and job_description from state
      2. Generates 3-5 personalized questions using tools + LLM
      3. Simulates candidate answers (for testing)
      4. Appends questions and answers to state
      5. Sets interview_complete = True

    FOLLOW-UP FLOW (when Verification flags inconsistencies):
      1. Reads follow_up_needed and consistency_flags from state
      2. Generates ONE clarifying question about the flagged inconsistency
      3. Simulates candidate answer
      4. Appends the follow-up Q&A to state
      5. Sets follow_up_needed = False
    """

    cv_parsed = state.get("cv_parsed", {})
    job_description = state.get("job_description", "")
    follow_up_needed = state.get("follow_up_needed", False)
    consistency_flags = state.get("consistency_flags", {})

    new_questions = []
    new_answers = []

    if follow_up_needed and consistency_flags:
        print("\n[LOOP] [Interviewer] Follow-up needed -- generating clarifying question...")

        follow_up_q = generate_follow_up_question(consistency_flags, cv_parsed)
        print(f"[Q] [Interviewer] Follow-up question: {follow_up_q}")

        follow_up_a = simulate_candidate_answer(follow_up_q, cv_parsed)
        print(f"[A] [Candidate] Answer: {follow_up_a}")

        new_questions.append(follow_up_q)
        new_answers.append(follow_up_a)
        
        return {
            "questions": new_questions,
            "answers": new_answers,
            "follow_up_needed": False,
        }

    print("\n[START] [Interviewer] Starting interview...")

    questions = generate_interview_questions(cv_parsed, job_description)

    if not questions:
        questions = [
            "Tell me about your most significant technical project.",
            "What skills from your background are most relevant to this role?",
            "Describe a challenging situation you faced at work and how you resolved it.",
            "Why are you interested in this position?",
        ]

    for i, question in enumerate(questions, 1):
        print(f"\n[Q] [Interviewer] Question {i}: {question}")

        answer = simulate_candidate_answer(question, cv_parsed)
        print(f"[A] [Candidate] Answer: {answer}")

        new_questions.append(question)
        new_answers.append(answer)

    print(f"\n[DONE] [Interviewer] Interview complete -- {len(new_questions)} questions asked.")

    return {
        "questions": new_questions,
        "answers": new_answers,
        "interview_complete": True,
    }
