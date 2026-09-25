"""
screener.py — Screener Agent (Person A)
First gate in the HR pipeline.
- Parses the CV into structured data
- Extracts atomic claims
- Computes match score between CV and JD
- Decides pass/reject before any interview happens
"""

import os
import sys

from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import HRState, ParsedCV, Claim
from tools.screener_tools import parse_cv_tool, match_score_tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage



# ── LLM Setup ────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

# Bind tools to the agent
llm_with_tools = llm.bind_tools([parse_cv_tool, match_score_tool])

# ── Threshold for pass/reject ─────────────────────────────
SCREENING_THRESHOLD = 0.6  # match_score >= 0.6 → pass

# ── System Prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """
You are a professional HR Screener Agent. Your job is to evaluate candidates
before they proceed to an interview.
You will:
1. Parse the candidate's CV using parse_cv_tool
2. Compute the match score between the parsed CV and job description using match_score_tool
3. Based on the match score, decide if the candidate passes screening
You must always call BOTH tools in order:
- First: parse_cv_tool
- Second: match_score_tool (using the parsed CV from step 1)
Be objective and professional.
"""

# ── Screener Node Function ────────────────────────────────
def screener_node(state: HRState) -> dict:
    """
    Screener Agent node for LangGraph.
    
    Reads:  state.cv_text, state.job_description
    Writes: state.parsed_cv, state.claims, 
            state.match_score, state.screening_passed
    """
    print("\n" + "="*50)
    print("🔍 SCREENER AGENT RUNNING...")
    print("="*50)

    # ── Step 1: Parse CV ──────────────────────────────────
    print("\n📄 Step 1: Parsing CV...")
    parse_result = parse_cv_tool.invoke({"cv_text": state.cv_text})

    # Extract parsed_cv and claims from tool result
    parsed_cv_dict = parse_result["parsed_cv"]
    claims_dicts = parse_result["claims"]

    # Convert dicts back to Pydantic objects
    parsed_cv = ParsedCV(**parsed_cv_dict)
    claims = [Claim(**c) for c in claims_dicts]

    print(f"✅ CV Parsed: {parsed_cv.name}")
    print(f"   Skills found: {parsed_cv.skills}")
    print(f"   Claims extracted: {len(claims)}")

    # ── Step 2: Compute Match Score ───────────────────────
    print("\n📊 Step 2: Computing match score...")
    match_score = match_score_tool.invoke({
        "parsed_cv": parsed_cv_dict,
        "job_description": state.job_description
    })

    print(f"✅ Match Score: {match_score:.2f}")

    # ── Step 3: Decide Pass/Reject ────────────────────────
    screening_passed = match_score >= SCREENING_THRESHOLD

    if screening_passed:
        print(f"\n✅ SCREENING PASSED (score {match_score:.2f} >= {SCREENING_THRESHOLD})")
        print("   → Proceeding to Interview")
    else:
        print(f"\n❌ SCREENING FAILED (score {match_score:.2f} < {SCREENING_THRESHOLD})")
        print("   → Candidate rejected")

    print("="*50)

    # ── Step 4: Write to State ────────────────────────────
    return {
        "parsed_cv": parsed_cv,
        "claims": claims,
        "match_score": match_score,
        "screening_passed": screening_passed
    }


# ── Conditional Edge Function ─────────────────────────────
def screening_router(state: HRState) -> str:
    """
    Conditional edge: routes graph based on screening result.
    
    Returns:
        "interviewer" if screening passed
        "end"         if screening failed
    """
    if state.screening_passed:
        return "interviewer"
    else:
        return "end"