from state import Claim, HRState, ParsedCV
from tools.screener_tools import match_score_tool, parse_cv_tool

# threshold for pass/reject 
SCREENING_THRESHOLD = 0.6  # match_score >= 0.6 → pass

CATEGORIES = {"experience", "skill", "education", "project"}

def screener_node(state: HRState) -> dict:
    if not state.cv_text.strip():
        raise ValueError("Screener: cv_text is empty — check the CV file path.")

    print("\n" + "=" * 50)
    print("[SCREENER] AGENT RUNNING...")
    print("=" * 50)

    # parse cv 
    print("\n[Step 1] Parsing CV...")
    parse_result = parse_cv_tool.invoke({"cv_text": state.cv_text})

    parsed_cv = ParsedCV.model_validate(parse_result["parsed_cv"])

    claims = []
    for c in parse_result.get("claims", []):
        category = str(c.get("category", "")).strip().lower().rstrip("s")  
        if category not in CATEGORIES or not str(c.get("text", "")).strip():
            continue  
        claims.append(Claim.model_validate({
            **c, "claim_id": f"C{len(claims) + 1}", "category": category,
        }))

    print(f"[OK] CV Parsed: {parsed_cv.name}")
    print(f"     Skills found: {parsed_cv.skills}")
    print(f"     GitHub: {parsed_cv.github_username}")
    print(f"     Claims extracted: {len(claims)}")

    # compute match score
    print("\n[Step 2] Computing match score...")
    match = match_score_tool.invoke({
        "parsed_cv": parse_result["parsed_cv"],
        "job_description": state.job_description,
    })
    match_score = float(match["score"])
    if 1 < match_score <= 100:
        match_score /= 100         
    match_score = max(0.0, min(1.0, match_score))
    print(f"[OK] Match Score: {match_score:.2f}")

    # decide pass/reject
    screening_passed = match_score >= SCREENING_THRESHOLD
    comparison = ">=" if screening_passed else "<"
    screening_reasoning = (
        f"{match.get('reasoning', '')} "
        f"Match score {match_score:.2f} {comparison} threshold {SCREENING_THRESHOLD}."
    ).strip()

    if screening_passed:
        print(f"\n[PASS] SCREENING PASSED (score {match_score:.2f} >= {SCREENING_THRESHOLD})")
        print("   -> Proceeding to Interview")
    else:
        print(f"\n[FAIL] SCREENING FAILED (score {match_score:.2f} < {SCREENING_THRESHOLD})")
        print("   -> Routing to Recruiter for the final reject decision")
    print("=" * 50)

    # write to state
    return {
        "parsed_cv": parsed_cv.model_dump(),
        "claims": [c.model_dump() for c in claims],
        "match_score": match_score,
        "screening_passed": screening_passed,
        "screening_reasoning": screening_reasoning,
    }