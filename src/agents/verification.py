import os
import re
import sys
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from state import HRState, Claim
from tools.verification_tools import (
    github_verify_tool,
    github_profile_scan_tool,
    consistency_check_tool,
)

class ClaimVerification(BaseModel):
    text: str = Field(description="Must exactly match the original claim text, unchanged")
    verified: bool
    confidence: float = Field(description="0-1: how likely the claim is to be TRUE")
    source: str = Field(description='Where the verification evidence came from, e.g. "github", "interview", "unverifiable"')


class VerificationResult(BaseModel):
    """Structured Output Mode target — the LLM returns exactly this shape."""

    claims: List[ClaimVerification] = Field(description="One entry per input claim, in the same order")
    follow_up_needed: bool = Field(description="True if any claim is contradicted or low-confidence")
    consistency_flags: Dict[str, bool] = Field(
        description="Map of claim text -> whether it is internally consistent with interview answers"
    )
    verification_notes: str = Field(description="2-3 sentence overall summary of what was found across all claims")


def get_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        max_retries=6,  
    )


def verifier_node(state: HRState) -> dict:
    """
    LangGraph node function for the Verification Agent.
    Returns a partial-state dict, as LangGraph nodes are expected to.
    """
    claims: List[Claim] = state.claims
    answers: List[str] = state.answers

    # one profile-wide GitHub scan per candidate (not per claim), so a bare
    # skill claim like "Python" can be checked against real repo languages
    # and READMEs, not just claims that literally say "github" in the text.
    profile_evidence = None
    if state.parsed_cv and state.parsed_cv.github_username:
        profile_evidence = github_profile_scan_tool.invoke({"username": state.parsed_cv.github_username})

    evidence_blobs = []
    for claim in claims:
        blob = {"claim": claim.text, "github_repo_evidence": None, "consistency_scores": []}
        # (gives commit history / direct language match a profile scan can't).
        if "github" in claim.text.lower() or "repo" in claim.text.lower():
            username, repo = _extract_github_ref(claim.text, state)
            if username and repo:
                blob["github_repo_evidence"] = github_verify_tool.invoke({"username": username, "repo": repo})

        for answer in answers:
            score = consistency_check_tool.invoke({"cv_claim": claim.text, "interview_answer": answer})
            blob["consistency_scores"].append({"answer": answer, "score": score})

        evidence_blobs.append(blob)

    prompt = _build_verification_prompt(claims, evidence_blobs, profile_evidence)

    structured_model = get_model().with_structured_output(VerificationResult)
    result: VerificationResult = structured_model.invoke(prompt)

    # Merge SOM results back into the ORIGINAL Claim objects so we keep
    # category instead of losing it.
    verified_by_text = {cv.text: cv for cv in result.claims}
    updated_claims: List[Claim] = []
    for claim in claims:
        match = verified_by_text.get(claim.text)
        if match:
            updated_claims.append(
                claim.model_copy(update={
                    "verified": match.verified,
                    "confidence": match.confidence,
                    "source": match.source,
                })
            )
        else:
            # SOM didn't return this claim for some reason, leave it untouched
            # rather than silently dropping it.
            updated_claims.append(claim)

    return {
        "claims": updated_claims,
        "follow_up_needed": result.follow_up_needed,
        "consistency_flags": result.consistency_flags,
        "verification_notes": result.verification_notes,
    }


def _extract_github_ref(claim_text: str, state: HRState):
    """
    Tries the claim text first (e.g. "...github.com/user/repo..."), then
    falls back to state.parsed_cv.github_username if the Screener already
    extracted it, paired with a repo-name guess from the claim text.
    Returns (username, repo) or (None, None).
    """
    match = re.search(r"github\.com/([\w-]+)/([\w-]+)", claim_text)
    if match:
        return match.group(1), match.group(2)

    if state.parsed_cv and state.parsed_cv.github_username:
        repo_match = re.search(r"[\w-]+/([\w-]+)", claim_text)
        if repo_match:
            return state.parsed_cv.github_username, repo_match.group(1)

    return None, None


def _build_verification_prompt(claims: List[Claim], evidence_blobs: list, profile_evidence: Optional[dict]) -> str:
    lines = [
        "You are verifying claims from a candidate's CV against gathered evidence.",
        "For each claim, decide `verified` (bool), `confidence` (0-1 float), and `source` "
        '(where the evidence came from, e.g. "github", "interview", "unverifiable").',
        "Base every decision strictly on the evidence given below — do not invent facts not present in it.",
        "`confidence` means how likely the claim is to be TRUE, not how sure you are of your verdict. "
        "A clearly FALSE claim should get a LOW confidence (e.g. 0.05-0.2), a clearly TRUE claim should "
        "get a HIGH confidence (e.g. 0.8-0.95). Never report high confidence for a claim you marked unverified.",
        "Set `follow_up_needed` to true if ANY claim is contradicted or has confidence below 0.5.",
        "Populate `consistency_flags` as claim_text -> bool (true = consistent, false = contradicted).",
        "Write `verification_notes` as a short 2-3 sentence summary of the overall findings.",
        "IMPORTANT: each returned claim's `text` field must exactly match the original claim text below, unchanged.",
        "",
    ]

    if profile_evidence is not None:
        lines.append("=== Candidate's GitHub profile (applies to ALL claims below, not just repo-specific ones) ===")
        if profile_evidence.get("exists"):
            lines.append(f"Languages used across public repos: {profile_evidence.get('languages_used')}")
            for repo in profile_evidence.get("repos", []):
                lines.append(f"  - Repo '{repo['name']}' ({repo.get('language')}): {repo.get('description')}")
                if repo.get("readme_excerpt"):
                    lines.append(f"    README excerpt: {repo['readme_excerpt']}")
        else:
            lines.append(f"GitHub profile not found or inaccessible: {profile_evidence.get('reason')}")
        lines.append("")

    for claim, blob in zip(claims, evidence_blobs):
        lines.append(f"Claim: {claim.text}")
        if blob["github_repo_evidence"] is not None:
            lines.append(f"  Specific-repo evidence: {blob['github_repo_evidence']}")
        for cs in blob["consistency_scores"]:
            lines.append(f'  Interview answer: "{cs["answer"]}" (similarity: {cs["score"]})')
        lines.append("")

    return "\n".join(lines)