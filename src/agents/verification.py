"""Person C's evidence collection and structured claim assessment."""

from pydantic import Field, model_validator

from src.github_refs import extract_github_repositories, repository_urls
from src.llm import structured_call
from src.state import Claim, HRState, Record, Score, Text
from src.tools.verification_tools import (
    consistency_check_tool, github_profile_scan_tool, github_verify_tool,
)

# Provisional ceilings for evidence strength, not calibrated probabilities.
INTERVIEW_SUPPORT_LIMIT = 0.80
PUBLIC_EVIDENCE_SUPPORT_LIMIT = 0.95


class ClaimVerification(Record):
    claim_id: Text
    verified: bool | None
    confidence: Score | None
    source: Text
    evidence: Text
    evidence_refs: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_verdict(self):
        if self.verified is None and self.confidence is not None:
            raise ValueError("Unknown claims must have null confidence.")
        if self.verified is not None and self.confidence is None:
            raise ValueError("Supported or contradicted claims require a confidence score.")
        if self.verified is False and self.confidence > 0.5:
            raise ValueError("A contradicted claim cannot have high support for its truth.")
        return self


class VerificationResult(Record):
    claims: list[ClaimVerification]
    verification_notes: Text


def _extract_github_ref(text: str) -> tuple[str, str] | None:
    refs = extract_github_repositories(text)
    return refs[0] if refs else None


def _apply_evidence_rules(claim: Claim, assessment: ClaimVerification, allowed_refs: set[str]) -> tuple[dict, list[str]]:
    """Prevent an LLM source label from inventing access or independent verification."""
    refs = list(dict.fromkeys(assessment.evidence_refs))
    if not set(refs).issubset(allowed_refs):
        raise ValueError(f"Verifier cited unavailable evidence for {claim.claim_id}.")
    has_interview = any(ref.startswith("interview:") for ref in refs)
    has_github = any(ref.startswith(("profile:", "repo:")) for ref in refs)
    source = ("interview + github" if has_interview and has_github else
              "github" if has_github else "interview" if has_interview else "unavailable")
    verified, confidence = assessment.verified, assessment.confidence
    notes = []
    if verified is not None and not refs:
        verified, confidence = None, None
        notes.append("No accessible evidence was cited; the claim remains unknown.")
    if verified is True and claim.category in {"education", "experience"}:
        # Current tools provide public repository metadata and interview answers,
        # not institutional records or independent employment documentation.
        verified, confidence = None, None
        notes.append("Education or employment history lacks independent documentation; interview statements and repository metadata do not establish it.")
    if verified is not None:
        limit = PUBLIC_EVIDENCE_SUPPORT_LIMIT if has_github else INTERVIEW_SUPPORT_LIMIT
        if confidence > limit:
            confidence = limit
            notes.append(f"Support limited to {limit:.2f} by the available evidence type; this is not a calibrated probability.")
    if has_interview and not has_github:
        notes.append("Interview evidence only; no independent verification or proof of project authorship.")
    return {
        **claim.model_dump(), "verified": verified, "confidence": confidence,
        "source": source, "evidence_refs": refs,
        "evidence": " ".join([assessment.evidence, *notes]),
    }, notes


def verifier_node(state: HRState) -> dict:
    # Read raw input as well as summaries, including older saved states where A
    # dropped the links. Follow-up answers can introduce additional references.
    urls = repository_urls(
        state.cv_text, *state.github_repository_urls,
        *(state.parsed_cv.projects if state.parsed_cv else []),
        *(claim.text for claim in state.claims), *state.answers,
    )
    cache = dict(state.github_evidence)
    cached_keys = {key.lower(): key for key in cache}
    repositories = {}
    for owner, repo in extract_github_repositories(*urls):
        key = cached_keys.get(f"repo:{owner}/{repo}".lower(), f"repo:{owner}/{repo}")
        if key not in cache:
            cache[key] = github_verify_tool.invoke({"username": owner, "repo": repo})
        repositories[(owner.lower(), repo.lower())] = key
    reference_updates = {"github_repository_urls": urls, "github_evidence": cache}

    if not state.claims:
        return {
            **reference_updates,
            "claims": [], "consistency_flags": {}, "follow_up_needed": False,
            "verification_completed": True,
            "verification_notes": "No checkable claims were extracted; required evidence is incomplete.",
        }

    # Cache public evidence within this candidate's thread across follow-up rounds.
    username = state.parsed_cv.github_username if state.parsed_cv else None
    profile = None
    profile_ref = None
    if username:
        key = f"profile:{username}"
        if key not in cache:
            cache[key] = github_profile_scan_tool.invoke({"username": username})
        profile = cache[key]
        if profile.get("status") == "found":
            profile_ref = key

    blobs = []
    interview_refs = {f"interview:{i}" for i in range(1, len(state.answers) + 1)}
    allowed_by_claim = {}
    for claim in state.claims:
        # Keep citation eligibility tied to links in the claim. Unrelated raw-CV
        # references are fetched and recorded, but cannot boost its support score.
        claim_repos = {repositories[(owner.lower(), repo.lower())]
                       for owner, repo in extract_github_repositories(claim.text)}
        repo_refs = {key for key in claim_repos if cache[key].get("status") == "found"}
        allowed_by_claim[claim.claim_id] = interview_refs | repo_refs | ({profile_ref} if profile_ref else set())
        blobs.append({
            "claim_id": claim.claim_id, "text": claim.text, "category": claim.category,
            "github_repo_evidence": {key: cache[key] for key in sorted(claim_repos)},
            "allowed_evidence_refs": sorted(allowed_by_claim[claim.claim_id]),
            "lexical_overlap_only": [
                consistency_check_tool.invoke({"cv_claim": claim.text, "interview_answer": answer})
                for answer in state.answers
            ],
        })

    result = structured_call(VerificationResult,
        "Assess EVERY input claim exactly once by claim_id using only the supplied evidence. "
        "verified=true means supported by the available evidence, false means explicitly "
        "contradicted, null means unknown or insufficient evidence. Unknown claims must have "
        "null confidence. Otherwise confidence (0-1) measures support for the claim being TRUE: "
        "a clear contradiction needs a low value, never high confidence in a false claim. "
        "Explain each verdict in evidence and cite evidence_refs from that claim's allowed_evidence_refs. "
        "Use interview:N for the relevant numbered answer and the supplied profile/repo references "
        "only when their retrieved contents actually support your assessment. Never cite a failed "
        "or unavailable lookup. Unknown claims may have no references. Interview-only support must "
        "not exceed 0.80; public GitHub support must not exceed 0.95. These are provisional strength "
        "ceilings, not factual probabilities. Education and employment history must stay unknown "
        "without independent documentation; current tools do not provide that documentation. "
        "A technical explanation can support technical understanding, not independently prove that "
        "the candidate authored a project or achieved a reported historical metric. "
        "Repository existence, stars, "
        "language, forks, and READMEs do not prove candidate authorship or expertise. A public "
        "profile may provide supporting clues only. referenced_repositories includes links "
        "from the raw CV and answers even when no extracted claim mentions them. These are "
        "context only unless listed in a claim's allowed_evidence_refs; do not assign their "
        "ownership to the candidate or treat them as proof of unrelated claims. "
        "A detailed technically correct answer "
        "can support a skill; repeating a CV assertion cannot verify employment, a degree, "
        "or authorship. Simulated answers are demo evidence only, never independent facts. "
        "GitHub unavailable/not_found can mean private, rate limited, or inaccessible: these "
        "are not contradictions. Lexical overlap is not semantic consistency and cannot "
        "detect negation; read the full question/answer text. Account for clarifications "
        "in follow-up answers. Return concise notes about evidence limits.",
        {"claims": blobs, "profile_evidence": profile, "profile_evidence_ref": profile_ref,
         "referenced_repositories": {key: cache[key] for key in repositories.values()},
         "interview": [{"evidence_ref": f"interview:{i}", "question": question, "answer": answer}
                       for i, (question, answer) in enumerate(zip(state.questions, state.answers), 1)],
         "answer_source": state.answer_source})

    by_id = {item.claim_id: item for item in result.claims}
    expected = {claim.claim_id for claim in state.claims}
    if len(by_id) != len(result.claims) or set(by_id) != expected:
        raise ValueError("Verifier must return every input claim ID exactly once; no stale verdicts are accepted.")
    updated, adjustments = [], []
    for claim in state.claims:
        item, notes = _apply_evidence_rules(claim, by_id[claim.claim_id], allowed_by_claim[claim.claim_id])
        updated.append(item)
        if notes:
            adjustments.append(f"{claim.claim_id}: {' '.join(notes)}")
    flags = {item["claim_id"]: item["verified"] is False for item in updated}
    follow_up = any(
        item["verified"] is not True or item["confidence"] is None or item["confidence"] < 0.5
        for item in updated
    )
    supported = sum(item["verified"] is True for item in updated)
    contradicted = sum(item["verified"] is False for item in updated)
    unknown = len(updated) - supported - contradicted
    notes = (f"After evidence checks: {supported} supported, {contradicted} contradicted, "
             f"{unknown} unknown. Support from an interview is not independent verification. ")
    # The LLM's preliminary summary may no longer match the corrected verdicts.
    notes += " ".join(adjustments) if adjustments else result.verification_notes
    return {
        "claims": updated, "consistency_flags": flags, "follow_up_needed": follow_up,
        "verification_completed": True,
        "verification_notes": notes.strip(),
        **reference_updates,
    }
