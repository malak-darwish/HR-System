"""D's job-specific evidence gate; never changes C's claim verdicts."""

from typing import Literal

from pydantic import Field

from llm import structured_call
from state import Claim, HRState, Record, RequirementAssessment, Text


class RequiredQualification(Record):
    requirement: Text
    job_description_quote: Text
    evidence_kind: Literal["capability", "documentary"]


class MandatoryRequirements(Record):
    requirements: list[RequiredQualification] = Field(default_factory=list)


class RequirementMatch(Record):
    requirement_id: Text
    claim_ids: list[Text] = Field(default_factory=list)
    evidence_sufficient: bool
    reasoning: Text


class RequirementMatches(Record):
    matches: list[RequirementMatch] = Field(default_factory=list)


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _cited_assessment(claim: Claim, state: HRState) -> bool:
    """Require an actual C verdict and citations accessible in this state."""
    if (claim.verified is None or claim.confidence is None or not claim.source
            or not claim.evidence or not claim.evidence_refs):
        return False
    allowed = {f"interview:{i}" for i in range(1, len(state.answers) + 1)}
    allowed.update(key for key, data in state.github_evidence.items()
                   if key.startswith(("repo:", "profile:"))
                   and isinstance(data, dict) and data.get("status") == "found")
    return set(claim.evidence_refs).issubset(allowed)


def assess_job_requirements(state: HRState) -> dict:
    """Extract requirements before seeing the candidate, then map C's evidence.

    Interpretation is model-based; ID coverage, quote provenance, cited evidence,
    and policy eligibility are checked in Python. A missing/empty interpretation
    cannot authorize a hire. No education/experience verdict is promoted here.
    """
    extracted = (structured_call(MandatoryRequirements,
        "Extract ALL mandatory job qualifications from the supplied job description, "
        "including essential technical and behavioral capabilities. Do not inspect or "
        "assume a candidate. Separate independently required capabilities into distinct "
        "requirements. Preserve alternatives such as 'degree OR equivalent experience' "
        "as one requirement; do not require both. Exclude explicitly optional, preferred, "
        "nice-to-have, or explicitly NOT required qualifications. Do NOT extract degree or "
        "education requirements, or interpersonal/soft-skill requirements (communication, "
        "teamwork); these are assessed manually by HR outside this system. Do not turn routine "
        "responsibilities into mandatory past employment. Use a verbatim supporting "
        "job_description_quote for each requirement. evidence_kind=capability for current "
        "knowledge/skills that can be assessed technically, including 'experience with' "
        "a tool when no historical duration or credential is required. Use documentary "
        "for mandatory degrees, licenses, employment history, years of experience, and "
        "interpersonal qualities (communication, teamwork) that require human assessment. "
        "For alternatives, use capability only if demonstrating a capability alone is "
        "explicitly sufficient. Return an empty list if no requirements can be identified.",
        {"job_description": state.job_description})
        if state.job_description.strip() else MandatoryRequirements())
    requirements = []
    for index, item in enumerate(extracted.requirements, 1):
        if _normalized(item.job_description_quote) not in _normalized(state.job_description):
            raise ValueError("A mandatory requirement must quote the supplied job description.")
        requirements.append({"requirement_id": f"R{index}", **item.model_dump()})

    by_id = {claim.claim_id: claim for claim in state.claims}
    if requirements and state.claims:
        available = {f"interview:{i}": {"question": q, "answer": a}
                     for i, (q, a) in enumerate(zip(state.questions, state.answers), 1)}
        available.update({key: data for key, data in state.github_evidence.items()
                          if isinstance(data, dict) and data.get("status") == "found"})
        mapped = structured_call(RequirementMatches,
            "Map EVERY supplied mandatory requirement exactly once by requirement_id to "
            "the supplied C claim assessments. Include a requirement even if the CV has "
            "no matching claim: return an empty claim_ids list and evidence_sufficient=false. "
            "Use only existing claim IDs. Select claims relevant to that requirement, "
            "not every claim in the CV. evidence_sufficient=true only when the cited "
            "claim assessments support the FULL requirement, including all parts of a "
            "compound requirement or one complete allowed alternative. Read the evidence "
            "and cited_evidence contents attached to each claim; do not infer coverage "
            "from a broad skill label. Only C's existing citations are provided. "
            "Judge support from the cited evidence text; superficial word overlap or a partly supported bundled claim is "
            "insufficient. Do not change C's verdicts, invent evidence, or promote unknown "
            "claims. Technical understanding does not prove employment, degrees, licenses, "
            "project authorship, or historical performance. Explain remaining gaps.",
            {"requirements": requirements,
             "claims": [{**claim.model_dump(), "cited_evidence": {
                 ref: available[ref] for ref in claim.evidence_refs if ref in available
             }} for claim in state.claims]})
        matches = {item.requirement_id: item for item in mapped.matches}
        if (len(matches) != len(mapped.matches)
                or set(matches) != {r["requirement_id"] for r in requirements}):
            raise ValueError("Requirement mapping must cover every mandatory requirement exactly once.")
        if any(not set(item.claim_ids).issubset(by_id) for item in matches.values()):
            raise ValueError("Requirement mapping cited an unknown claim ID.")
    else:
        matches = {r["requirement_id"]: RequirementMatch(
            requirement_id=r["requirement_id"], claim_ids=[], evidence_sufficient=False,
            reasoning="No candidate claims are available to support this requirement.",
        ) for r in requirements}

    assessments, relevant_ids = [], set()
    for requirement in requirements:
        match = matches[requirement["requirement_id"]]
        ids = list(dict.fromkeys(match.claim_ids))
        relevant_ids.update(ids)
        supported = bool(match.evidence_sufficient and ids and all(
            _cited_assessment(by_id[cid], state) and by_id[cid].verified is True
            and by_id[cid].confidence >= 0.5
            and not state.consistency_flags.get(cid, False) for cid in ids
        ))
        reason = match.reasoning
        if requirement["evidence_kind"] == "documentary":
            # The current tools do not retrieve authenticated degree/employer records.
            supported = False
            reason += " Independent credential/employment documentation is required; current tools do not provide it."
        elif match.evidence_sufficient and not supported:
            reason += " C's cited assessments do not establish complete, non-contradicted support."
        assessments.append(RequirementAssessment(
            **requirement, claim_ids=ids, supported=supported, reasoning=reason,
        ))

    # A partial numeric score can still be shown, but missing mandatory evidence
    # separately prevents a hire. Unknowns are never assigned zero or full credit.
    scored = [c.claim_id for c in state.claims
              if c.claim_id in relevant_ids and _cited_assessment(c, state)]
    excluded = [c.claim_id for c in state.claims if c.claim_id not in scored]
    coverage = sum(r.supported for r in assessments) / len(assessments) if assessments else None
    limitations = []
    if not assessments:
        limitations.append("No mandatory job requirements could be established; manual review is required.")
    missing = [r.requirement_id for r in assessments if not r.supported]
    if missing:
        limitations.append("Mandatory requirements lacking sufficient evidence: " + ", ".join(missing) + ".")
    unknown = [c.claim_id for c in state.claims if c.verified is None]
    if unknown:
        limitations.append("Claims still unverified: " + ", ".join(unknown) + ". Their verdicts remain unchanged.")
    contradicted = [c.claim_id for c in state.claims
                    if c.verified is False or state.consistency_flags.get(c.claim_id, False)]
    if contradicted:
        limitations.append("Unresolved contradictions requiring review: " + ", ".join(contradicted) + ".")
    if excluded:
        limitations.append("Claims excluded from the role-specific score (unassessed or not mapped to mandatory requirements): "
                           + ", ".join(excluded) + ".")
    if scored:
        limitations.append("The overall score uses assessed claims mapped to mandatory job requirements; it is not verification of the entire CV.")
    interview_only = [c.claim_id for c in state.claims if c.claim_id in scored and c.source == "interview"]
    if interview_only:
        limitations.append("Interview-only support, not independent verification: " + ", ".join(interview_only) + ".")

    limitations.append("Degree and soft-skill requirements are excluded from automated "
                       "assessment and require manual HR review.")
    return {
        "requirement_assessments": [r.model_dump() for r in assessments],
        "requirement_coverage": coverage,
        "required_evidence_complete": bool(assessments) and all(r.supported for r in assessments),
        "scored_claim_ids": scored, "excluded_claim_ids": excluded,
        "verification_limitations": limitations,
    }