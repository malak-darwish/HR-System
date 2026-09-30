from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Score = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Decision = Literal["hire", "reject", "waitlist"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParsedCV(Record):
    name: str = ""
    email: str = ""
    phone: str = ""
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    github_username: str | None = None
    years_of_experience: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None


class Claim(Record):
    claim_id: Text
    text: Text
    category: Literal["experience", "skill", "education", "project"]
    verified: Annotated[bool, Field(strict=True)] | None = None
    confidence: Score | None = None
    source: Text | None = None
    evidence: Text | None = None
    evidence_refs: list[Text] = Field(default_factory=list)


class RequirementAssessment(Record):
    requirement_id: Text
    requirement: Text
    job_description_quote: Text
    evidence_kind: Literal["capability", "documentary"]
    claim_ids: list[Text] = Field(default_factory=list)
    supported: bool
    reasoning: Text


class HRState(Record):
    cv_text: str = ""
    job_description: str = ""
    parsed_cv: ParsedCV | None = None
    match_score: Score | None = None
    screening_passed: Annotated[bool, Field(strict=True)] | None = None
    screening_reasoning: Text | None = None
    claims: list[Claim] = Field(default_factory=list)

    questions: list[Text] = Field(default_factory=list)
    answers: list[Text] = Field(default_factory=list)
    interview_scores: list[Score] = Field(default_factory=list)
    interview_score_reasoning: list[Text] = Field(default_factory=list)
    interview_complete: bool = False
    answer_source: Literal["candidate", "simulated"] = "candidate"

    consistency_flags: dict[str, Annotated[bool, Field(strict=True)]] = Field(default_factory=dict)
    follow_up_needed: bool = False
    verification_completed: bool = False
    verification_notes: Text | None = None
    github_repository_urls: list[Text] = Field(default_factory=list)
    github_evidence: dict[str, Any] = Field(default_factory=dict)

    interview_round: Annotated[int, Field(strict=True, ge=0)] = 0
    follow_up_count: Annotated[int, Field(strict=True, ge=0)] = 0
    max_follow_ups: Annotated[int, Field(strict=True, ge=0, le=5)] = 2

    overall_score: Score | None = None
    interview_average: Score | None = None
    requirement_assessments: list[RequirementAssessment] = Field(default_factory=list)
    requirement_coverage: Score | None = None
    required_evidence_complete: bool = False
    scored_claim_ids: list[Text] = Field(default_factory=list)
    excluded_claim_ids: list[Text] = Field(default_factory=list)
    verification_limitations: list[Text] = Field(default_factory=list)
    final_decision: Decision | None = None
    decision_reasoning: Text | None = None

    @model_validator(mode="after")
    def check_contract(self):
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("Claim IDs must be unique.")
        if not set(self.consistency_flags).issubset(ids):
            raise ValueError("Consistency flags must refer to existing claim IDs.")
        if len(self.questions) != len(self.answers):
            raise ValueError("Every interview question must have one answer.")
        if len(self.interview_scores) > len(self.answers):
            raise ValueError("Interview scores cannot outnumber answers.")
        if len(self.interview_score_reasoning) != len(self.interview_scores):
            raise ValueError("Every interview score needs a rationale.")
        if self.follow_up_count > self.max_follow_ups:
            raise ValueError("Follow-up count exceeds the configured limit.")
        return self