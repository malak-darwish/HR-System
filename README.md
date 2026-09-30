# Multi-Agent HR System

A LangGraph-based hiring pipeline that takes a candidate from CV submission, through a live interview, to a final **hire / waitlist / reject** decision. Four specialized agents share one Pydantic state and hand off to each other through conditional edges. The Verification Agent cross-checks CV claims against interview answers and GitHub evidence, and triggers follow-up questions when it finds contradictions.

---

## How It Works

```
CV + Job Description
        │
        ▼
 ┌──────────────┐   fails (< 0.6)
 │   Screener   │ ─────────────────────────────┐
 └──────────────┘                              │
        │ passes                               │
        ▼                                      │
 ┌──────────────┐                              │
 │ Interviewer  │ ◄──────────┐                 │
 └──────────────┘            │ follow-up       │
        │                    │ (max 2)         │
        ▼                    │                 │
 ┌──────────────┐            │                 │
 │ Verification │ ───────────┘                 │
 └──────────────┘                              │
        │ resolved / cap reached               │
        ▼                                      ▼
 ┌──────────────┐                     ┌──────────────┐
 │  Recruiter   │ ──────────────────► │   Decision   │
 └──────────────┘                     └──────────────┘
                              hire / waitlist / reject
```

| Agent | Owner | Role |
|---|---|---|
| **Screener** | Person A | Parses the CV, extracts claims (each with a unique `claim_id`), scores the match against the job description |
| **Interviewer** | Person B | Asks questions based on the CV and job, and generates targeted follow-ups for flagged claims |
| **Verification** | Person C | Checks each claim against interview answers and GitHub evidence using Structured Output Mode; flags contradictions |
| **Recruiter** | Person D | Aggregates scores and verification results, applies the decision policy, owns the graph |

---

## Key Features

- **Contradiction detection:** if a candidate says one thing in the CV or an earlier answer and something different later, the claim is flagged and the Interviewer asks a follow-up.
- **Code-decided resolution:** the LLM reports which answers support or deny each claim; Python decides whether a contradiction is resolved (a clean supporting answer must come *after* the last denial). Resolved claims are capped at 0.6 confidence; unresolved ones are forced to `verified=False` at 0.2.
- **GitHub evidence:** a profile-wide scan of the candidate's public repos (languages, README excerpts) is used as evidence for every claim.
- **Bounded follow-up loop:** at most 2 follow-up rounds before the pipeline moves to the Recruiter.
- **Live web interface:** job board plus a chat interview where the candidate types answers in real time.

---

## Project Requirements Mapping

| Req. | Requirement | Where |
|---|---|---|
| A | Multiple agents | 4 agents in `src/agents/` |
| B | Distinct toolsets per agent | `src/tools/` (screener, interviewer, verification, recruiter tools) |
| C | Pydantic state + MemorySaver | `src/state.py`, checkpointer in `src/graph.py` |
| D | Structured Output Mode | Verification Agent (`VerificationResult` schema) |
| E | Non-string / Optional fields | e.g. `match_score: float`, `verified: Optional[bool]`, `github_evidence` |
| F | Conditional edges | Screening pass/fail, follow-up loop, final decision routing |
| G | Interface | Flask app in `src/web/` |

---

## Tech Stack

Python · LangGraph · LangChain · Pydantic · Google Gemini (`langchain_google_genai`) · Flask · GitHub API

---

## Project Structure

```
HR-System/
├── src/
│   ├── state.py              # Shared Pydantic contract (single source of truth)
│   ├── llm.py                # get_model() — shared Gemini config
│   ├── graph.py              # LangGraph wiring + conditional edges
│   ├── main.py               # CLI pipeline runner
│   ├── agents/
│   │   ├── screener.py
│   │   ├── interviewer.py
│   │   ├── verification.py
│   │   └── recruiter.py
│   ├── tools/
│   │   ├── interviewer_tools.py
│   │   ├── verification_tools.py
│   │   ├── recruiter_tools.py
│   │   └── requirement_tools.py
│   ├── data/
│   │   ├── sample_cv.txt
│   │   └── sample_jd.txt
│   └── web/
│       ├── app.py            # Flask server
│       ├── jobs.json         # Open roles
│       └── static/index.html # Job board + live interview UI
├── .env
└── requirements.txt
```

---

## Setup

```bash
git clone <repo-url>
cd HR-System
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

---

## Running

Always run from the project root.

**Web interface (live interview):**
```bash
python -m src.web.app
```
**CLI pipeline (simulated candidate):**

## Decision Outcomes

| Outcome | When |
|---|---|
| **Hire** | Screening passed, claims verified, no unresolved contradictions |
| **Waitlist** | Contradiction still unresolved after the follow-up cap |
| **Reject** | Screening score below 0.6, or failed core requirements |

---

## Limitations

- Verification relies on public GitHub data; private work can't be confirmed.
- Degree and soft-skill claims can't be verified from interview answers alone.
- LLM calls depend on the Gemini API (retries are configured for transient errors).
