import os
import base64
from difflib import SequenceMatcher
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import requests
from langchain_core.tools import tool

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


def _github_headers():
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _fetch_readme_excerpt(username: str, repo: str, max_chars: int = 300) -> Optional[str]:
    readme_resp = requests.get(f"{GITHUB_API}/repos/{username}/{repo}/readme", headers=_github_headers(), timeout=10)
    if readme_resp.status_code != 200:
        return None
    content = readme_resp.json().get("content", "")
    try:
        return base64.b64decode(content).decode("utf-8", errors="ignore")[:max_chars]
    except Exception:
        return None


@tool
def github_verify_tool(username: str, repo: str, claimed_language: Optional[str] = None) -> dict:
    """
    Checks ONE claimed GitHub repo against reality: existence, primary
    language match, whether it has commit history, and a README excerpt.
    Use when a claim names a SPECIFIC repo. Never raises on a missing
    repo — reports absence instead.
    """
    repo_url = f"{GITHUB_API}/repos/{username}/{repo}"
    resp = requests.get(repo_url, headers=_github_headers(), timeout=10)

    if resp.status_code != 200:
        return {"exists": False, "reason": f"GitHub API returned {resp.status_code}"}

    data = resp.json()
    evidence = {
        "exists": True,
        "language": data.get("language"),
        "language_matches_claim": (
            claimed_language is not None
            and data.get("language") is not None
            and claimed_language.lower() == data.get("language", "").lower()
        ),
        "stars": data.get("stargazers_count", 0),
        "description": data.get("description"),
    }

    commits_resp = requests.get(
        f"{GITHUB_API}/repos/{username}/{repo}/commits",
        headers=_github_headers(),
        params={"per_page": 1},
        timeout=10,
    )
    evidence["has_commits"] = len(commits_resp.json()) > 0 if commits_resp.status_code == 200 else None
    evidence["readme_excerpt"] = _fetch_readme_excerpt(username, repo)

    return evidence


@tool
def github_profile_scan_tool(username: str, max_repos: int = 8) -> dict:
    """
    Scans a candidate's whole GitHub profile: lists their public repos
    (most recently updated first), collects each repo's language and
    description, and pulls README excerpts for the top few. Returns an
    aggregate `languages_used` list and per-repo summaries.

    This evidence applies to ANY claim (e.g. a bare skill claim like
    "Python" or "SQL"), not just claims that name a specific repo —
    unlike github_verify_tool, which only checks one named repo.
    Never raises on a missing/private user — reports that instead.
    """
    user_resp = requests.get(f"{GITHUB_API}/users/{username}", headers=_github_headers(), timeout=10)
    if user_resp.status_code != 200:
        return {"exists": False, "reason": f"GitHub API returned {user_resp.status_code} for user '{username}'"}

    repos_resp = requests.get(
        f"{GITHUB_API}/users/{username}/repos",
        headers=_github_headers(),
        params={"sort": "updated", "per_page": max_repos},
        timeout=10,
    )
    if repos_resp.status_code != 200:
        return {"exists": True, "repos": [], "reason": f"Could not list repos ({repos_resp.status_code})"}

    repos_data = repos_resp.json()
    languages_seen = set()
    repo_summaries = []

    for i, repo in enumerate(repos_data[:max_repos]):
        name = repo.get("name")
        language = repo.get("language")
        if language:
            languages_seen.add(language)

        summary = {
            "name": name,
            "language": language,
            "description": repo.get("description"),
            "stars": repo.get("stargazers_count", 0),
        }
        # Only pull READMEs for the first few repos, to stay light on API calls.
        if i < 5:
            summary["readme_excerpt"] = _fetch_readme_excerpt(username, name)

        repo_summaries.append(summary)

    return {
        "exists": True,
        "languages_used": sorted(languages_seen),
        "repos": repo_summaries,
    }


@tool
def consistency_check_tool(cv_claim: str, interview_answer: str) -> float:
    """
    Returns a 0-1 similarity score between a CV claim and an interview
    answer. Uses sequence-matching as a lightweight, dependency-free proxy
    for semantic similarity. This score is handed to the LLM as evidence
    for the SOM confidence call, not used as confidence directly.
    """
    return round(SequenceMatcher(None, cv_claim.lower(), interview_answer.lower()).ratio(), 3)