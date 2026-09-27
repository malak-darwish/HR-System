"""Person C's public GitHub tools and a lexical-overlap diagnostic."""

import base64
import os
import re
from difflib import SequenceMatcher

import requests
from langchain_core.tools import tool

GITHUB_API = "https://api.github.com"


def _github_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(path: str, *, params: dict | None = None) -> dict:
    try:
        response = requests.get(
            GITHUB_API + path, headers=_github_headers(), params=params, timeout=10,
        )
        if response.status_code == 404:
            return {"status": "not_found", "reason": "Not publicly accessible (missing or private)."}
        if response.status_code != 200:
            return {"status": "unavailable", "reason": f"GitHub HTTP {response.status_code}."}
        return {"status": "found", "data": response.json()}
    except (requests.RequestException, ValueError):
        # Exception messages can contain request details. Return only a safe category.
        return {"status": "unavailable", "reason": "GitHub request failed or returned invalid JSON."}


def _valid_ref(username: str, repo: str | None = None) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9-]{1,39}", username)) and (
        repo is None or bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", repo))
        and repo not in {".", ".."})


def _readme(username: str, repo: str) -> dict:
    result = _get(f"/repos/{username}/{repo}/readme")
    if result["status"] != "found":
        return result
    data = result["data"]
    try:
        excerpt = base64.b64decode(data["content"]).decode("utf-8", errors="replace")[:1000]
        return {"status": "found", "excerpt": excerpt}
    except (KeyError, TypeError, ValueError):
        return {"status": "unavailable", "reason": "README content could not be decoded."}


@tool
def github_verify_tool(username: str, repo: str, claimed_language: str | None = None) -> dict:
    """Check one public repo; inaccessible evidence never means a false candidate claim."""
    if not _valid_ref(username, repo):
        return {"status": "unavailable", "exists": None, "reason": "Invalid GitHub reference."}
    result = _get(f"/repos/{username}/{repo}")
    if result["status"] != "found":
        return {**result, "exists": None}
    data = result["data"]
    if not isinstance(data, dict):
        return {"status": "unavailable", "exists": None, "reason": "Unexpected repository response."}
    language = data.get("language")
    commits = _get(f"/repos/{username}/{repo}/commits", params={"per_page": 1})
    return {
        "status": "found", "exists": True,
        "url": f"https://github.com/{username}/{repo}",
        "language": language,
        "language_matches_claim": (
            claimed_language.casefold() == language.casefold() if claimed_language and language else None),
        "description": data.get("description"), "stars": data.get("stargazers_count", 0),
        "fork": data.get("fork", False),
        "has_commits": bool(commits["data"]) if commits["status"] == "found" else None,
        "commits_status": commits["status"], "readme": _readme(username, repo),
    }


@tool
def github_profile_scan_tool(username: str, max_repos: int = 8) -> dict:
    """Collect public repo metadata and a few READMEs, without inferring authorship."""
    if not _valid_ref(username) or not 1 <= max_repos <= 20:
        return {"status": "unavailable", "exists": None, "reason": "Invalid profile scan arguments."}
    user = _get(f"/users/{username}")
    if user["status"] != "found":
        return {**user, "exists": None}
    listing = _get(f"/users/{username}/repos", params={"sort": "updated", "per_page": max_repos})
    if listing["status"] != "found":
        return {**listing, "exists": True, "repos": []}
    if not isinstance(listing["data"], list):
        return {"status": "unavailable", "exists": True, "reason": "Unexpected repository list."}
    repos = []
    for index, data in enumerate(listing["data"][:max_repos]):
        if not isinstance(data, dict):
            continue
        name = data.get("name", "")
        summary = {key: data.get(key) for key in ("name", "language", "description", "fork")}
        if index < 3 and isinstance(name, str) and _valid_ref(username, name):
            summary["readme"] = _readme(username, name)
        repos.append(summary)
    return {
        "status": "found", "exists": True, "repos": repos,
        "languages_used": sorted({r["language"] for r in repos if r.get("language")}),
        "scope": f"At most {max_repos} recent public repos; absence is not proof of a false claim.",
    }


@tool
def consistency_check_tool(cv_claim: str, interview_answer: str) -> float:
    """Return lexical overlap ONLY; this cannot detect negation or establish truth."""
    return round(SequenceMatcher(None, cv_claim.lower(), interview_answer.lower()).ratio(), 3)
