"""Extract repository links without depending on an LLM's CV summary."""

import re
from urllib.parse import urlsplit


_LINK = re.compile(
    r"(?<![\w/@.:-])(?:https?://(?:www\.)?github\.com|(?:www\.)?github\.com)"
    r"/[^\s<>\"'\]\[(){}]+", re.IGNORECASE,
)


def extract_github_repositories(*texts: str) -> list[tuple[str, str]]:
    """Return unique owner/repo pairs, preserving the first spelling.

    Profile links are excluded; tree/blob/issue paths resolve to the repository.
    A link is a reference, never evidence of the candidate's ownership.
    """
    found = {}
    for text in texts:
        for match in _LINK.finditer(text):
            raw = match.group().rstrip(".,;:!?")
            url = urlsplit(raw if "://" in raw else "https://" + raw)
            if url.netloc.lower() not in {"github.com", "www.github.com"}:
                continue
            parts = url.path.strip("/").split("/")
            if len(parts) < 2:
                continue
            owner, repo = parts[:2]
            repo = re.sub(r"\.git$", "", repo, flags=re.IGNORECASE)
            if (not re.fullmatch(r"[A-Za-z0-9-]{1,39}", owner)
                    or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", repo)
                    or repo in {".", ".."}):
                continue
            found.setdefault((owner.lower(), repo.lower()), (owner, repo))
    return list(found.values())


def repository_urls(*texts: str) -> list[str]:
    return [f"https://github.com/{owner}/{repo}"
            for owner, repo in extract_github_repositories(*texts)]
