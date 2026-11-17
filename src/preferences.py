"""Local human-preference store for the old-vs-new comparison.

Votes are appended as one JSON object per line to artifacts/preferences.jsonl.
No database, no network: the file is the record, and it can be read with any tool.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import ART_DIR

PATH = ART_DIR / "preferences.jsonl"
CHOICES = ("original", "merged")

_lock = threading.Lock()


def load(path: Path = PATH) -> list[dict]:
    """Read all votes, skipping malformed lines instead of failing."""
    if not path.exists():
        return []
    votes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            votes.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return votes


def tally(votes: list[dict]) -> dict:
    """Count choices; unknown values are ignored rather than guessed."""
    original = sum(1 for v in votes if v.get("choice") == "original")
    merged = sum(1 for v in votes if v.get("choice") == "merged")
    total = original + merged
    return {
        "original": original,
        "merged": merged,
        "total": total,
        "merged_share": round(100 * merged / total, 1) if total else 0.0,
    }


def record(vote: dict, path: Path = PATH) -> dict:
    """Append one vote and return the updated summary."""
    if vote.get("choice") not in CHOICES:
        raise ValueError("choice must be 'original' or 'merged'")
    entry = {**vote, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return summary(path)


def summary(path: Path = PATH) -> dict:
    votes = load(path)
    return {"tally": tally(votes), "recent": votes[-5:]}
