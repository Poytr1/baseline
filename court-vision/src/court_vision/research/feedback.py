"""Human feedback log: append-only JSONL per clip, applied to ground truth.

    court-vision research feedback add houston28s --frame 1216 --player near_player --stroke forehand
    court-vision research feedback add houston28s --point 2 --winner far_player
    court-vision research feedback add houston28s --frame 888 --delete   # not a hit
    court-vision research feedback apply houston28s

Entries share the review.json vocabulary so Claude reviews and human notes
flow through the same ``apply_review_to_ground_truth``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from court_vision.research.clips import get_clip
from court_vision.research.review import apply_review_to_ground_truth

FEEDBACK_DIR = Path("research/feedback")


def feedback_path(clip_name: str, root: Path = FEEDBACK_DIR) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{clip_name}.jsonl"


def add_feedback(clip_name: str, entry: dict, root: Path = FEEDBACK_DIR, author: str = "human") -> Path:
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "author": author, **entry}
    p = feedback_path(clip_name, root)
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return p


def load_feedback(clip_name: str, root: Path = FEEDBACK_DIR) -> list[dict]:
    p = feedback_path(clip_name, root)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def feedback_to_review(entries: list[dict]) -> dict:
    """Convert feedback lines into the review.json structure."""
    review = {"reviewer": "feedback", "shots": [], "missing_shots": [], "points": [], "notes": ""}
    for e in entries:
        if e.get("point") is not None and (e.get("winner") or e.get("server")):
            review["points"].append({"point_number": int(e["point"]), "winner": e.get("winner"), "server": e.get("server")})
        elif e.get("frame") is not None:
            if e.get("delete"):
                review["shots"].append({"frame": int(e["frame"]), "verdict": "false_positive", "fix_ground_truth": True})
            elif e.get("add"):
                review["missing_shots"].append({"frame": int(e["frame"]), "player": e.get("player"), "stroke": e.get("stroke")})
            else:
                review["shots"].append({"frame": int(e["frame"]), "verdict": "wrong", "stroke": e.get("stroke"),
                                        "player": e.get("player"), "fix_ground_truth": True})
        if e.get("note"):
            review["notes"] += e["note"] + "\n"
    return review


def apply_feedback(clip_name: str, root: Path = FEEDBACK_DIR, registry: Path | None = None) -> Path | None:
    clip = get_clip(clip_name, registry)
    entries = load_feedback(clip_name, root)
    if not entries or not clip.has_ground_truth:
        return None
    review = feedback_to_review(entries)
    return apply_review_to_ground_truth(review, clip.ground_truth)
