"""Review packet + verdict ingestion (Claude Code or human).

``REVIEW.md`` tells the reviewer which contact sheets to open and what to
judge. Verdicts go into ``review.json`` next to it:

{
  "reviewer": "claude-code",
  "shots": [                      # one per predicted shot frame
    {"frame": 94,  "verdict": "ok"},
    {"frame": 42,  "verdict": "false_positive"},
    {"frame": 182, "verdict": "wrong", "stroke": "forehand", "player": "near_player",
     "gt_frame": 182, "fix_ground_truth": true}   # GT itself was wrong here
  ],
  "missing_shots": [{"frame": 129, "player": "far_player", "stroke": "slice"}],
  "points": [{"point_number": 1, "winner": "near_player", "server": "near_player"}],
  "notes": "free text"
}

``apply_review`` folds ``fix_ground_truth`` / ``missing_shots`` / ``points``
entries into a corrected ground-truth file, so reviews improve the labels
the next experiment is scored against.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from court_vision.research.clips import Clip
from court_vision.research.scorecard import Scorecard
from court_vision.review_data import load_match_json, save_match_json
from court_vision.shot_classify import MatchData, Shot, point_winner

STROKES = ("forehand", "backhand", "serve", "volley", "overhead", "slice")


def write_review_packet(
    exp_dir: Path,
    clip: Clip,
    sc: Scorecard,
    pred: MatchData,
    gt: MatchData | None,
    sheets: list[dict],
    overrides: dict,
) -> Path:
    lines: list[str] = []
    lines.append(f"# Review packet — {clip.name}")
    lines.append("")
    lines.append(f"Experiment: `{exp_dir}`  ")
    lines.append(f"Overrides: `{json.dumps(overrides)}`  ")
    lines.append(f"Clip notes: {clip.notes}")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append(f"`{sc.summary()}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in ("points_matched", "gt_points", "pred_points", "outcome_correct", "outcome_total",
              "shot_precision", "shot_recall", "shot_f1", "player_accuracy", "stroke_accuracy",
              "side_accuracy", "serve_precision", "serve_recall", "slice_precision", "slice_recall",
              "ball_coverage", "players_both_rate", "court_success_rate",
              "ball_detect_rate", "ball_loc_error_px", "player_mean_iou", "court_reproj_error_px"):
        v = getattr(sc, k)
        if isinstance(v, float):
            v = f"{v:.3f}"
        lines.append(f"| {k} | {v} |")
    if sc.stroke_confusion:
        lines.append("")
        lines.append("Stroke confusion (GT -> predicted):")
        lines.append("")
        for g, row in sc.stroke_confusion.items():
            lines.append(f"- {g}: " + ", ".join(f"{p}={n}" for p, n in row.items()))
    lines.append("")
    lines.append("## Predicted points and shots")
    lines.append("")
    for p in pred.points:
        lines.append(f"- **Point {p.point_number}** frames {p.start_frame}-{p.end_frame} "
                     f"server={p.server} winner={p.winner} (source={p.outcome_source})")
        for s in p.shots:
            lines.append(f"  - #{s.shot_number} f{s.frame} ({s.time_s:.2f}s) {s.player} **{s.stroke}** conf={s.confidence:.2f}"
                         + (f" placement={s.placement.zone}" if s.placement else ""))
    if gt is not None:
        lines.append("")
        lines.append("## Ground truth (human-corrected)")
        lines.append("")
        for p in gt.points:
            lines.append(f"- Point {p.point_number} frames {p.start_frame}-{p.end_frame} winner={point_winner(p)}")
            for s in p.shots:
                lines.append(f"  - #{s.shot_number} f{s.frame} {s.player} {s.stroke}")
    lines.append("")
    lines.append("## Keyframes to inspect")
    lines.append("")
    lines.append("Open each contact sheet (2 columns, up to 6 frames). Yellow trail = ball over the last "
                 "0.5s (hollow = interpolated), boxes = players, cyan = pose, white = projected court, "
                 "magenta cross = GT position. `shot_f<frame>_zoom.jpg` files are 1.6x crops around each contact.")
    lines.append("")
    for sh in sheets:
        lines.append(f"- `{sh['file']}`")
        for m in sh["items"]:
            if m["kind"] == "pred":
                lines.append(f"  - f{m['frame']} P{m['point']}: pred {m['pred']} | {m['gt']} -> {m['verdict']}")
            elif m["kind"] == "missed":
                lines.append(f"  - f{m['frame']} P{m['point']}: GT {m['gt']} MISSED")
            else:
                lines.append(f"  - f{m['frame']} P{m['point']}: {m['kind']}")
    lines.append("")
    lines.append("## How to review")
    lines.append("")
    lines.append("1. For every predicted shot decide: `ok` (right frame ±0.5s, right player, right stroke), "
                 "`wrong` (give the correct `stroke`/`player`), or `false_positive` (no hit here).")
    lines.append("2. For every GT shot marked MISSED confirm it is a real hit (add to `missing_shots`) or note the GT is wrong.")
    lines.append("3. For every point check `winner` and `server` against the scoreboard/broadcast; correct in `points`.")
    lines.append("4. If the *ground truth* is wrong, set `fix_ground_truth: true` on that entry so `research feedback apply` fixes the label.")
    lines.append("5. Write verdicts to `review.json` in this directory (schema in `court_vision/research/review.py`).")
    lines.append("")
    lines.append("Stroke vocabulary: " + ", ".join(STROKES) + ". Players: near_player (bottom of frame), far_player.")
    path = exp_dir / "REVIEW.md"
    path.write_text("\n".join(lines))
    return path


def load_review(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def apply_review_to_ground_truth(
    review: dict,
    gt_path: Path,
    out_path: Path | None = None,
    fps: float | None = None,
    tolerance_frames: int = 15,
) -> Path:
    """Fold GT corrections from a review into a (new) ground-truth file.

    Only entries with ``fix_ground_truth: true`` (shots) plus ``missing_shots``
    and ``points`` are applied; ``ok``/``false_positive`` verdicts describe
    the *prediction* and leave the labels alone.
    """
    gt = load_match_json(gt_path)
    fps = fps or float(gt.metadata.get("fps", 30.0))
    all_shots = [(p, s) for p in gt.points for s in p.shots]

    def nearest(frame: int):
        cands = [(abs(s.frame - frame), p, s) for p, s in all_shots if abs(s.frame - frame) <= tolerance_frames]
        return min(cands, key=lambda x: x[0]) if cands else None

    changes: list[str] = []
    for entry in review.get("shots", []):
        if not entry.get("fix_ground_truth"):
            continue
        frame = int(entry.get("gt_frame", entry["frame"]))
        hit = nearest(frame)
        if hit is None:
            continue
        _, p, s = hit
        if entry.get("verdict") == "false_positive" or entry.get("delete"):
            p.shots.remove(s)
            changes.append(f"removed GT shot f{s.frame}")
            continue
        for k in ("stroke", "player"):
            if entry.get(k) and getattr(s, k) != entry[k]:
                changes.append(f"f{s.frame}: {k} {getattr(s, k)} -> {entry[k]}")
                setattr(s, k, entry[k])
        if entry.get("frame") and int(entry["frame"]) != s.frame:
            changes.append(f"f{s.frame}: frame -> {entry['frame']}")
            s.frame = int(entry["frame"])
            s.time_s = s.frame / fps

    for entry in review.get("missing_shots", []):
        frame = int(entry["frame"])
        if nearest(frame) is not None:
            continue
        point = next((p for p in gt.points if p.start_frame <= frame <= p.end_frame), None)
        if point is None:
            point = min(gt.points, key=lambda p: min(abs(p.start_frame - frame), abs(p.end_frame - frame)))
        point.shots.append(Shot(shot_number=0, frame=frame, time_s=frame / fps, player=entry["player"],
                                stroke=entry.get("stroke", "forehand"), placement=None, confidence=1.0))
        changes.append(f"added GT shot f{frame} {entry['player']} {entry.get('stroke')}")

    for entry in review.get("points", []):
        p = next((p for p in gt.points if p.point_number == entry.get("point_number")), None)
        if p is None:
            continue
        if entry.get("winner"):
            p.winner = entry["winner"]
            last = p.shots[-1].player if p.shots else None
            p.outcome = "winner" if entry["winner"] == last else "error"
            p.outcome_player = last
            changes.append(f"point {p.point_number}: winner -> {entry['winner']}")
        if entry.get("server"):
            p.server = entry["server"]

    for p in gt.points:
        p.shots.sort(key=lambda s: s.frame)
        for i, s in enumerate(p.shots, 1):
            s.shot_number = i
        p.rally_length = len(p.shots)
        p.review_status = "corrected"
    gt.metadata["review_history"] = gt.metadata.get("review_history", []) + [
        {"date": str(date.today()), "reviewer": review.get("reviewer", "unknown"), "changes": changes}
    ]
    out_path = Path(out_path or gt_path)
    if out_path == Path(gt_path):
        backup = gt_path.with_suffix(".bak.json")
        backup.write_text(Path(gt_path).read_text())
    save_match_json(gt, out_path)
    return out_path
