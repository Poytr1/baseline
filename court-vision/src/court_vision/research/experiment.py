"""Experiments: run a clip with a config, score it, write a review packet.

An experiment directory (``runs/experiments/<stamp>_<clip>_<tag>/``) holds
everything a reviewer needs and everything a later comparison needs:

    config.yaml         full resolved config + the overrides that made it
    scorecard.json      metrics (see research/scorecard.py)
    match_data.json     pipeline output (same schema as ground truth)
    tracking_data.json  per-frame ball/players/poses
    keyframes/          rendered frames + contact sheets for review
    REVIEW.md           what to look at, and how to record verdicts

Every run also appends one line to ``runs/leaderboard.jsonl``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from court_vision.config import PipelineConfig, apply_overrides, load_config
from court_vision.export import export_json
from court_vision.research.clips import Clip, get_clip
from court_vision.research.keyframes import contact_sheet, crop_around, render_keyframe
from court_vision.research.labels import load_labels
from court_vision.research.review import write_review_packet
from court_vision.research.runner import RunArtifacts, StageCache, run_clip
from court_vision.research.scorecard import Scorecard, score_run
from court_vision.review_data import load_match_json
from court_vision.serialize import tracking_list_to_json
from court_vision.shot_classify import MatchData, point_winner

DEFAULT_ROOT = Path("runs")


@dataclass
class ExperimentResult:
    directory: Path
    clip: Clip
    scorecard: Scorecard
    artifacts: RunArtifacts
    overrides: dict


def parse_override(text: str) -> tuple[str, object]:
    """``key=value`` with YAML-typed value (``0.4``, ``true``, ``wasb``)."""
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def run_experiment(
    clip_name: str,
    overrides: dict | None = None,
    tag: str | None = None,
    root: Path = DEFAULT_ROOT,
    registry: Path | None = None,
    force: set[str] | None = None,
    keyframes: bool = True,
    config_path: Path | None = None,
    log=print,
) -> ExperimentResult:
    overrides = overrides or {}
    clip = get_clip(clip_name, registry)
    # clip-level facts from the registry (e.g. a known left-hander) apply
    # first; explicit experiment overrides win.
    clip_overrides = dict(clip.extra.get("overrides") or {})
    config = apply_overrides(load_config(config_path), {**clip_overrides, **overrides})
    root = Path(root)
    cache = StageCache(root / "cache")

    t0 = time.time()
    art = run_clip(clip, config, cache=cache, force=force, log=log)
    wall = time.time() - t0

    gt = load_match_json(clip.ground_truth) if clip.has_ground_truth else None
    labels = load_labels(clip.stage_labels) if (clip.stage_labels and clip.stage_labels.exists()) else None
    court_h = {}
    if labels:
        for cl in labels.court:
            for i, seg in enumerate(art.segments):
                if seg.start_frame <= cl.frame <= seg.end_frame and i < len(art.court) and art.court[i].success:
                    court_h[cl.frame] = art.court[i].homography
    sc = score_run(
        clip.name, gt, art.match_data, art.tracking, art.segments,
        [c.success for c in art.court], art.frame_seq.fps, labels=labels, court_homographies=court_h,
    )
    sc.notes.append(f"wall={wall:.1f}s cache_hits={sorted(k for k, v in art.cache_hits.items() if v)}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{clip.name}" + (f"_{tag}" if tag else "")
    exp_dir = root / "experiments" / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "config.yaml").write_text(yaml.safe_dump({
        "clip": clip.name, "tag": tag, "overrides": overrides, "config": config.model_dump(),
    }, sort_keys=False))
    (exp_dir / "scorecard.json").write_text(json.dumps(sc.to_dict(), indent=2))
    export_json(art.match_data, exp_dir / "match_data.json")
    (exp_dir / "tracking_data.json").write_text(json.dumps(tracking_list_to_json(art.tracking)))
    (exp_dir / "segments.json").write_text(json.dumps([s.__dict__ for s in art.segments], indent=2))
    if art.scoreboard is not None:
        (exp_dir / "scoreboard.json").write_text(json.dumps({
            "roi": art.scoreboard.roi,
            "samples": [{"frame": s.frame, "rows": [r.__dict__ for r in s.rows]} for s in art.scoreboard.samples],
        }, indent=1, default=str))

    sheets: list[dict] = []
    if keyframes:
        sheets = render_review_keyframes(exp_dir / "keyframes", art, gt)
    write_review_packet(exp_dir, clip, sc, art.match_data, gt, sheets, overrides)

    row = {
        "dir": str(exp_dir), "clip": clip.name, "tag": tag, "overrides": overrides,
        "score": round(sc.score, 4), "shot_f1": round(sc.shot_f1, 3), "outcome": f"{sc.outcome_correct}/{sc.outcome_total}",
        "points": f"{sc.points_matched}/{sc.gt_points}", "side": f"{sc.side_correct}/{sc.side_total}",
        "stroke_acc": round(sc.stroke_accuracy, 3), "ball_cov": round(sc.ball_coverage, 3),
        "wall_s": round(wall, 1), "time": stamp,
    }
    with open(root / "leaderboard.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    log(f"[experiment] {exp_dir}\n[scorecard] {sc.summary()}")
    return ExperimentResult(directory=exp_dir, clip=clip, scorecard=sc, artifacts=art, overrides=overrides)


def render_review_keyframes(out_dir: Path, art: RunArtifacts, gt: MatchData | None, per_sheet: int = 6) -> list[dict]:
    """Render predicted shots, missed GT shots and point boundaries; group
    them into contact sheets. Returns sheet descriptors for REVIEW.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = art.frame_seq.frames_dir
    fps = art.frame_seq.fps
    by = {t.frame_index: t for t in art.tracking}
    tol = max(1, int(round(0.5 * fps)))

    def seg_h(frame: int):
        for i, seg in enumerate(art.segments):
            if seg.start_frame <= frame <= seg.end_frame and i < len(art.court) and art.court[i].success:
                return art.court[i].homography
        return None

    gt_shots = {s.frame: (p.point_number, s) for p in gt.points for s in p.shots} if gt else {}
    pred_shots = [(p.point_number, s) for p in art.match_data.points for s in p.shots]
    hits_meta = {h["frame"]: h for h in art.match_data.metadata.get("hits", [])}

    items: list[tuple[str, np.ndarray, dict]] = []
    trail = max(10, int(fps * 0.5))
    for pn, s in pred_shots:
        img = cv2.imread(str(frames_dir / f"frame_{s.frame:06d}.jpg"))
        if img is None:
            continue
        near = [(gf, g) for gf, (gp, g) in gt_shots.items() if abs(gf - s.frame) <= tol]
        hm = hits_meta.get(s.frame, {})
        if near:
            gf, g = min(near, key=lambda x: abs(x[0] - s.frame))
            verdict = "OK" if (g.stroke == s.stroke and g.player == s.player) else "MISMATCH"
            gt_txt = f"GT f{gf} {g.player} {g.stroke}"
        else:
            verdict, gt_txt = "FP?", "GT: none within 0.5s"
        cap = (f"PRED P{pn} #{s.shot_number} f{s.frame} t={s.time_s:.2f}s {s.player} {s.stroke} c={s.confidence:.1f}"
               f" [{hm.get('kind', '?')}]\n{gt_txt} -> {verdict}")
        full = render_keyframe(img, by, s.frame, homography=seg_h(s.frame), caption=cap, trail=trail)
        t = by.get(s.frame)
        center = None
        if t and t.ball:
            center = (t.ball.x, t.ball.y)
        else:
            pl = next((p for p in (t.players if t else []) if p.role == s.player), None)
            if pl:
                center = ((pl.bbox[0] + pl.bbox[2]) / 2, (pl.bbox[1] + pl.bbox[3]) / 2)
        zoom = crop_around(full, center, size=360, zoom=1.6) if center else full
        items.append((f"shot_f{s.frame:06d}", full, {"kind": "pred", "frame": s.frame, "point": pn,
                                                      "pred": f"{s.player} {s.stroke}", "gt": gt_txt, "verdict": verdict}))
        cv2.imwrite(str(out_dir / f"shot_f{s.frame:06d}_zoom.jpg"), zoom)

    for gf, (gp, g) in sorted(gt_shots.items()):
        if any(abs(s.frame - gf) <= tol for _, s in pred_shots):
            continue
        img = cv2.imread(str(frames_dir / f"frame_{gf:06d}.jpg"))
        if img is None:
            continue
        cap = f"GT MISSED P{gp} f{gf} t={gf / fps:.2f}s {g.player} {g.stroke}\n(no predicted shot within 0.5s)"
        full = render_keyframe(img, by, gf, homography=seg_h(gf), caption=cap, trail=trail)
        items.append((f"missed_f{gf:06d}", full, {"kind": "missed", "frame": gf, "point": gp, "gt": f"{g.player} {g.stroke}"}))

    for p in art.match_data.points:
        for tag, f in (("start", p.start_frame), ("end", p.end_frame)):
            img = cv2.imread(str(frames_dir / f"frame_{f:06d}.jpg"))
            if img is None:
                continue
            cap = (f"POINT {p.point_number} {tag} f{f} t={f / fps:.2f}s server={p.server}"
                   f" winner={p.winner} ({p.outcome_source}) shots={len(p.shots)}")
            full = render_keyframe(img, by, f, homography=seg_h(f), caption=cap, trail=trail)
            items.append((f"point{p.point_number}_{tag}_f{f:06d}", full, {"kind": f"point-{tag}", "frame": f, "point": p.point_number}))

    sheets: list[dict] = []
    for name, img, _ in items:
        cv2.imwrite(str(out_dir / f"{name}.jpg"), img)
    for i in range(0, len(items), per_sheet):
        chunk = items[i:i + per_sheet]
        sheet = contact_sheet([im for _, im, _ in chunk], cols=2, thumb_w=640)
        sheet_name = f"sheet_{i // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(out_dir / sheet_name), sheet)
        sheets.append({"file": f"keyframes/{sheet_name}", "items": [m for _, _, m in chunk],
                       "names": [n for n, _, _ in chunk]})
    return sheets


def load_leaderboard(root: Path = DEFAULT_ROOT) -> list[dict]:
    p = Path(root) / "leaderboard.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
