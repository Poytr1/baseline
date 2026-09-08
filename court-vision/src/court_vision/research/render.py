"""Render an annotated video from an experiment directory.

Reuses the cached per-frame tracking and the match data of a finished
experiment, so no model runs again: court lines, ball trail, player boxes
and pose, plus what the pipeline concluded — each contact flashes its
stroke label, a rally strip lists the point so far, and a banner shows
the point winner and where it came from (scoreboard / landing / unknown).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from court_vision.court_detect import compute_segment_homographies
from court_vision.research.clips import get_clip
from court_vision.research.keyframes import render_keyframe
from court_vision.review_data import load_match_json
from court_vision.serialize import segment_from_dict, tracking_list_from_json
from court_vision.shot_classify import MatchData

_ROLE_SHORT = {"near_player": "near", "far_player": "far"}
_ROLE_COLOR = {"near_player": (0, 255, 0), "far_player": (0, 0, 255)}


def _text(img, text, org, scale=0.7, color=(255, 255, 255), thickness=2, bg=(0, 0, 0)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + 8), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return th + 16


def render_experiment_video(
    exp_dir: Path,
    output: Path | None = None,
    registry: Path | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    gameplay_only: bool = True,
    log=print,
) -> Path:
    exp_dir = Path(exp_dir)
    cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())
    clip = get_clip(cfg["clip"], registry)
    frames_dir = clip.frames_dir or (clip.video.parent / "frames")
    match: MatchData = load_match_json(exp_dir / "match_data.json")
    fps = float(match.metadata.get("fps", 30.0))
    tracking = tracking_list_from_json(json.loads((exp_dir / "tracking_data.json").read_text()))
    by_frame = {t.frame_index: t for t in tracking}
    segments = [segment_from_dict(d) for d in json.loads((exp_dir / "segments.json").read_text())]
    court = compute_segment_homographies(frames_dir, segments, method=cfg["config"]["pipeline"].get("court_method", "auto"))

    def seg_h(frame: int):
        for i, seg in enumerate(segments):
            if seg.start_frame <= frame <= seg.end_frame and i < len(court) and court[i].success:
                return court[i].homography
        return None

    total = max(t.frame_index for t in tracking) + 1 if tracking else 0
    first = start_frame if start_frame is not None else 0
    last = end_frame if end_frame is not None else total - 1
    frames = list(range(first, last + 1))
    if gameplay_only:
        in_play = set()
        for seg in segments:
            in_play.update(range(seg.start_frame, seg.end_frame + 1))
        frames = [f for f in frames if f in in_play]
    if not frames:
        raise ValueError("no frames to render")

    sample = cv2.imread(str(frames_dir / f"frame_{frames[0]:06d}.jpg"))
    h, w = sample.shape[:2]
    if output is None:
        output = exp_dir / "annotated.mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    flash = int(fps * 0.8)
    trail = max(10, int(fps * 0.5))
    points = match.points
    for n, f in enumerate(frames):
        img = cv2.imread(str(frames_dir / f"frame_{f:06d}.jpg"))
        if img is None:
            continue
        out = render_keyframe(img, by_frame, f, homography=seg_h(f), trail=trail)
        point = next((p for p in points if p.start_frame <= f <= p.end_frame), None)
        y = 30
        if point is not None:
            winner = point.winner or "unknown"
            src = point.outcome_source or ""
            server = _ROLE_SHORT.get(point.server or "", "?")
            y += _text(out, f"Point {point.point_number}   server: {server}   t={f / fps:5.1f}s", (12, y), 0.7)
            done = [s for s in point.shots if s.frame <= f]
            if done:
                strip = "  ".join(f"{s.shot_number}.{_ROLE_SHORT[s.player][0].upper()} {s.stroke}" for s in done)
                y += _text(out, strip, (12, y), 0.6, (255, 255, 0))
            # flash the current contact
            for s in point.shots:
                if 0 <= f - s.frame <= flash:
                    t = by_frame.get(s.frame)
                    pl = next((p for p in (t.players if t else []) if p.role == s.player), None)
                    color = _ROLE_COLOR[s.player]
                    if pl is not None:
                        x1, y1, x2, y2 = (int(v) for v in pl.bbox)
                        cv2.rectangle(out, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 3)
                        _text(out, f"{s.stroke.upper()} ({_ROLE_SHORT[s.player]})", (max(12, x1), max(30, y1 - 12)), 0.8, color)
                    if t is not None and t.ball is not None:
                        cv2.circle(out, (int(t.ball.x), int(t.ball.y)), 18, color, 3)
            # winner banner after the last shot / at the end of the point
            last_shot = point.shots[-1].frame if point.shots else point.start_frame
            if f >= last_shot + int(fps * 1.0) or f >= point.end_frame - int(fps * 1.5):
                label = (f"Point {point.point_number}: {_ROLE_SHORT.get(winner, winner)} wins  [{src}]"
                         if point.winner else f"Point {point.point_number}: winner unknown  [{src}]")
                col = _ROLE_COLOR.get(winner, (200, 200, 200))
                _text(out, label, (12, h - 24), 0.9, col)
        writer.write(out)
        if n % 300 == 0:
            log(f"[render] {n}/{len(frames)} frames")
    writer.release()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    "-movflags", "+faststart", str(output)], capture_output=True)
    Path(tmp_path).unlink(missing_ok=True)
    log(f"[render] wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return output
