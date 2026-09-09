"""Render an annotated video from an experiment directory.

Reuses the cached per-frame tracking and the match data of a finished
experiment, so no model runs again: court lines, ball trail, player boxes
and pose, plus what the pipeline concluded — each contact flashes its
stroke label, a rally strip lists the point so far, a banner shows the
point winner and where it came from (scoreboard / landing / unknown), and
a top-down court in the bottom-right corner shows the players and, for
the point so far, each shot's contact and landing — the two ground points
the shot speed is measured between.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
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


# court geometry in metres (ITF): half length, singles / doubles half width, service line
_BASE, _SINGLES, _DOUBLES, _SERVICE = 11.885, 4.115, 5.485, 6.4


def draw_minimap(
    img,
    point,
    frame: int,
    by_frame: dict,
    scale_px_m: float = 9.0,
    margin: int = 16,
    pad: int = 26,
):
    """Top-down court in the bottom-right corner (near player at the bottom,
    like the camera): the players' feet now (ringed dots), and for the point
    so far where each shot was hit from (numbered dot, matching the rally
    strip) and where it first bounced (cross), in the hitter's colour. The
    latest shot's two points are joined and labelled with its speed — that
    chord and flight time are what the speed is averaged over. A bounce
    appears once it has happened; the airborne ball is not drawn because its
    ground projection is metres off.
    """
    h, w = img.shape[:2]
    cw, ch = int(round(2 * _DOUBLES * scale_px_m)), int(round(2 * _BASE * scale_px_m))
    pw, ph = cw + 2 * pad, ch + 2 * pad
    x0, y0 = w - margin - pw, h - margin - ph
    if x0 < 0 or y0 < 0:
        return img
    roi = img[y0:y0 + ph, x0:x0 + pw]
    panel = roi.copy()
    panel[:] = (24, 24, 24)
    cv2.addWeighted(panel, 0.8, roi, 0.2, 0, roi)

    def px(x: float, y: float) -> tuple[int, int]:
        x = max(-_DOUBLES - pad / scale_px_m, min(_DOUBLES + pad / scale_px_m, x))
        y = max(-_BASE - pad / scale_px_m, min(_BASE + pad / scale_px_m, y))
        return int(round(x0 + pad + (x + _DOUBLES) * scale_px_m)), int(round(y0 + pad + (_BASE - y) * scale_px_m))

    line = (235, 235, 235)
    for (ax, ay), (bx, by) in (
        ((-_DOUBLES, -_BASE), (_DOUBLES, -_BASE)), ((-_DOUBLES, _BASE), (_DOUBLES, _BASE)),
        ((-_DOUBLES, -_BASE), (-_DOUBLES, _BASE)), ((_DOUBLES, -_BASE), (_DOUBLES, _BASE)),
        ((-_SINGLES, -_BASE), (-_SINGLES, _BASE)), ((_SINGLES, -_BASE), (_SINGLES, _BASE)),
        ((-_SINGLES, -_SERVICE), (_SINGLES, -_SERVICE)), ((-_SINGLES, _SERVICE), (_SINGLES, _SERVICE)),
        ((0.0, -_SERVICE), (0.0, _SERVICE)),
    ):
        cv2.line(img, px(ax, ay), px(bx, by), line, 1, cv2.LINE_AA)
    cv2.line(img, px(-_DOUBLES - 0.9, 0.0), px(_DOUBLES + 0.9, 0.0), (80, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "top view", (x0 + 6, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    t = by_frame.get(frame)
    for p in (t.players if t is not None else []):
        if p.court_position is not None:
            color = _ROLE_COLOR.get(p.role, (200, 200, 200))
            cv2.circle(img, px(*p.court_position), 6, color, -1, cv2.LINE_AA)
            cv2.circle(img, px(*p.court_position), 7, (255, 255, 255), 1, cv2.LINE_AA)

    if point is None:
        return img
    done = [s for s in point.shots if s.frame <= frame]
    landed = [s for s in done if s.placement is not None and (s.bounce_frame is None or s.bounce_frame <= frame)]
    latest = landed[-1] if landed else None
    for s in done:
        color = _ROLE_COLOR.get(s.player, (200, 200, 200))
        if s.contact is not None:
            cx, cy = px(*s.contact)
            cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 4, (24, 24, 24), 1, cv2.LINE_AA)
            dx = 6 if s.contact[0] >= 0 else -14
            cv2.putText(img, str(s.shot_number), (cx + dx, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        if s in landed:
            cx, cy = px(s.placement.x, s.placement.y)
            r = 6 if s is latest else 4
            cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), color, 2, cv2.LINE_AA)
            cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), color, 2, cv2.LINE_AA)
    if latest is not None:
        color = _ROLE_COLOR.get(latest.player, (200, 200, 200))
        lx, ly = px(latest.placement.x, latest.placement.y)
        if latest.contact is not None:
            cv2.line(img, px(*latest.contact), (lx, ly), color, 1, cv2.LINE_AA)
        if latest.speed_kmh:
            label = f"{latest.speed_kmh:.0f} km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = min(max(x0 + 4, lx - tw // 2), x0 + pw - tw - 4)
            ty = ly - 12 if ly - 12 - th > y0 + 4 else ly + 12 + th
            cv2.rectangle(img, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (24, 24, 24), -1)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


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
        draw_minimap(out, point, f, by_frame)
        y = 30
        if point is not None:
            winner = point.winner or "unknown"
            src = point.outcome_source or ""
            server = _ROLE_SHORT.get(point.server or "", "?")
            y += _text(out, f"Point {point.point_number}   server: {server}   t={f / fps:5.1f}s", (12, y), 0.7)
            done = [s for s in point.shots if s.frame <= f]
            if done:
                tokens = [
                    f"{s.shot_number}.{_ROLE_SHORT[s.player][0].upper()} {s.stroke}"
                    + (f" {s.speed_kmh:.0f}" if s.speed_kmh else "")
                    for s in done
                ]
                # wrap the rally strip so a long rally never runs off the frame
                lines, cur = [], ""
                for tok in tokens:
                    cand = f"{cur}  {tok}" if cur else tok
                    if cur and cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] > w - 24:
                        lines.append(cur)
                        cur = tok
                    else:
                        cur = cand
                lines.append(cur)
                for line in lines:
                    y += _text(out, line, (12, y), 0.6, (255, 255, 0))
            # flash the current contact
            for s in point.shots:
                if 0 <= f - s.frame <= flash:
                    t = by_frame.get(s.frame)
                    pl = next((p for p in (t.players if t else []) if p.role == s.player), None)
                    color = _ROLE_COLOR[s.player]
                    if pl is not None:
                        x1, y1, x2, y2 = (int(v) for v in pl.bbox)
                        cv2.rectangle(out, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 3)
                        label = f"{s.stroke.upper()} ({_ROLE_SHORT[s.player]})" + (f"  {s.speed_kmh:.0f} km/h" if s.speed_kmh else "")
                        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
                        _text(out, label, (min(max(12, x1), w - tw - 12), max(30, y1 - 12)), 0.8, color)
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
