"""Keyframe rendering for review — overlays with ball trail, hits and GT.

The harness samples the frames that matter (predicted hits, ground-truth
shots that were missed, point boundaries, a few evenly spaced frames) and
renders them with everything the pipeline believed, so a reviewer (Claude
Code or a human) can judge each stage from a handful of images instead of a
whole video.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from court_vision.overlay import draw_court, draw_players, draw_poses
from court_vision.player_detect import FrameTrackingResult

TRAIL_COLOR = (0, 200, 255)
HIT_COLOR = (0, 0, 255)
GT_COLOR = (255, 0, 255)


@dataclass
class Keyframe:
    frame: int
    tag: str  # e.g. "hit:2", "gt-miss", "fp", "point-start"
    caption: str
    path: Path | None = None


def draw_trail(
    frame: np.ndarray,
    tracking_by_frame: dict[int, FrameTrackingResult],
    frame_index: int,
    trail: int = 15,
) -> np.ndarray:
    """Draw the ball trail over the previous ``trail`` frames (fading) and the
    current ball position (solid). Interpolated positions are hollow."""
    out = frame.copy()
    pts = []
    for k in range(trail, -1, -1):
        t = tracking_by_frame.get(frame_index - k)
        if t is None or t.ball is None:
            pts.append(None)
            continue
        pts.append((int(t.ball.x), int(t.ball.y), t.ball.interpolated, t.ball.confidence))
    prev = None
    for i, p in enumerate(pts):
        if p is None:
            prev = None
            continue
        alpha = (i + 1) / len(pts)
        color = tuple(int(c * (0.35 + 0.65 * alpha)) for c in TRAIL_COLOR)
        if prev is not None:
            cv2.line(out, prev[:2], p[:2], color, 2)
        r = 6 if i == len(pts) - 1 else 3
        if p[2]:
            cv2.circle(out, p[:2], r, color, 1)
        else:
            cv2.circle(out, p[:2], r, color, -1)
        prev = p
    cur = pts[-1]
    if cur is not None:
        cv2.circle(out, cur[:2], 12, (0, 255, 255), 2)
    return out


def render_keyframe(
    frame: np.ndarray,
    tracking_by_frame: dict[int, FrameTrackingResult],
    frame_index: int,
    homography: np.ndarray | None = None,
    caption: str = "",
    gt_marker: tuple[float, float] | None = None,
    trail: int = 15,
) -> np.ndarray:
    out = draw_court(frame, homography)
    t = tracking_by_frame.get(frame_index)
    if t is not None:
        out = draw_players(out, t.players)
        out = draw_poses(out, t.poses)
    out = draw_trail(out, tracking_by_frame, frame_index, trail=trail)
    if gt_marker is not None:
        cv2.drawMarker(out, (int(gt_marker[0]), int(gt_marker[1])), GT_COLOR, cv2.MARKER_CROSS, 30, 2)
    if caption:
        _put_caption(out, caption)
    return out


def _put_caption(img: np.ndarray, text: str) -> None:
    lines = text.split("\n")
    y = 28
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (8, y - th - 6), (14 + tw, y + 6), (0, 0, 0), -1)
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += th + 14


def contact_sheet(
    images: list[np.ndarray],
    cols: int = 2,
    thumb_w: int = 640,
) -> np.ndarray:
    """Tile images into a grid so several keyframes fit in one review image."""
    if not images:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    thumbs = []
    for im in images:
        h, w = im.shape[:2]
        s = thumb_w / w
        thumbs.append(cv2.resize(im, (thumb_w, int(h * s))))
    th = max(t.shape[0] for t in thumbs)
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        row = [np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0))) for t in row]
        while len(row) < cols:
            row.append(np.zeros((th, thumb_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def crop_around(
    img: np.ndarray,
    center: tuple[float, float],
    size: int = 360,
    zoom: float = 1.5,
) -> np.ndarray:
    """Zoomed crop around a point (e.g. a hit) so small far-court detail is legible."""
    h, w = img.shape[:2]
    half = size // 2
    cx, cy = int(center[0]), int(center[1])
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return img
    return cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
