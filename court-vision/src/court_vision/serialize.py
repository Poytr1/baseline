"""JSON (de)serialization for pipeline dataclasses.

Single home for the dict <-> dataclass conversions that used to be duplicated
across the review UI, the tuner and the CLI. Everything here is plain JSON so
stage outputs can be cached on disk and diffed between experiments.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from court_vision.ball_tracker import BallDetection
from court_vision.court_detect import CourtDetectionResult
from court_vision.player_detect import FrameTrackingResult, PlayerDetection, PoseKeypoints
from court_vision.scene_filter import GameplaySegment


# ── Ball ─────────────────────────────────────────────────────────────────────

def ball_to_dict(ball: BallDetection | None) -> dict | None:
    return None if ball is None else asdict(ball)


def ball_from_dict(d: dict | None) -> BallDetection | None:
    if not d:
        return None
    return BallDetection(
        frame_index=int(d["frame_index"]),
        x=float(d["x"]),
        y=float(d["y"]),
        confidence=float(d["confidence"]),
        interpolated=bool(d.get("interpolated", False)),
    )


# ── Players / poses ──────────────────────────────────────────────────────────

def player_from_dict(d: dict) -> PlayerDetection:
    cp = d.get("court_position")
    return PlayerDetection(
        frame_index=int(d["frame_index"]),
        bbox=tuple(float(v) for v in d["bbox"]),
        confidence=float(d["confidence"]),
        court_position=tuple(float(v) for v in cp) if cp else None,
        role=d.get("role"),
    )


def pose_from_dict(d: dict) -> PoseKeypoints:
    return PoseKeypoints(
        frame_index=int(d["frame_index"]),
        role=str(d["role"]),
        keypoints={k: tuple(float(x) for x in v) for k, v in d["keypoints"].items()},
    )


def tracking_to_dict(t: FrameTrackingResult) -> dict:
    return {
        "frame_index": t.frame_index,
        "ball": ball_to_dict(t.ball),
        "players": [asdict(p) for p in t.players],
        "poses": [asdict(p) for p in t.poses],
    }


def tracking_from_dict(d: dict) -> FrameTrackingResult:
    return FrameTrackingResult(
        frame_index=int(d["frame_index"]),
        ball=ball_from_dict(d.get("ball")),
        players=[player_from_dict(p) for p in d.get("players", [])],
        poses=[pose_from_dict(p) for p in d.get("poses", [])],
    )


def tracking_list_to_json(results: list[FrameTrackingResult]) -> list[dict]:
    return [tracking_to_dict(t) for t in results]


def tracking_list_from_json(raw: list[dict]) -> list[FrameTrackingResult]:
    return sorted((tracking_from_dict(d) for d in raw), key=lambda t: t.frame_index)


# ── Segments ─────────────────────────────────────────────────────────────────

def segment_to_dict(s: GameplaySegment) -> dict:
    return asdict(s)


def segment_from_dict(d: dict) -> GameplaySegment:
    return GameplaySegment(
        start_frame=int(d["start_frame"]),
        end_frame=int(d["end_frame"]),
        start_time_s=float(d["start_time_s"]),
        end_time_s=float(d["end_time_s"]),
        frame_count=int(d.get("frame_count", d["end_frame"] - d["start_frame"] + 1)),
    )


# ── Court ────────────────────────────────────────────────────────────────────

def court_to_dict(c: CourtDetectionResult) -> dict:
    return {
        "success": bool(c.success),
        "homography": None if c.homography is None else np.asarray(c.homography).tolist(),
        "pixel_keypoints": None if c.pixel_keypoints is None else [list(map(float, p)) for p in c.pixel_keypoints],
        "num_lines_detected": int(c.num_lines_detected),
    }


def court_from_dict(d: dict) -> CourtDetectionResult:
    H = d.get("homography")
    return CourtDetectionResult(
        success=bool(d["success"]),
        homography=None if H is None else np.asarray(H, dtype=np.float64),
        pixel_keypoints=None if d.get("pixel_keypoints") is None else [tuple(p) for p in d["pixel_keypoints"]],
        num_lines_detected=int(d.get("num_lines_detected", 0)),
    )


def _json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
