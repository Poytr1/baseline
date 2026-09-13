"""Per-frame tracking data types shared by the perception and analysis layers.

This module has no model or OpenCV dependency on purpose: everything the
analysis layer (trajectory cleaning, hit detection, stroke classification,
points) consumes is one of these records, serialised as plain JSON
(``serialize.py``). A port to another language reproduces these four types
and the JSON, and the analysis functions follow.
"""

from __future__ import annotations

from dataclasses import dataclass

from court_vision.trajectory import BallDetection


@dataclass
class PlayerDetection:
    """Single-frame player detection result."""

    frame_index: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixel coords
    confidence: float
    court_position: tuple[float, float] | None = None  # (x, y) meters
    role: str | None = None  # "near_player" or "far_player"


@dataclass
class PoseKeypoints:
    """Pose estimation result for a single player in a single frame."""

    frame_index: int
    role: str  # "near_player" or "far_player"
    keypoints: dict[str, tuple[float, float, float]]  # name -> (x, y, visibility)


@dataclass
class FrameTrackingResult:
    """Combined tracking result for a single frame."""

    frame_index: int
    ball: BallDetection | None
    players: list[PlayerDetection]
    poses: list[PoseKeypoints]


__all__ = ["BallDetection", "PlayerDetection", "PoseKeypoints", "FrameTrackingResult"]
