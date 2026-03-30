"""Overlay — render visual annotations on video frames."""

import cv2
import numpy as np

from court_vision.ball_tracker import BallDetection
from court_vision.player_detect import (
    FrameTrackingResult,
    PlayerDetection,
    PoseKeypoints,
)

# Colors (BGR)
BALL_COLOR = (0, 255, 255)       # Yellow
NEAR_PLAYER_COLOR = (0, 255, 0)  # Green
FAR_PLAYER_COLOR = (0, 0, 255)   # Red
POSE_COLOR = (255, 255, 0)       # Cyan
POSE_LINK_COLOR = (200, 200, 0)  # Light cyan

# Pose skeleton connections (pairs of keypoint names)
SKELETON_LINKS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def draw_ball(frame: np.ndarray, ball: BallDetection) -> np.ndarray:
    """Draw a circle at the ball position.

    Args:
        frame: BGR image (H, W, 3).
        ball: Ball detection with x, y coordinates.

    Returns:
        Copy of frame with ball overlay drawn.
    """
    out = frame.copy()
    center = (int(ball.x), int(ball.y))
    cv2.circle(out, center, 8, BALL_COLOR, -1)
    cv2.circle(out, center, 10, BALL_COLOR, 2)
    return out


def draw_players(frame: np.ndarray, players: list[PlayerDetection]) -> np.ndarray:
    """Draw bounding boxes with role labels.

    Args:
        frame: BGR image (H, W, 3).
        players: List of player detections.

    Returns:
        Copy of frame with player overlays drawn.
    """
    out = frame.copy()
    for player in players:
        color = NEAR_PLAYER_COLOR if player.role == "near_player" else FAR_PLAYER_COLOR
        x1, y1, x2, y2 = (int(v) for v in player.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = player.role or "unknown"
        cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def draw_poses(frame: np.ndarray, poses: list[PoseKeypoints]) -> np.ndarray:
    """Draw pose keypoints and skeleton links.

    Args:
        frame: BGR image (H, W, 3).
        poses: List of pose keypoint sets.

    Returns:
        Copy of frame with pose overlays drawn.
    """
    out = frame.copy()
    for pose in poses:
        kp = pose.keypoints
        # Draw keypoint dots
        for name, (x, y, vis) in kp.items():
            if vis > 0.3:
                cv2.circle(out, (int(x), int(y)), 4, POSE_COLOR, -1)

        # Draw skeleton links
        for name_a, name_b in SKELETON_LINKS:
            if name_a in kp and name_b in kp:
                a = kp[name_a]
                b = kp[name_b]
                if a[2] > 0.3 and b[2] > 0.3:
                    cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                             POSE_LINK_COLOR, 1)
    return out


def render_overlay(frame: np.ndarray, tracking: FrameTrackingResult) -> np.ndarray:
    """Render all overlays for a single frame.

    Combines ball, player, and pose overlays.

    Args:
        frame: BGR image (H, W, 3).
        tracking: Per-frame tracking data.

    Returns:
        Copy of frame with all overlays drawn.
    """
    out = frame.copy()
    if tracking.ball is not None:
        out = draw_ball(out, tracking.ball)
    if tracking.players:
        out = draw_players(out, tracking.players)
    if tracking.poses:
        out = draw_poses(out, tracking.poses)
    return out
