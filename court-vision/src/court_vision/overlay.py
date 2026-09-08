"""Overlay — render visual annotations on video frames."""

import cv2
import numpy as np

from court_vision.ball_tracker import BallDetection
from court_vision.court_detect import COURT_KEYPOINTS
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
COURT_LINE_COLOR = (255, 255, 255)  # White

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


BALL_OVERLAY_MIN_CONFIDENCE = 0.75
BALL_WEAK_NEIGHBOR_WINDOW = 5


def _is_strong_ball(ball: BallDetection | None) -> bool:
    return (
        ball is not None
        and not ball.interpolated
        and ball.confidence >= BALL_OVERLAY_MIN_CONFIDENCE
    )


def compute_strong_ball_frames(
    tracking_results: list[FrameTrackingResult],
) -> set[int]:
    """Frame indices with a strong ball detection (non-interpolated, high-conf)."""
    return {t.frame_index for t in tracking_results if _is_strong_ball(t.ball)}


def draw_ball(frame: np.ndarray, ball: BallDetection) -> np.ndarray:
    """Draw a circle at the ball position.

    High-confidence real detections (≥ BALL_OVERLAY_MIN_CONFIDENCE, not
    interpolated) render as a bold yellow filled circle. Lower-confidence
    or interpolated detections render as a smaller, hollow, dimmer circle
    so the viewer can see what the tracker knows without the "phantom
    arc" look of an equally-bold overlay everywhere.

    Args:
        frame: BGR image (H, W, 3).
        ball: Ball detection with x, y coordinates.

    Returns:
        Copy of frame with ball overlay drawn.
    """
    out = frame.copy()
    center = (int(ball.x), int(ball.y))
    strong = (not ball.interpolated) and ball.confidence >= BALL_OVERLAY_MIN_CONFIDENCE
    if strong:
        cv2.circle(out, center, 8, BALL_COLOR, -1)
        cv2.circle(out, center, 10, BALL_COLOR, 2)
    else:
        # Interpolated or low-confidence: magenta hollow ring — visible
        # against court green/blue without looking like a confirmed detection.
        weak_color = (255, 0, 255)  # magenta (BGR)
        cv2.circle(out, center, 10, weak_color, 2)
        cv2.circle(out, center, 3, weak_color, -1)
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


# Court line connections as pairs of COURT_KEYPOINTS names.
COURT_LINE_CONNECTIONS = [
    # Near baseline
    ("baseline_near_left_doubles", "baseline_near_right_doubles"),
    # Far baseline
    ("baseline_far_left_doubles", "baseline_far_right_doubles"),
    # Left singles sideline
    ("baseline_near_left_singles", "baseline_far_left_singles"),
    # Right singles sideline
    ("baseline_near_right_singles", "baseline_far_right_singles"),
    # Left doubles sideline
    ("baseline_near_left_doubles", "baseline_far_left_doubles"),
    # Right doubles sideline
    ("baseline_near_right_doubles", "baseline_far_right_doubles"),
    # Near service line
    ("service_near_left", "service_near_right"),
    # Far service line
    ("service_far_left", "service_far_right"),
    # Center service line
    ("service_near_center", "service_far_center"),
    # Net
    ("net_left_doubles", "net_right_doubles"),
]


def draw_court(frame: np.ndarray, homography: np.ndarray | None) -> np.ndarray:
    """Draw court lines projected onto the frame.

    Args:
        frame: BGR image (H, W, 3).
        homography: 3x3 pixel-to-court homography, or None.

    Returns:
        Copy of frame with court lines drawn.
    """
    out = frame.copy()
    if homography is None:
        return out

    # Inverse homography: court (meters) -> pixel coordinates
    H_inv = np.linalg.inv(homography)

    # Project all court keypoints to pixel space
    pixel_pts: dict[str, tuple[int, int]] = {}
    for name, (cx, cy) in COURT_KEYPOINTS.items():
        court_h = np.array([cx, cy, 1.0], dtype=np.float64)
        px_h = H_inv @ court_h
        w = px_h[2]
        if abs(w) < 1e-10:
            continue
        pixel_pts[name] = (int(px_h[0] / w), int(px_h[1] / w))

    # Draw each court line
    for name_a, name_b in COURT_LINE_CONNECTIONS:
        if name_a in pixel_pts and name_b in pixel_pts:
            cv2.line(out, pixel_pts[name_a], pixel_pts[name_b],
                     COURT_LINE_COLOR, 2)

    return out


def render_overlay(
    frame: np.ndarray,
    tracking: FrameTrackingResult,
    homography: np.ndarray | None = None,
    strong_ball_frames: set[int] | None = None,
) -> np.ndarray:
    """Render all overlays for a single frame.

    Combines court, ball, player, and pose overlays.

    Args:
        frame: BGR image (H, W, 3).
        tracking: Per-frame tracking data.
        homography: Court homography (pixel -> meters).
        strong_ball_frames: Set of frame indices that have a high-
            confidence real ball detection. When provided, weak or
            interpolated detections are only rendered if there is a
            strong detection within BALL_WEAK_NEIGHBOR_WINDOW frames —
            this suppresses phantom detections between points while
            keeping uncertain-but-real mid-rally / impact detections.
            When None, all detections render.
    """
    out = frame.copy()
    out = draw_court(out, homography)
    ball = tracking.ball
    if ball is not None:
        if _is_strong_ball(ball) or strong_ball_frames is None:
            out = draw_ball(out, ball)
        else:
            idx = tracking.frame_index
            has_strong_neighbor = any(
                (idx + k) in strong_ball_frames
                for k in range(-BALL_WEAK_NEIGHBOR_WINDOW, BALL_WEAK_NEIGHBOR_WINDOW + 1)
                if k != 0
            )
            if has_strong_neighbor:
                out = draw_ball(out, ball)
    if tracking.players:
        out = draw_players(out, tracking.players)
    if tracking.poses:
        out = draw_poses(out, tracking.poses)
    return out
