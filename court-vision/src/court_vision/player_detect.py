"""Player detection — YOLOv8 person detection, role assignment, and pose estimation."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from court_vision.ball_tracker import BallDetection

_PERSON_CLASS_ID = 0


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


@lru_cache(maxsize=1)
def _get_yolo_model():
    """Load the YOLOv8n model (cached singleton)."""
    from ultralytics import YOLO

    return YOLO("yolov8n.pt")


def detect_players_in_frame(
    frame: np.ndarray,
    frame_index: int = 0,
    confidence_threshold: float = 0.5,
) -> list[PlayerDetection]:
    """Detect players (persons) in a single frame using YOLOv8.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        frame_index: Index of this frame in the video sequence.
        confidence_threshold: Minimum confidence to keep a detection.

    Returns:
        List of PlayerDetection for detected persons.
    """
    model = _get_yolo_model()
    results = model(frame, verbose=False)

    detections: list[PlayerDetection] = []

    for result in results:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            if int(classes[i]) != _PERSON_CLASS_ID:
                continue
            if confs[i] < confidence_threshold:
                continue

            x1, y1, x2, y2 = xyxy[i]
            detections.append(PlayerDetection(
                frame_index=frame_index,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(confs[i]),
            ))

    return detections


def assign_player_roles(
    players: list[PlayerDetection],
) -> list[PlayerDetection]:
    """Assign near_player/far_player roles based on vertical position.

    The player closer to the bottom of the frame (larger y2) is the
    near player. The player closer to the top (smaller y2) is the far player.

    If more than 2 players are detected, keeps the 2 with highest confidence.

    Args:
        players: List of detected players in a single frame.

    Returns:
        List of up to 2 PlayerDetections with role assigned.
    """
    if not players:
        return []

    sorted_by_conf = sorted(players, key=lambda p: p.confidence, reverse=True)
    top_players = sorted_by_conf[:2]

    if len(top_players) == 1:
        top_players[0].role = "near_player"
        return top_players

    sorted_by_y = sorted(top_players, key=lambda p: p.bbox[3], reverse=True)
    sorted_by_y[0].role = "near_player"
    sorted_by_y[1].role = "far_player"

    return sorted_by_y


def map_player_to_court(
    player: PlayerDetection,
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    """Map a player's feet position to court coordinates.

    Uses center-bottom of bounding box as feet approximation.

    Args:
        player: Player detection with bounding box.
        homography: 3x3 homography matrix, or None if unavailable.

    Returns:
        (x, y) court coordinates in meters, or None if homography is None.
    """
    if homography is None:
        return None

    x = (player.bbox[0] + player.bbox[2]) / 2
    y = player.bbox[3]

    pixel = np.array([x, y, 1.0], dtype=np.float64)
    transformed = homography @ pixel
    w = transformed[2]
    if abs(w) < 1e-10:
        return (0.0, 0.0)

    return (float(transformed[0] / w), float(transformed[1] / w))
