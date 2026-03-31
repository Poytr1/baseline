"""Player detection — YOLOv8 person detection, role assignment, and pose estimation."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from court_vision.ball_tracker import BallDetection, BallTrajectory, build_trajectory, reject_stationary_detections
from court_vision.scene_filter import GameplaySegment

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


# MediaPipe landmark names (subset relevant to tennis)
_POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


@lru_cache(maxsize=1)
def _get_pose_model_path() -> str:
    """Download and cache the MediaPipe pose landmarker model.

    Returns:
        Path to the .task model file.
    """
    import urllib.request

    cache_dir = Path.home() / ".cache" / "court-vision" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "pose_landmarker_lite.task"

    if not model_path.exists():
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, str(model_path))

    return str(model_path)


@lru_cache(maxsize=1)
def _get_pose_estimator():
    """Load MediaPipe PoseLandmarker (cached singleton)."""
    import mediapipe as mp

    model_path = _get_pose_model_path()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def estimate_pose(
    frame: np.ndarray,
    player: PlayerDetection,
) -> PoseKeypoints | None:
    """Estimate pose keypoints for a detected player.

    Crops the frame to the player's bounding box and runs MediaPipe Pose.

    Args:
        frame: Full BGR frame.
        player: Player detection with bounding box and role.

    Returns:
        PoseKeypoints with named keypoints, or None if pose not detected.
    """
    import mediapipe as mp

    x1, y1, x2, y2 = player.bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Clamp to frame bounds
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    landmarker = _get_pose_estimator()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
    results = landmarker.detect(mp_image)

    if not results.pose_landmarks:
        return None

    keypoints: dict[str, tuple[float, float, float]] = {}
    crop_h, crop_w = crop.shape[:2]

    for i, landmark in enumerate(results.pose_landmarks[0]):
        if i < len(_POSE_LANDMARK_NAMES):
            name = _POSE_LANDMARK_NAMES[i]
            px = x1 + landmark.x * crop_w
            py = y1 + landmark.y * crop_h
            keypoints[name] = (float(px), float(py), float(landmark.visibility))

    return PoseKeypoints(
        frame_index=player.frame_index,
        role=player.role or "unknown",
        keypoints=keypoints,
    )


def track_segment(
    frames_dir: Path,
    segment: GameplaySegment,
    homography: np.ndarray | None = None,
) -> list[FrameTrackingResult]:
    """Track ball, players, and poses for all frames in a gameplay segment.

    Uses build_trajectory for ball detection (with gap interpolation
    and stationarity rejection) instead of per-frame detection.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        segment: Gameplay segment defining frame range.
        homography: Homography matrix for this segment, or None.

    Returns:
        List of FrameTrackingResult, one per successfully read frame.
    """
    # Build ball trajectory for the whole segment
    trajectory = build_trajectory(
        frames_dir, segment.start_frame, segment.end_frame, fps=30.0,
    )

    # Apply stationarity rejection
    raw_dets: list[BallDetection | None] = [None] * (segment.end_frame - segment.start_frame + 1)
    for det in trajectory.detections:
        idx = det.frame_index - segment.start_frame
        if 0 <= idx < len(raw_dets):
            raw_dets[idx] = det

    filtered_dets = reject_stationary_detections(raw_dets)

    # Build lookup: frame_index -> filtered ball detection
    ball_by_frame: dict[int, BallDetection | None] = {}
    for i, det in enumerate(filtered_dets):
        ball_by_frame[segment.start_frame + i] = det

    results: list[FrameTrackingResult] = []

    for frame_idx in range(segment.start_frame, segment.end_frame + 1):
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Ball from trajectory (already filtered)
        ball = ball_by_frame.get(frame_idx)

        # Player detection + role assignment
        raw_players = detect_players_in_frame(frame, frame_index=frame_idx)
        players = assign_player_roles(raw_players)

        # Map players to court coordinates
        if homography is not None:
            for player in players:
                player.court_position = map_player_to_court(player, homography)

        # Pose estimation per player
        poses: list[PoseKeypoints] = []
        for player in players:
            pose = estimate_pose(frame, player)
            if pose is not None:
                poses.append(pose)

        results.append(FrameTrackingResult(
            frame_index=frame_idx,
            ball=ball,
            players=players,
            poses=poses,
        ))

    return results
