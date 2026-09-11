"""Player detection — YOLO-pose person detection, court-aware role tracking.

One YOLO-pose pass gives both the bounding box and 17 COCO keypoints per
person, so no separate pose model is needed. The far player is tiny in a
broadcast frame (~50px tall at 720p), so when a court homography is
available the far half of the court is additionally cropped and upscaled
before detection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from court_vision.ball_tracker import BallDetection, build_trajectory
from court_vision.trajectory import finalize_ball_by_frame  # noqa: F401  (re-exported)
from court_vision.scene_filter import GameplaySegment

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

_CACHE_DIR = Path.home() / ".cache" / "court-vision" / "models"
_NEAR_BASELINE_Y = -11.885
_FAR_BASELINE_Y = 11.885
_DOUBLES_HALF = 5.485


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


@dataclass
class PersonCandidate:
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: dict[str, tuple[float, float, float]]
    court_position: tuple[float, float] | None = None


# ── Model ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _get_pose_model(model_name: str = "yolov8s-pose.pt"):
    """Load a YOLO-pose model (cached singleton). Weights are downloaded by
    ultralytics into ~/.cache/court-vision/models on first use."""
    from ultralytics import YOLO

    from court_vision.device import get_device

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(model_name)
    if not path.is_absolute() and not path.exists():
        path = _CACHE_DIR / model_name
        if not path.exists():
            # Let ultralytics download into the cache dir by resolving cwd there.
            import os
            cwd = os.getcwd()
            try:
                os.chdir(_CACHE_DIR)
                YOLO(model_name)  # triggers download
            finally:
                os.chdir(cwd)
    model = YOLO(str(path))
    model.to(get_device())
    return model


def detect_persons(
    frame: np.ndarray,
    model_name: str = "yolov8s-pose.pt",
    imgsz: int = 1280,
    conf: float = 0.15,
) -> list[PersonCandidate]:
    """Run YOLO-pose on a frame; returns every person with bbox + keypoints."""
    model = _get_pose_model(model_name)
    results = model(frame, verbose=False, conf=conf, imgsz=imgsz, classes=[0])
    out: list[PersonCandidate] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
        kconf = (
            r.keypoints.conf.cpu().numpy()
            if r.keypoints is not None and r.keypoints.conf is not None else None
        )
        for i in range(len(xyxy)):
            kps: dict[str, tuple[float, float, float]] = {}
            if kxy is not None:
                for j, name in enumerate(COCO_KEYPOINT_NAMES):
                    if j < kxy.shape[1]:
                        c = float(kconf[i][j]) if kconf is not None else 1.0
                        kps[name] = (float(kxy[i][j][0]), float(kxy[i][j][1]), c)
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            out.append(PersonCandidate(bbox=(x1, y1, x2, y2), confidence=float(confs[i]), keypoints=kps))
    return out


# ── Geometry helpers ─────────────────────────────────────────────────────────

def pixel_to_court(x: float, y: float, homography: np.ndarray) -> tuple[float, float] | None:
    p = homography @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-10:
        return None
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def court_to_pixel(cx: float, cy: float, homography_inv: np.ndarray) -> tuple[float, float] | None:
    p = homography_inv @ np.array([cx, cy, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-10:
        return None
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def map_player_to_court(
    player: PlayerDetection,
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    """Map a player's feet position (center-bottom of bbox) to court coordinates."""
    if homography is None:
        return None
    x = (player.bbox[0] + player.bbox[2]) / 2
    y = player.bbox[3]
    pos = pixel_to_court(x, y, homography)
    return pos if pos is not None else (0.0, 0.0)


def far_court_roi(
    homography: np.ndarray,
    frame_shape: tuple[int, ...],
    side_margin: float = 0.5,
    back_margin: float = 1.5,
) -> tuple[int, int, int, int] | None:
    """Pixel ROI covering the far half of the court plus room behind the
    baseline, used to zoom in on the far player."""
    try:
        H_inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    pts = [
        court_to_pixel(-_DOUBLES_HALF, _FAR_BASELINE_Y, H_inv),
        court_to_pixel(_DOUBLES_HALF, _FAR_BASELINE_Y, H_inv),
        court_to_pixel(-_DOUBLES_HALF, 0.0, H_inv),
        court_to_pixel(_DOUBLES_HALF, 0.0, H_inv),
    ]
    if any(p is None for p in pts):
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    far_y, net_y = min(ys), max(ys)
    if net_y - far_y < 20:
        return None
    h, w = frame_shape[:2]
    span_x = max(xs) - min(xs)
    x1 = int(max(0, min(xs) - side_margin * span_x))
    x2 = int(min(w, max(xs) + side_margin * span_x))
    y1 = int(max(0, far_y - back_margin * (net_y - far_y)))
    y2 = int(min(h, net_y + 0.15 * (net_y - far_y)))
    if x2 - x1 < 40 or y2 - y1 < 30:
        return None
    return (x1, y1, x2, y2)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detect_persons_with_far_crop(
    frame: np.ndarray,
    homography: np.ndarray | None,
    model_name: str = "yolov8s-pose.pt",
    imgsz: int = 1280,
    conf: float = 0.15,
    far_crop: bool = True,
    far_tiles: bool = False,
) -> list[PersonCandidate]:
    """Full-frame detection plus an upscaled far-court crop; merged by IoU.
    With ``far_tiles`` a wide crop is cut into overlapping tiles (see
    :func:`far_crop_tiles`)."""
    cands = detect_persons(frame, model_name=model_name, imgsz=imgsz, conf=conf)
    roi = far_court_roi(homography, frame.shape) if (far_crop and homography is not None) else None
    if roi is None:
        return cands
    x1, y1, x2, y2 = roi
    mapped: list[PersonCandidate] = []
    for tx1, tx2 in (far_crop_tiles(x1, x2, imgsz) if far_tiles else [(x1, x2)]):
        crop = frame[y1:y2, tx1:tx2]
        scale = min(4.0, max(1.0, imgsz / max(crop.shape[1], 1)))
        if scale > 1.05:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for c in detect_persons(crop, model_name=model_name, imgsz=imgsz, conf=conf):
            bx1, by1, bx2, by2 = c.bbox
            bbox = (bx1 / scale + tx1, by1 / scale + y1, bx2 / scale + tx1, by2 / scale + y1)
            kps = {k: (v[0] / scale + tx1, v[1] / scale + y1, v[2]) for k, v in c.keypoints.items()}
            cand = PersonCandidate(bbox=bbox, confidence=c.confidence, keypoints=kps)
            # the same person seen in two overlapping tiles: keep the more confident one
            dup = next((m for m in mapped if _iou(m.bbox, bbox) >= 0.5), None)
            if dup is None:
                mapped.append(cand)
            elif cand.confidence > dup.confidence:
                mapped[mapped.index(dup)] = cand
    merged: list[PersonCandidate] = list(mapped)
    for c in cands:
        if all(_iou(c.bbox, m.bbox) < 0.5 for m in mapped):
            merged.append(c)
    return merged


def far_crop_tiles(x1: int, x2: int, imgsz: int, max_fraction: float = 0.6, overlap: float = 0.2) -> list[tuple[int, int]]:
    """Split a wide far-court ROI into overlapping tiles so each can be
    upscaled properly.

    The detector resizes a crop to ``imgsz`` on its long side, so a far
    half that spans most of the frame (an oblique corner camera) gets
    almost no zoom and a 60 px player is missed. A ROI narrower than
    ``max_fraction`` of ``imgsz`` stays one tile (broadcast framing is
    unchanged); wider ones are cut into tiles of about half ``imgsz``
    with ``overlap`` shared between neighbours.
    """
    width = x2 - x1
    if width <= max_fraction * imgsz:
        return [(x1, x2)]
    n = int(np.ceil(width / (0.5 * imgsz)))
    step = width / n
    tile = step * (1.0 + overlap)
    tiles = []
    for i in range(n):
        tx1 = int(round(x1 + i * step - (tile - step) / 2))
        tx2 = int(round(tx1 + tile))
        tiles.append((max(x1, tx1), min(x2, tx2)))
    return tiles


# ── Court-aware role tracker ─────────────────────────────────────────────────

class PlayerTracker:
    """Assign near/far roles with court geometry + temporal continuity.

    With a homography: a candidate's feet are projected to court metres; it
    must lie within an extended court rectangle, and its court y sign picks
    the half. Within a half, the best candidate is the one closest to where
    that role was last seen (or, initially, closest to the centre line),
    weighted by detector confidence. Without a homography, the frame is
    split at ~45% height and the legacy margin filter is used.
    """

    def __init__(
        self,
        frame_shape: tuple[int, ...],
        homography: np.ndarray | None,
        max_court_x: float = 8.0,
        max_court_y: float = 17.0,
        max_missing: int = 15,
    ) -> None:
        self.h, self.w = frame_shape[:2]
        self.H = homography
        self.max_x = max_court_x
        self.max_y = max_court_y
        self.max_missing = max_missing
        self.last: dict[str, PlayerDetection] = {}
        self.missing: dict[str, int] = {"near_player": 0, "far_player": 0}
        self.last_pose: dict[str, PoseKeypoints] = {}

    def _half_of(self, c: PersonCandidate) -> str | None:
        fx = (c.bbox[0] + c.bbox[2]) / 2
        fy = c.bbox[3]
        if self.H is not None:
            pos = pixel_to_court(fx, fy, self.H)
            if pos is None:
                return None
            c.court_position = pos
            if abs(pos[0]) > self.max_x or abs(pos[1]) > self.max_y:
                return None
            # Ball kids kneel at the net posts, line judges sit outside the
            # doubles lines: nobody playing stands >1.5 m outside the court
            # right next to the net.
            if abs(pos[1]) < 3.0 and abs(pos[0]) > _DOUBLES_HALF + 1.0:
                return None
            return "near_player" if pos[1] < 0 else "far_player"
        # No homography: split the image; drop tiny margin blobs.
        area = (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1])
        in_margin = fx < self.w * 0.22 or fx > self.w * 0.78
        if in_margin and area < 2500:
            return None
        return "near_player" if fy > self.h * 0.45 else "far_player"

    def _score(self, role: str, c: PersonCandidate) -> float:
        score = c.confidence
        cx = (c.bbox[0] + c.bbox[2]) / 2
        cy = c.bbox[3]
        prev = self.last.get(role)
        if prev is not None:
            px = (prev.bbox[0] + prev.bbox[2]) / 2
            py = prev.bbox[3]
            d = np.hypot(cx - px, cy - py) / self.w
            if d > 0.25 and self.missing[role] < 5:
                return -1.0  # implausible jump while we still trust the track
            score += 1.0 - min(d / 0.25, 1.0)
        elif c.court_position is not None:
            score += 0.5 * (1.0 - min(abs(c.court_position[0]) / self.max_x, 1.0))
            # players live around the baselines; a body at the net with no
            # track history is more likely a ball kid
            if abs(c.court_position[1]) < 3.0:
                score -= 0.4
        else:
            score += 0.5 * (1.0 - abs(cx - self.w / 2) / (self.w / 2))
        # Bigger boxes are more likely the player than a ball kid in the same half.
        score += 0.2 * min((c.bbox[3] - c.bbox[1]) / (self.h * 0.25), 1.0)
        return score

    def update(
        self,
        frame_index: int,
        candidates: list[PersonCandidate],
    ) -> tuple[list[PlayerDetection], list[PoseKeypoints]]:
        pools: dict[str, list[PersonCandidate]] = {"near_player": [], "far_player": []}
        for c in candidates:
            half = self._half_of(c)
            if half is not None:
                pools[half].append(c)

        players: list[PlayerDetection] = []
        poses: list[PoseKeypoints] = []
        for role in ("near_player", "far_player"):
            best, best_score = None, 0.0
            for c in pools[role]:
                s = self._score(role, c)
                if s > best_score:
                    best, best_score = c, s
            if best is not None:
                det = PlayerDetection(
                    frame_index=frame_index, bbox=best.bbox, confidence=best.confidence,
                    court_position=best.court_position, role=role,
                )
                pose = PoseKeypoints(frame_index=frame_index, role=role, keypoints=dict(best.keypoints))
                self.last[role] = det
                self.last_pose[role] = pose
                self.missing[role] = 0
                players.append(det)
                poses.append(pose)
            elif role in self.last and self.missing[role] < self.max_missing:
                self.missing[role] += 1
                players.append(replace(self.last[role], frame_index=frame_index))
                if role in self.last_pose:
                    poses.append(replace(self.last_pose[role], frame_index=frame_index))
        return players, poses


# ── Segment-level drivers ────────────────────────────────────────────────────

def build_ball_trajectory(
    frames_dir: Path,
    segment: GameplaySegment,
    fps: float = 30.0,
    ball_method: str = "wasb",
    progress_callback: Callable[[int, int], None] | None = None,
    confidence_threshold: float = 0.5,
    frame_step: int = 1,
    max_speed_px: float = 150.0,
    max_gap_s: float = 0.5,
    smooth_window: int = 3,
    stationary_std_px: float = 3.0,
    strong_confidence: float = 0.75,
    far_roi: tuple[int, int, int, int] | None = None,
    homography: np.ndarray | None = None,
) -> dict[int, BallDetection | None]:
    """Build ball trajectory for a segment, returning a per-frame lookup."""
    trajectory = build_trajectory(
        frames_dir, segment.start_frame, segment.end_frame, fps=fps,
        method=ball_method,
        progress_callback=progress_callback,
        confidence_threshold=confidence_threshold,
        frame_step=frame_step,
        max_speed_px=max_speed_px,
        max_gap_s=max_gap_s,
        smooth_window=smooth_window,
        far_roi=far_roi,
        homography=homography,
    )
    return finalize_ball_by_frame(
        trajectory.detections, segment,
        stationary_std_px=stationary_std_px, strong_confidence=strong_confidence, fps=fps,
    )


def detect_players_segment(
    frames_dir: Path,
    segment: GameplaySegment,
    ball_by_frame: dict[int, BallDetection | None],
    homography: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    player_detect_stride: int = 1,
    model_name: str = "yolov8s-pose.pt",
    imgsz: int = 1280,
    conf: float = 0.15,
    far_crop: bool = True,
    far_tiles: bool = False,
    max_court_x: float = 7.0,
    max_court_y: float = 17.0,
) -> list[FrameTrackingResult]:
    """Detect players and poses for all frames in a segment.

    YOLO-pose runs every ``player_detect_stride`` frames; in-between frames
    reuse the previous detection.
    """
    results: list[FrameTrackingResult] = []
    tracker: PlayerTracker | None = None
    last_players: list[PlayerDetection] = []
    last_poses: list[PoseKeypoints] = []
    total = segment.end_frame - segment.start_frame + 1

    for frame_idx in range(segment.start_frame, segment.end_frame + 1):
        frame = cv2.imread(str(frames_dir / f"frame_{frame_idx:06d}.jpg"))
        if frame is None:
            continue
        if tracker is None:
            tracker = PlayerTracker(frame.shape, homography, max_court_x=max_court_x, max_court_y=max_court_y)

        offset = frame_idx - segment.start_frame
        if offset % player_detect_stride == 0:
            candidates = detect_persons_with_far_crop(
                frame, homography, model_name=model_name, imgsz=imgsz, conf=conf, far_crop=far_crop, far_tiles=far_tiles,
            )
            players, poses = tracker.update(frame_idx, candidates)
            last_players, last_poses = players, poses
        else:
            players = [replace(p, frame_index=frame_idx) for p in last_players]
            poses = [replace(p, frame_index=frame_idx) for p in last_poses]

        results.append(FrameTrackingResult(
            frame_index=frame_idx,
            ball=ball_by_frame.get(frame_idx),
            players=players,
            poses=poses,
        ))
        if progress_callback:
            progress_callback(offset + 1, total)

    return results
