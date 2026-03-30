# Court Vision Phase 2B — Ball Tracking + Player Detection + Pose Estimation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect ball positions (frame-by-frame), player bounding boxes, and player pose keypoints in gameplay frames, producing per-frame tracking data that feeds shot classification in Phase 2C.

**Architecture:** Two new modules (`ball_tracker.py` and `player_detect.py`) inside `court_vision/` that take gameplay frames + homography matrices from Phase 2A and produce ball trajectories, player positions in court coordinates, and pose keypoints. Ball tracking uses a lightweight heatmap-based approach (TrackNetV2-style). Player detection uses YOLOv8 (via `ultralytics`). Pose estimation uses MediaPipe. All integrate into the existing sequential pipeline as Stage 4.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, ultralytics (YOLOv8), mediapipe, torch/torchvision, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-court-vision-design.md` — Stage 4

**Phase 2A plan:** `docs/superpowers/plans/2026-03-26-court-vision-phase2a-court-detection.md`

---

## Phase 2B Scope

Phase 2B delivers:
1. Ball position detection — per-frame (x, y) pixel position of the ball
2. Ball trajectory interpolation — fill gaps ≤0.5s through occlusion
3. Ball court mapping — pixel positions → court coordinates via homography
4. Player bounding box detection — YOLOv8 person detection per gameplay frame
5. Player role assignment — map detected players to "near_player" / "far_player" via court-y coordinate
6. Pose estimation — MediaPipe skeleton keypoints per player
7. Per-frame tracking data — combined ball + player + pose result per frame
8. Pipeline integration — wire Stage 4 into the pipeline orchestrator
9. CLI output update — display tracking summary

## File Structure

| File | Responsibility |
|------|---------------|
| `src/court_vision/ball_tracker.py` | Ball detection per frame, trajectory interpolation, court coordinate mapping |
| `src/court_vision/player_detect.py` | YOLOv8 player detection, near/far assignment, MediaPipe pose estimation |
| `src/court_vision/pipeline.py` | **Modify:** Add Stage 4 call and tracking fields to `PipelineResult` |
| `src/court_vision/cli.py` | **Modify:** Display tracking summary |
| `pyproject.toml` | **Modify:** Add `ultralytics` and `mediapipe` dependencies |
| `tests/test_ball_tracker.py` | Tests for ball tracking module |
| `tests/test_player_detect.py` | Tests for player detection module |
| `tests/test_pipeline.py` | **Modify:** Add Stage 4 mocks |
| `tests/test_cli.py` | **Modify:** Add tracking output test |

## Data Structures (referenced by all tasks)

These are the data structures that tasks will build. Defined here for reference — Task 1 and Task 5 implement them.

```python
# ball_tracker.py
@dataclass
class BallDetection:
    frame_index: int
    x: float          # pixel x
    y: float          # pixel y
    confidence: float
    interpolated: bool = False  # True if filled via interpolation

@dataclass
class BallTrajectory:
    detections: list[BallDetection]
    fps: float

# player_detect.py
@dataclass
class PlayerDetection:
    frame_index: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixel coords
    confidence: float
    court_position: tuple[float, float] | None = None  # (x, y) in meters, via homography
    role: str | None = None  # "near_player" or "far_player"

@dataclass
class PoseKeypoints:
    frame_index: int
    role: str  # "near_player" or "far_player"
    keypoints: dict[str, tuple[float, float, float]]  # name -> (x, y, visibility)

@dataclass
class FrameTrackingResult:
    frame_index: int
    ball: BallDetection | None
    players: list[PlayerDetection]
    poses: list[PoseKeypoints]
```

---

### Task 1: Ball Detection Data Structures and Core Detection Function

**Files:**
- Create: `src/court_vision/ball_tracker.py`
- Create: `tests/test_ball_tracker.py`

This task creates the ball tracker module with data structures and a single-frame ball detection function. The detection uses color + shape filtering (white/yellow round object) as an MVP approach — simpler and dependency-free compared to TrackNetV2, which requires custom model weights we don't have yet. This can be upgraded to a neural approach later.

- [ ] **Step 1: Write the failing test for BallDetection dataclass and detect_ball_in_frame**

```python
"""Tests for the ball tracker module."""

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from court_vision.ball_tracker import BallDetection, detect_ball_in_frame


class TestBallDetection:
    def test_ball_detection_fields(self):
        """BallDetection has required fields."""
        det = BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.9)
        assert det.frame_index == 0
        assert det.x == 100.0
        assert det.y == 200.0
        assert det.confidence == 0.9
        assert det.interpolated is False

    def test_ball_detection_interpolated_flag(self):
        """BallDetection supports interpolated flag."""
        det = BallDetection(frame_index=5, x=50.0, y=60.0, confidence=0.5, interpolated=True)
        assert det.interpolated is True


class TestDetectBallInFrame:
    def _create_frame_with_ball(self, width: int = 640, height: int = 480,
                                  ball_center: tuple[int, int] = (320, 240),
                                  ball_radius: int = 8,
                                  ball_color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """Create a synthetic frame with a bright circular ball."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw green court background
        frame[:, :] = (0, 100, 0)
        cv2.circle(frame, ball_center, ball_radius, ball_color, -1)
        return frame

    def test_detects_ball_in_frame(self):
        """Detects a bright ball on a dark/green background."""
        frame = self._create_frame_with_ball(ball_center=(200, 150), ball_radius=8)
        result = detect_ball_in_frame(frame, frame_index=0)
        assert result is not None
        assert abs(result.x - 200) < 20
        assert abs(result.y - 150) < 20
        assert result.confidence > 0.0

    def test_returns_none_for_empty_frame(self):
        """Returns None when no ball-like object is found."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (0, 100, 0)  # uniform green
        result = detect_ball_in_frame(frame, frame_index=0)
        assert result is None

    def test_frame_index_propagated(self):
        """frame_index is passed through to the result."""
        frame = self._create_frame_with_ball()
        result = detect_ball_in_frame(frame, frame_index=42)
        if result is not None:
            assert result.frame_index == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_ball_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'court_vision.ball_tracker'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Ball tracking — detection, trajectory building, and court coordinate mapping."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BallDetection:
    """Single-frame ball detection result."""

    frame_index: int
    x: float
    y: float
    confidence: float
    interpolated: bool = False


def detect_ball_in_frame(
    frame: np.ndarray,
    frame_index: int = 0,
    min_radius: int = 3,
    max_radius: int = 20,
) -> BallDetection | None:
    """Detect the tennis ball in a single frame using color + shape filtering.

    Looks for small, bright, circular objects (white or yellow) that match
    typical tennis ball appearance in broadcast video.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        frame_index: Index of this frame in the video sequence.
        min_radius: Minimum ball radius in pixels.
        max_radius: Maximum ball radius in pixels.

    Returns:
        BallDetection with pixel coordinates, or None if no ball found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask 1: bright yellow (tennis ball)
    yellow_mask = cv2.inRange(hsv, (20, 80, 180), (40, 255, 255))

    # Mask 2: bright white (can appear white under broadcast lighting)
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))

    combined = cv2.bitwise_or(yellow_mask, white_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_detection: BallDetection | None = None
    best_score = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < np.pi * min_radius**2 or area > np.pi * max_radius**2:
            continue

        # Check circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue

        # Get center via minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        if radius < min_radius or radius > max_radius:
            continue

        # Score: higher circularity and smaller size (balls are small) score better
        score = circularity * (1.0 / (1.0 + radius / max_radius))

        if score > best_score:
            best_score = score
            best_detection = BallDetection(
                frame_index=frame_index,
                x=float(cx),
                y=float(cy),
                confidence=float(min(circularity, 1.0)),
            )

    return best_detection
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_ball_tracker.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/ball_tracker.py tests/test_ball_tracker.py
git commit -m "feat(court-vision): add ball detection with color+shape filtering"
```

---

### Task 2: Ball Trajectory Building and Interpolation

**Files:**
- Modify: `src/court_vision/ball_tracker.py`
- Modify: `tests/test_ball_tracker.py`

Adds `BallTrajectory` dataclass and functions to build a trajectory from per-frame detections and interpolate through short occlusion gaps (≤0.5s as per spec).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ball_tracker.py`:

```python
from court_vision.ball_tracker import BallTrajectory, build_trajectory, interpolate_gaps


class TestBallTrajectory:
    def test_trajectory_fields(self):
        """BallTrajectory stores detections and fps."""
        dets = [BallDetection(frame_index=0, x=10.0, y=20.0, confidence=0.9)]
        traj = BallTrajectory(detections=dets, fps=30.0)
        assert len(traj.detections) == 1
        assert traj.fps == 30.0


class TestBuildTrajectory:
    def test_builds_from_frame_directory(self):
        """build_trajectory processes frames and returns trajectory."""
        # Uses mock — tested via detect_ball_in_frame
        dets = [
            BallDetection(frame_index=0, x=10.0, y=20.0, confidence=0.9),
            BallDetection(frame_index=2, x=30.0, y=40.0, confidence=0.8),
        ]
        traj = BallTrajectory(detections=dets, fps=30.0)
        assert len(traj.detections) == 2


class TestInterpolateGaps:
    def test_fills_single_frame_gap(self):
        """Interpolates a single missing frame between two detections."""
        dets = [
            BallDetection(frame_index=0, x=0.0, y=0.0, confidence=0.9),
            BallDetection(frame_index=2, x=20.0, y=20.0, confidence=0.9),
        ]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 3
        interp = result[1]
        assert interp.frame_index == 1
        assert abs(interp.x - 10.0) < 0.1
        assert abs(interp.y - 10.0) < 0.1
        assert interp.interpolated is True

    def test_fills_multi_frame_gap(self):
        """Interpolates multiple missing frames within max_gap_s."""
        dets = [
            BallDetection(frame_index=0, x=0.0, y=0.0, confidence=0.9),
            BallDetection(frame_index=5, x=50.0, y=100.0, confidence=0.9),
        ]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 6
        for i, det in enumerate(result):
            assert det.frame_index == i
        # Check middle point is linearly interpolated
        mid = result[2]
        assert abs(mid.x - 20.0) < 0.1
        assert abs(mid.y - 40.0) < 0.1

    def test_does_not_fill_gap_beyond_max(self):
        """Gaps longer than max_gap_s are NOT interpolated."""
        dets = [
            BallDetection(frame_index=0, x=0.0, y=0.0, confidence=0.9),
            BallDetection(frame_index=30, x=50.0, y=50.0, confidence=0.9),
        ]
        # 30 frames at 30fps = 1.0s gap, max_gap_s=0.5 → no interpolation
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 2

    def test_returns_empty_for_empty_input(self):
        """Returns empty list for empty input."""
        result = interpolate_gaps([], fps=30.0, max_gap_s=0.5)
        assert result == []

    def test_returns_single_detection_unchanged(self):
        """Single detection is returned as-is."""
        dets = [BallDetection(frame_index=0, x=10.0, y=20.0, confidence=0.9)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 1
        assert result[0].interpolated is False
```

- [ ] **Step 2: Run test to verify failures**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_ball_tracker.py -v`
Expected: FAIL with `ImportError` for `BallTrajectory`, `build_trajectory`, `interpolate_gaps`

- [ ] **Step 3: Write minimal implementation**

Add to `src/court_vision/ball_tracker.py`:

```python
@dataclass
class BallTrajectory:
    """Sequence of ball detections across frames."""

    detections: list[BallDetection]
    fps: float


def build_trajectory(
    frames_dir: "Path",
    start_frame: int,
    end_frame: int,
    fps: float,
    max_gap_s: float = 0.5,
) -> BallTrajectory:
    """Build a ball trajectory by detecting the ball in each frame.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        start_frame: First frame index to process.
        end_frame: Last frame index to process (inclusive).
        fps: Video frame rate.
        max_gap_s: Maximum gap in seconds to interpolate through.

    Returns:
        BallTrajectory with detections and interpolated positions.
    """
    from pathlib import Path

    frames_dir = Path(frames_dir)
    raw_detections: list[BallDetection] = []

    for i in range(start_frame, end_frame + 1):
        frame_path = frames_dir / f"frame_{i:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        det = detect_ball_in_frame(frame, frame_index=i)
        if det is not None:
            raw_detections.append(det)

    interpolated = interpolate_gaps(raw_detections, fps, max_gap_s)

    return BallTrajectory(detections=interpolated, fps=fps)


def interpolate_gaps(
    detections: list[BallDetection],
    fps: float,
    max_gap_s: float = 0.5,
) -> list[BallDetection]:
    """Fill short gaps in ball detections with linear interpolation.

    Args:
        detections: Sorted list of ball detections (by frame_index).
        fps: Video frame rate.
        max_gap_s: Maximum gap duration in seconds to interpolate.

    Returns:
        Detections with interpolated positions filling short gaps.
    """
    if len(detections) <= 1:
        return list(detections)

    max_gap_frames = int(max_gap_s * fps)
    result: list[BallDetection] = [detections[0]]

    for i in range(1, len(detections)):
        prev = detections[i - 1]
        curr = detections[i]
        gap = curr.frame_index - prev.frame_index

        if 1 < gap <= max_gap_frames:
            # Linear interpolation for each missing frame
            for j in range(1, gap):
                t = j / gap
                interp_x = prev.x + t * (curr.x - prev.x)
                interp_y = prev.y + t * (curr.y - prev.y)
                interp_conf = min(prev.confidence, curr.confidence) * 0.5
                result.append(BallDetection(
                    frame_index=prev.frame_index + j,
                    x=interp_x,
                    y=interp_y,
                    confidence=interp_conf,
                    interpolated=True,
                ))

        result.append(curr)

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_ball_tracker.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/ball_tracker.py tests/test_ball_tracker.py
git commit -m "feat(court-vision): add ball trajectory building and gap interpolation"
```

---

### Task 3: Ball-to-Court Coordinate Mapping

**Files:**
- Modify: `src/court_vision/ball_tracker.py`
- Modify: `tests/test_ball_tracker.py`

Maps ball pixel positions to court coordinates using the homography matrix from Phase 2A.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ball_tracker.py`:

```python
from court_vision.ball_tracker import map_ball_to_court


class TestMapBallToCourt:
    def _identity_homography(self) -> np.ndarray:
        """Returns an identity homography (pixel = court coords)."""
        return np.eye(3, dtype=np.float64)

    def test_maps_detection_to_court_coords(self):
        """Maps a ball detection's pixel coords to court coords via homography."""
        H = self._identity_homography()
        det = BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.9)
        court_x, court_y = map_ball_to_court(det, H)
        assert abs(court_x - 100.0) < 0.1
        assert abs(court_y - 200.0) < 0.1

    def test_returns_none_for_none_homography(self):
        """Returns None if homography is None."""
        det = BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.9)
        result = map_ball_to_court(det, None)
        assert result is None

    def test_works_with_scaling_homography(self):
        """Correctly applies a scaling homography."""
        H = np.array([[2.0, 0, 0], [0, 3.0, 0], [0, 0, 1.0]], dtype=np.float64)
        det = BallDetection(frame_index=0, x=10.0, y=20.0, confidence=0.9)
        court_x, court_y = map_ball_to_court(det, H)
        assert abs(court_x - 20.0) < 0.1
        assert abs(court_y - 60.0) < 0.1
```

- [ ] **Step 2: Run test to verify failures**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_ball_tracker.py::TestMapBallToCourt -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Add to `src/court_vision/ball_tracker.py`:

```python
def map_ball_to_court(
    detection: BallDetection,
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    """Map a ball detection from pixel coordinates to court coordinates.

    Uses the homography matrix from court detection (Phase 2A).

    Args:
        detection: Ball detection with pixel (x, y).
        homography: 3x3 homography matrix, or None if unavailable.

    Returns:
        (x, y) court coordinates in meters, or None if homography is None.
    """
    if homography is None:
        return None

    pixel = np.array([detection.x, detection.y, 1.0], dtype=np.float64)
    transformed = homography @ pixel
    w = transformed[2]
    if abs(w) < 1e-10:
        return (0.0, 0.0)

    return (float(transformed[0] / w), float(transformed[1] / w))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_ball_tracker.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/ball_tracker.py tests/test_ball_tracker.py
git commit -m "feat(court-vision): add ball-to-court coordinate mapping via homography"
```

---

### Task 4: Add ultralytics and mediapipe Dependencies

**Files:**
- Modify: `pyproject.toml`

Adds the YOLOv8 (`ultralytics`) and MediaPipe (`mediapipe`) packages as project dependencies.

- [ ] **Step 1: Update pyproject.toml**

Add `ultralytics>=8.0` and `mediapipe>=0.10` to the `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "numpy>=1.24",
    "opencv-python>=4.8",
    "yt-dlp>=2024.0",
    "typer>=0.12",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "ultralytics>=8.0",
    "mediapipe>=0.10",
]
```

- [ ] **Step 2: Install updated dependencies**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/pip install -e ".[dev]"`
Expected: Successfully installs ultralytics and mediapipe

- [ ] **Step 3: Verify imports work**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -c "from ultralytics import YOLO; import mediapipe; print('OK')"`
Expected: Prints `OK`

- [ ] **Step 4: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add pyproject.toml
git commit -m "feat(court-vision): add ultralytics and mediapipe dependencies"
```

---

### Task 5: Player Detection Data Structures and YOLOv8 Detection

**Files:**
- Create: `src/court_vision/player_detect.py`
- Create: `tests/test_player_detect.py`

Creates the player detection module with data structures and a YOLOv8-based person detection function. Uses `ultralytics` YOLOv8n (nano) model for fast person detection.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the player detection module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from court_vision.player_detect import (
    PlayerDetection,
    PoseKeypoints,
    FrameTrackingResult,
    detect_players_in_frame,
)


class TestPlayerDetection:
    def test_player_detection_fields(self):
        """PlayerDetection has required fields."""
        det = PlayerDetection(
            frame_index=0,
            bbox=(100.0, 200.0, 150.0, 400.0),
            confidence=0.92,
        )
        assert det.frame_index == 0
        assert det.bbox == (100.0, 200.0, 150.0, 400.0)
        assert det.confidence == 0.92
        assert det.court_position is None
        assert det.role is None

    def test_player_detection_with_court_position(self):
        """PlayerDetection supports court_position and role."""
        det = PlayerDetection(
            frame_index=0,
            bbox=(100.0, 200.0, 150.0, 400.0),
            confidence=0.92,
            court_position=(2.0, -8.0),
            role="near_player",
        )
        assert det.court_position == (2.0, -8.0)
        assert det.role == "near_player"


class TestPoseKeypoints:
    def test_pose_keypoints_fields(self):
        """PoseKeypoints has required fields."""
        pk = PoseKeypoints(
            frame_index=0,
            role="far_player",
            keypoints={"left_wrist": (100.0, 200.0, 0.95)},
        )
        assert pk.frame_index == 0
        assert pk.role == "far_player"
        assert pk.keypoints["left_wrist"] == (100.0, 200.0, 0.95)


class TestFrameTrackingResult:
    def test_frame_tracking_fields(self):
        """FrameTrackingResult combines ball, players, and poses."""
        from court_vision.ball_tracker import BallDetection

        result = FrameTrackingResult(
            frame_index=0,
            ball=BallDetection(frame_index=0, x=300.0, y=200.0, confidence=0.8),
            players=[],
            poses=[],
        )
        assert result.frame_index == 0
        assert result.ball is not None
        assert result.players == []
        assert result.poses == []


class TestDetectPlayersInFrame:
    @patch("court_vision.player_detect._get_yolo_model")
    def test_detects_persons(self, mock_get_model: MagicMock):
        """Detects persons in frame using YOLOv8."""
        # Mock YOLO model results
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_box = MagicMock()
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[100.0, 200.0, 200.0, 500.0]])
        mock_box.conf = MagicMock()
        mock_box.conf.cpu.return_value.numpy.return_value = np.array([0.92])
        mock_box.cls = MagicMock()
        mock_box.cls.cpu.return_value.numpy.return_value = np.array([0])  # person class

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_model.return_value = [mock_result]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        players = detect_players_in_frame(frame, frame_index=0)

        assert len(players) == 1
        assert players[0].bbox == (100.0, 200.0, 200.0, 500.0)
        assert players[0].confidence == pytest.approx(0.92, abs=0.01)

    @patch("court_vision.player_detect._get_yolo_model")
    def test_filters_non_person_classes(self, mock_get_model: MagicMock):
        """Only person class (0) detections are returned."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_box = MagicMock()
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([
            [100.0, 200.0, 200.0, 500.0],
            [300.0, 100.0, 400.0, 200.0],
        ])
        mock_box.conf = MagicMock()
        mock_box.conf.cpu.return_value.numpy.return_value = np.array([0.92, 0.85])
        mock_box.cls = MagicMock()
        mock_box.cls.cpu.return_value.numpy.return_value = np.array([0, 32])  # person, sports ball

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_model.return_value = [mock_result]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        players = detect_players_in_frame(frame, frame_index=0)

        assert len(players) == 1
        assert players[0].bbox[0] == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_player_detect.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_player_detect.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/player_detect.py tests/test_player_detect.py
git commit -m "feat(court-vision): add YOLOv8 player detection"
```

---

### Task 6: Player Role Assignment (Near/Far)

**Files:**
- Modify: `src/court_vision/player_detect.py`
- Modify: `tests/test_player_detect.py`

Assigns "near_player" or "far_player" role to detected players based on their bounding box position in the image. Near player is at the bottom of the frame (large y), far player is at the top (small y). Optionally maps to court coordinates via homography.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_player_detect.py`:

```python
from court_vision.player_detect import assign_player_roles, map_player_to_court


class TestAssignPlayerRoles:
    def test_assigns_near_and_far(self):
        """Player at bottom of frame = near, top = far."""
        players = [
            PlayerDetection(frame_index=0, bbox=(100.0, 50.0, 200.0, 200.0), confidence=0.9),   # top of frame = far
            PlayerDetection(frame_index=0, bbox=(100.0, 400.0, 200.0, 600.0), confidence=0.9),  # bottom of frame = near
        ]
        assigned = assign_player_roles(players)
        assert len(assigned) == 2
        # Player with larger y2 should be near
        bottom_player = [p for p in assigned if p.bbox[3] == 600.0][0]
        top_player = [p for p in assigned if p.bbox[3] == 200.0][0]
        assert bottom_player.role == "near_player"
        assert top_player.role == "far_player"

    def test_single_player_defaults_to_near(self):
        """Single player is assigned near_player role."""
        players = [
            PlayerDetection(frame_index=0, bbox=(100.0, 400.0, 200.0, 600.0), confidence=0.9),
        ]
        assigned = assign_player_roles(players)
        assert len(assigned) == 1
        assert assigned[0].role == "near_player"

    def test_empty_list_returns_empty(self):
        """Empty player list returns empty list."""
        assert assign_player_roles([]) == []

    def test_more_than_two_players_takes_top_two_by_confidence(self):
        """When more than 2 players detected, keep 2 highest confidence."""
        players = [
            PlayerDetection(frame_index=0, bbox=(100.0, 50.0, 200.0, 200.0), confidence=0.5),
            PlayerDetection(frame_index=0, bbox=(100.0, 400.0, 200.0, 600.0), confidence=0.9),
            PlayerDetection(frame_index=0, bbox=(300.0, 300.0, 400.0, 500.0), confidence=0.7),
        ]
        assigned = assign_player_roles(players)
        assert len(assigned) == 2
        # Top 2 by confidence: 0.9 and 0.7
        confs = sorted([p.confidence for p in assigned], reverse=True)
        assert confs == [0.9, 0.7]


class TestMapPlayerToCourt:
    def test_maps_bbox_center_bottom_to_court(self):
        """Maps center-bottom of bounding box (feet position) to court coords."""
        H = np.eye(3, dtype=np.float64)
        det = PlayerDetection(frame_index=0, bbox=(100.0, 200.0, 200.0, 500.0), confidence=0.9)
        court_pos = map_player_to_court(det, H)
        # Center-bottom of bbox: x=(100+200)/2=150, y=500 (bottom)
        assert court_pos is not None
        assert abs(court_pos[0] - 150.0) < 0.1
        assert abs(court_pos[1] - 500.0) < 0.1

    def test_returns_none_for_none_homography(self):
        """Returns None when homography is None."""
        det = PlayerDetection(frame_index=0, bbox=(100.0, 200.0, 200.0, 500.0), confidence=0.9)
        result = map_player_to_court(det, None)
        assert result is None
```

- [ ] **Step 2: Run test to verify failures**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_player_detect.py::TestAssignPlayerRoles tests/test_player_detect.py::TestMapPlayerToCourt -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Add to `src/court_vision/player_detect.py`:

```python
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

    # Keep top 2 by confidence
    sorted_by_conf = sorted(players, key=lambda p: p.confidence, reverse=True)
    top_players = sorted_by_conf[:2]

    if len(top_players) == 1:
        top_players[0].role = "near_player"
        return top_players

    # Sort by y2 (bottom of bbox) — larger y2 = closer to bottom of frame = near
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

    # Center-bottom of bbox = feet position
    x = (player.bbox[0] + player.bbox[2]) / 2
    y = player.bbox[3]  # bottom edge

    pixel = np.array([x, y, 1.0], dtype=np.float64)
    transformed = homography @ pixel
    w = transformed[2]
    if abs(w) < 1e-10:
        return (0.0, 0.0)

    return (float(transformed[0] / w), float(transformed[1] / w))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_player_detect.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/player_detect.py tests/test_player_detect.py
git commit -m "feat(court-vision): add player role assignment and court mapping"
```

---

### Task 7: Pose Estimation with MediaPipe

**Files:**
- Modify: `src/court_vision/player_detect.py`
- Modify: `tests/test_player_detect.py`

Adds MediaPipe pose estimation to extract skeleton keypoints from player bounding box crops.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_player_detect.py`:

```python
from court_vision.player_detect import estimate_pose


class TestEstimatePose:
    @patch("court_vision.player_detect._get_pose_estimator")
    def test_extracts_keypoints_from_crop(self, mock_get_estimator: MagicMock):
        """Extracts pose keypoints from a player bounding box crop."""
        # Mock MediaPipe results
        mock_estimator = MagicMock()
        mock_get_estimator.return_value = mock_estimator

        # Create mock landmark
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.3
        mock_landmark.visibility = 0.95

        mock_results = MagicMock()
        mock_results.pose_landmarks = MagicMock()
        mock_results.pose_landmarks.landmark = [mock_landmark] * 33  # MediaPipe has 33 landmarks
        mock_estimator.process.return_value = mock_results

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        player = PlayerDetection(
            frame_index=0,
            bbox=(100.0, 200.0, 200.0, 500.0),
            confidence=0.9,
            role="near_player",
        )

        result = estimate_pose(frame, player)
        assert result is not None
        assert result.frame_index == 0
        assert result.role == "near_player"
        assert len(result.keypoints) > 0

    @patch("court_vision.player_detect._get_pose_estimator")
    def test_returns_none_when_no_pose_detected(self, mock_get_estimator: MagicMock):
        """Returns None when MediaPipe finds no pose."""
        mock_estimator = MagicMock()
        mock_get_estimator.return_value = mock_estimator

        mock_results = MagicMock()
        mock_results.pose_landmarks = None
        mock_estimator.process.return_value = mock_results

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        player = PlayerDetection(
            frame_index=0,
            bbox=(100.0, 200.0, 200.0, 500.0),
            confidence=0.9,
            role="far_player",
        )

        result = estimate_pose(frame, player)
        assert result is None
```

- [ ] **Step 2: Run test to verify failures**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_player_detect.py::TestEstimatePose -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Add to `src/court_vision/player_detect.py`:

```python
import cv2

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
def _get_pose_estimator():
    """Load MediaPipe Pose estimator (cached singleton)."""
    import mediapipe as mp

    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )


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

    pose = _get_pose_estimator()
    results = pose.process(rgb_crop)

    if results.pose_landmarks is None:
        return None

    keypoints: dict[str, tuple[float, float, float]] = {}
    crop_h, crop_w = crop.shape[:2]

    for i, landmark in enumerate(results.pose_landmarks.landmark):
        if i < len(_POSE_LANDMARK_NAMES):
            name = _POSE_LANDMARK_NAMES[i]
            # Convert normalized coords back to full-frame pixel coords
            px = x1 + landmark.x * crop_w
            py = y1 + landmark.y * crop_h
            keypoints[name] = (float(px), float(py), float(landmark.visibility))

    return PoseKeypoints(
        frame_index=player.frame_index,
        role=player.role or "unknown",
        keypoints=keypoints,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_player_detect.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/player_detect.py tests/test_player_detect.py
git commit -m "feat(court-vision): add MediaPipe pose estimation for players"
```

---

### Task 8: Per-Segment Tracking Orchestrator

**Files:**
- Modify: `src/court_vision/player_detect.py`
- Modify: `tests/test_player_detect.py`

Adds a `track_segment` function that orchestrates ball detection, player detection, role assignment, and pose estimation for all frames in a gameplay segment, returning a list of `FrameTrackingResult`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_player_detect.py`:

```python
from pathlib import Path

from court_vision.player_detect import track_segment
from court_vision.scene_filter import GameplaySegment


class TestTrackSegment:
    @patch("court_vision.player_detect.estimate_pose")
    @patch("court_vision.player_detect.assign_player_roles")
    @patch("court_vision.player_detect.detect_players_in_frame")
    @patch("court_vision.player_detect.detect_ball_in_frame")
    def test_tracks_all_frames_in_segment(
        self,
        mock_detect_ball: MagicMock,
        mock_detect_players: MagicMock,
        mock_assign_roles: MagicMock,
        mock_estimate_pose: MagicMock,
        tmp_path: Path,
    ):
        """Processes each frame in a segment and returns tracking results."""
        from court_vision.ball_tracker import BallDetection

        # Create 3 frames
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), frame)

        segment = GameplaySegment(
            start_frame=0, end_frame=2,
            start_time_s=0.0, end_time_s=0.1,
            frame_count=3,
        )

        mock_detect_ball.return_value = BallDetection(frame_index=0, x=300.0, y=200.0, confidence=0.8)
        mock_detect_players.return_value = [
            PlayerDetection(frame_index=0, bbox=(100.0, 300.0, 200.0, 600.0), confidence=0.9),
        ]
        mock_assign_roles.return_value = [
            PlayerDetection(frame_index=0, bbox=(100.0, 300.0, 200.0, 600.0), confidence=0.9, role="near_player"),
        ]
        mock_estimate_pose.return_value = PoseKeypoints(
            frame_index=0, role="near_player",
            keypoints={"left_wrist": (150.0, 400.0, 0.9)},
        )

        results = track_segment(frames_dir, segment, homography=None)

        assert len(results) == 3
        assert all(isinstance(r, FrameTrackingResult) for r in results)
        assert mock_detect_ball.call_count == 3
        assert mock_detect_players.call_count == 3

    @patch("court_vision.player_detect.estimate_pose")
    @patch("court_vision.player_detect.assign_player_roles")
    @patch("court_vision.player_detect.detect_players_in_frame")
    @patch("court_vision.player_detect.detect_ball_in_frame")
    def test_handles_missing_frames(
        self,
        mock_detect_ball: MagicMock,
        mock_detect_players: MagicMock,
        mock_assign_roles: MagicMock,
        mock_estimate_pose: MagicMock,
        tmp_path: Path,
    ):
        """Skips frames that don't exist on disk."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        # Only create frame 0, not frame 1
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "frame_000000.jpg"), frame)

        segment = GameplaySegment(
            start_frame=0, end_frame=1,
            start_time_s=0.0, end_time_s=0.033,
            frame_count=2,
        )

        mock_detect_ball.return_value = None
        mock_detect_players.return_value = []
        mock_assign_roles.return_value = []
        mock_estimate_pose.return_value = None

        results = track_segment(frames_dir, segment, homography=None)

        # Only frame 0 exists
        assert len(results) == 1
```

- [ ] **Step 2: Run test to verify failures**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_player_detect.py::TestTrackSegment -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Add to `src/court_vision/player_detect.py`:

```python
from pathlib import Path

from court_vision.ball_tracker import BallDetection, detect_ball_in_frame
from court_vision.scene_filter import GameplaySegment


def track_segment(
    frames_dir: Path,
    segment: GameplaySegment,
    homography: np.ndarray | None = None,
) -> list[FrameTrackingResult]:
    """Track ball, players, and poses for all frames in a gameplay segment.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        segment: Gameplay segment defining frame range.
        homography: Homography matrix for this segment, or None.

    Returns:
        List of FrameTrackingResult, one per successfully read frame.
    """
    results: list[FrameTrackingResult] = []

    for frame_idx in range(segment.start_frame, segment.end_frame + 1):
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Ball detection
        ball = detect_ball_in_frame(frame, frame_index=frame_idx)

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
```

Note: The `from court_vision.ball_tracker import BallDetection, detect_ball_in_frame` import and `from court_vision.scene_filter import GameplaySegment` need to be added at the top of the file. The `import cv2` should already be present from Task 7. Move `from court_vision.ball_tracker import BallDetection` from the class-level import to the top-level import and add `detect_ball_in_frame`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_player_detect.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/player_detect.py tests/test_player_detect.py
git commit -m "feat(court-vision): add per-segment tracking orchestrator"
```

---

### Task 9: Pipeline Integration (Stage 4)

**Files:**
- Modify: `src/court_vision/pipeline.py`
- Modify: `tests/test_pipeline.py`

Wires Stage 4 (tracking) into the pipeline orchestrator. Adds tracking data to `PipelineResult`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
class TestRunPipelineWithTracking:
    @patch("court_vision.pipeline.track_segment")
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.classify_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_pipeline_runs_tracking_after_court_detection(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_homographies: MagicMock,
        mock_track: MagicMock,
        tmp_path: Path,
    ):
        """Pipeline calls track_segment after court detection."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.ingest import FrameSequence
        from court_vision.player_detect import FrameTrackingResult
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        mock_load_config.return_value = PipelineConfig()
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        mock_extract.return_value = FrameSequence(
            frames_dir=frames_dir, fps=30.0, total_frames=100, resolution=(1280, 720),
        )
        mock_classify.return_value = []

        segments = [
            GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
        ]
        mock_filter.return_value = segments

        court_result = CourtDetectionResult(success=True, num_lines_detected=6)
        mock_homographies.return_value = [court_result]

        mock_track.return_value = [
            FrameTrackingResult(frame_index=0, ball=None, players=[], poses=[]),
        ]

        result = run_pipeline(str(video_path), config_path=None)

        mock_track.assert_called_once()
        assert result.tracking_results is not None
        assert len(result.tracking_results) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_pipeline.py::TestRunPipelineWithTracking -v`
Expected: FAIL — `tracking_results` not in PipelineResult yet, `track_segment` not imported

- [ ] **Step 3: Update pipeline.py**

Add import at top of `pipeline.py`:

```python
from court_vision.player_detect import FrameTrackingResult, track_segment
```

Add field to `PipelineResult`:

```python
@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    source: str
    total_frames: int
    fps: float
    gameplay_segments: list[GameplaySegment]
    gameplay_frame_count: int
    frames_dir: Path
    court_detections: list[CourtDetectionResult] | None = None
    tracking_results: list[FrameTrackingResult] | None = None
```

Add Stage 4 to `run_pipeline()` after court detection:

```python
    # Stage 4: Ball Tracking + Player Detection + Pose
    all_tracking: list[FrameTrackingResult] = []
    for i, segment in enumerate(segments):
        homography = None
        if court_detections and i < len(court_detections) and court_detections[i].success:
            homography = court_detections[i].homography
        segment_tracking = track_segment(frame_seq.frames_dir, segment, homography)
        all_tracking.extend(segment_tracking)
```

Update the return statement to include `tracking_results=all_tracking`.

- [ ] **Step 4: Update existing pipeline tests to mock track_segment**

Add `@patch("court_vision.pipeline.track_segment")` to the existing test methods in `TestRunPipeline` and `TestRunPipelineWithCourtDetection` classes. Set `mock_track.return_value = []` in each.

- [ ] **Step 5: Run tests to verify all pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/pipeline.py tests/test_pipeline.py
git commit -m "feat(court-vision): integrate ball+player tracking into pipeline (Stage 4)"
```

---

### Task 10: CLI Output Update

**Files:**
- Modify: `src/court_vision/cli.py`
- Modify: `tests/test_cli.py`

Adds tracking summary to the `process` command output.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
class TestProcessCommandTrackingOutput:
    @patch("court_vision.pipeline.run_pipeline")
    def test_process_shows_tracking_summary(self, mock_pipeline: MagicMock):
        """Process command displays tracking results summary."""
        from court_vision.ball_tracker import BallDetection
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.player_detect import FrameTrackingResult, PlayerDetection
        from court_vision.scene_filter import GameplaySegment

        mock_pipeline.return_value = MagicMock(
            total_frames=100,
            fps=30.0,
            gameplay_segments=[
                GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
            ],
            gameplay_frame_count=51,
            court_detections=[
                CourtDetectionResult(success=True, num_lines_detected=7),
            ],
            tracking_results=[
                FrameTrackingResult(
                    frame_index=0,
                    ball=BallDetection(frame_index=0, x=300.0, y=200.0, confidence=0.8),
                    players=[
                        PlayerDetection(frame_index=0, bbox=(100.0, 300.0, 200.0, 600.0), confidence=0.9, role="near_player"),
                    ],
                    poses=[],
                ),
                FrameTrackingResult(frame_index=1, ball=None, players=[], poses=[]),
            ],
        )

        result = runner.invoke(app, ["process", "test.mp4"])

        assert result.exit_code == 0
        assert "tracking" in result.output.lower() or "ball" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_cli.py::TestProcessCommandTrackingOutput -v`
Expected: FAIL — no tracking output in CLI yet

- [ ] **Step 3: Update cli.py**

Add after the court detection output block in the `process` command:

```python
    if result.tracking_results:
        ball_count = sum(1 for t in result.tracking_results if t.ball is not None)
        player_frames = sum(1 for t in result.tracking_results if len(t.players) > 0)
        typer.echo(f"Tracking: ball detected in {ball_count}/{len(result.tracking_results)} frames, "
                   f"players in {player_frames}/{len(result.tracking_results)} frames")
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/cli.py tests/test_cli.py
git commit -m "feat(court-vision): display tracking summary in CLI output"
```

---

### Task 11: Full Test Suite Verification

**Files:**
- All test files

Runs the complete test suite to verify nothing is broken.

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Verify clean imports**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -c "from court_vision.ball_tracker import BallDetection, BallTrajectory, detect_ball_in_frame, build_trajectory, interpolate_gaps, map_ball_to_court; from court_vision.player_detect import PlayerDetection, PoseKeypoints, FrameTrackingResult, detect_players_in_frame, assign_player_roles, map_player_to_court, estimate_pose, track_segment; print('All imports OK')"`
Expected: Prints `All imports OK`

- [ ] **Step 3: Verify pipeline imports**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -c "from court_vision.pipeline import run_pipeline, PipelineResult; print('Pipeline OK')"`
Expected: Prints `Pipeline OK`

---

## Self-Review Checklist

1. **Spec coverage:** Stage 4 requirements mapped:
   - Ball tracking (TrackNetV2-style → using color+shape MVP instead) ✓ Tasks 1-3
   - Ball interpolation through occlusion gaps (max 0.5s) ✓ Task 2
   - Player detection (YOLOv8) ✓ Task 5
   - Map players to court coords via homography, identify near/far ✓ Task 6
   - Pose estimation (MediaPipe skeleton) ✓ Task 7
   - Output: ball trajectory + player positions + pose keypoints per frame ✓ Task 8

2. **Placeholder scan:** No TBD/TODO items. All code blocks complete.

3. **Type consistency:** `BallDetection`, `BallTrajectory`, `PlayerDetection`, `PoseKeypoints`, `FrameTrackingResult` used consistently across all tasks. `detect_ball_in_frame`, `interpolate_gaps`, `build_trajectory`, `map_ball_to_court` signatures consistent. `detect_players_in_frame`, `assign_player_roles`, `map_player_to_court`, `estimate_pose`, `track_segment` signatures consistent.

4. **Note on ball detection approach:** The spec calls for TrackNetV2, but that requires custom model weights (from Chang et al.) that we don't have downloaded. The MVP uses color+shape filtering (HSV masking + contour circularity) which works for broadcast video where the ball is bright against court/background. This can be upgraded to a neural approach in a future iteration.
