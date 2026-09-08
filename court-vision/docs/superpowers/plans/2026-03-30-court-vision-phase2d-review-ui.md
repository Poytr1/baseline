# Phase 2D: Streamlit Review UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit-based human review UI that lets users view pipeline output with video frame overlays, edit shot classifications, and export corrected match data.

**Architecture:** Single Streamlit app (`review_app.py`) with a match data loader module (`review_data.py`) for JSON deserialization and overlay rendering (`overlay.py`) using OpenCV. The CLI gets a `review` command that launches the Streamlit app. The UI is organized into a sidebar point list, a main frame viewer with overlays, and an editing panel below.

**Tech Stack:** Streamlit, OpenCV (already installed), dataclasses (existing), JSON (existing)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/court_vision/review_data.py` | Load MatchData from JSON with full deserialization (points, shots, placements), save back to JSON with review updates |
| `src/court_vision/overlay.py` | Render frame overlays: ball position circle, player bounding boxes, pose skeleton, court lines |
| `src/court_vision/review_app.py` | Streamlit app: point list, frame viewer, shot editor, bulk actions |
| `src/court_vision/cli.py` | Add `review` command that launches the Streamlit app |
| `tests/test_review_data.py` | Tests for JSON round-trip serialization/deserialization |
| `tests/test_overlay.py` | Tests for overlay rendering functions |
| `tests/test_cli.py` | Add test for `review` command |

---

### Task 1: Add Streamlit Dependency

**Files:**
- Modify: `pyproject.toml:10-22`

- [ ] **Step 1: Add streamlit to pyproject.toml**

In `pyproject.toml`, add `"streamlit>=1.30"` to the `dependencies` list:

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
    "pandas>=2.0",
    "streamlit>=1.30",
]
```

- [ ] **Step 2: Install updated dependencies**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/pip install -e .`
Expected: Successfully installed streamlit and dependencies.

- [ ] **Step 3: Verify streamlit import**

Run: `.venv/bin/python -c "import streamlit; print(streamlit.__version__)"`
Expected: Prints version >= 1.30

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat(court-vision): add streamlit dependency for review UI"
```

---

### Task 2: Match Data JSON Round-Trip (review_data.py)

**Files:**
- Create: `src/court_vision/review_data.py`
- Create: `tests/test_review_data.py`

The existing `export_json` uses `dataclasses.asdict()` for serialization, but there's no deserialization path. We need `load_match_json` to reconstruct full `MatchData` with nested `Point`, `Shot`, and `ShotPlacement` objects from JSON, and `save_match_json` to persist review edits.

- [ ] **Step 1: Write failing tests for load_match_json and save_match_json**

Create `tests/test_review_data.py`:

```python
"""Tests for review data loading and saving."""

import json
from pathlib import Path

from court_vision.shot_classify import MatchData, Point, Shot, ShotPlacement


class TestLoadMatchJson:
    def test_load_empty_match(self, tmp_path: Path):
        """Load a match JSON with no points."""
        from court_vision.review_data import load_match_json

        match_file = tmp_path / "match.json"
        match_file.write_text(json.dumps({
            "match_id": "abc123",
            "source_url": "test.mp4",
            "metadata": {"players": ["near_player", "far_player"]},
            "points": [],
        }))

        match = load_match_json(match_file)

        assert isinstance(match, MatchData)
        assert match.match_id == "abc123"
        assert match.source_url == "test.mp4"
        assert match.points == []

    def test_load_match_with_shots(self, tmp_path: Path):
        """Load a match JSON with points and shots."""
        from court_vision.review_data import load_match_json

        match_file = tmp_path / "match.json"
        match_file.write_text(json.dumps({
            "match_id": "abc123",
            "source_url": "test.mp4",
            "metadata": {},
            "points": [
                {
                    "point_number": 1,
                    "start_frame": 0,
                    "end_frame": 100,
                    "start_time_s": 0.0,
                    "end_time_s": 3.33,
                    "server": "near_player",
                    "shots": [
                        {
                            "shot_number": 1,
                            "frame": 10,
                            "time_s": 0.33,
                            "player": "near_player",
                            "stroke": "serve",
                            "placement": {
                                "x": 1.5,
                                "y": 4.0,
                                "zone": "t",
                            },
                            "confidence": 0.8,
                        },
                    ],
                    "outcome": "winner",
                    "outcome_player": "near_player",
                    "rally_length": 1,
                    "review_status": "pending",
                },
            ],
        }))

        match = load_match_json(match_file)

        assert len(match.points) == 1
        point = match.points[0]
        assert isinstance(point, Point)
        assert point.point_number == 1
        assert point.review_status == "pending"
        assert len(point.shots) == 1
        shot = point.shots[0]
        assert isinstance(shot, Shot)
        assert shot.stroke == "serve"
        assert isinstance(shot.placement, ShotPlacement)
        assert shot.placement.zone == "t"

    def test_load_match_with_null_placement(self, tmp_path: Path):
        """Load a shot with null placement."""
        from court_vision.review_data import load_match_json

        match_file = tmp_path / "match.json"
        match_file.write_text(json.dumps({
            "match_id": "abc123",
            "source_url": "test.mp4",
            "metadata": {},
            "points": [
                {
                    "point_number": 1,
                    "start_frame": 0, "end_frame": 50,
                    "start_time_s": 0.0, "end_time_s": 1.67,
                    "server": None,
                    "shots": [
                        {
                            "shot_number": 1, "frame": 10, "time_s": 0.33,
                            "player": "near_player", "stroke": "forehand",
                            "placement": None, "confidence": 0.4,
                        },
                    ],
                    "outcome": None, "outcome_player": None,
                    "rally_length": 1, "review_status": "pending",
                },
            ],
        }))

        match = load_match_json(match_file)
        assert match.points[0].shots[0].placement is None


class TestSaveMatchJson:
    def test_save_and_reload(self, tmp_path: Path):
        """Save match data and reload it — round-trip test."""
        from court_vision.review_data import load_match_json, save_match_json

        match = MatchData(
            match_id="test123",
            source_url="test.mp4",
            metadata={"players": ["near_player", "far_player"]},
            points=[
                Point(
                    point_number=1,
                    start_frame=0, end_frame=100,
                    start_time_s=0.0, end_time_s=3.33,
                    server="near_player",
                    shots=[
                        Shot(
                            shot_number=1, frame=10, time_s=0.33,
                            player="near_player", stroke="serve",
                            placement=ShotPlacement(x=1.5, y=4.0, zone="t"),
                            confidence=0.8,
                        ),
                    ],
                    outcome="winner", outcome_player="near_player",
                    rally_length=1, review_status="approved",
                ),
            ],
        )

        output_file = tmp_path / "output.json"
        save_match_json(match, output_file)
        reloaded = load_match_json(output_file)

        assert reloaded.match_id == "test123"
        assert reloaded.points[0].review_status == "approved"
        assert reloaded.points[0].shots[0].stroke == "serve"
        assert reloaded.points[0].shots[0].placement.zone == "t"

    def test_save_preserves_review_status(self, tmp_path: Path):
        """Save preserves updated review_status values."""
        from court_vision.review_data import load_match_json, save_match_json

        match = MatchData(
            match_id="test456",
            source_url="test.mp4",
            metadata={},
            points=[
                Point(
                    point_number=1,
                    start_frame=0, end_frame=50,
                    start_time_s=0.0, end_time_s=1.67,
                    server=None,
                    shots=[],
                    outcome=None, outcome_player=None,
                    rally_length=0, review_status="corrected",
                ),
            ],
        )

        output_file = tmp_path / "output.json"
        save_match_json(match, output_file)

        with open(output_file) as f:
            raw = json.load(f)

        assert raw["points"][0]["review_status"] == "corrected"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_review_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'court_vision.review_data'`

- [ ] **Step 3: Implement review_data.py**

Create `src/court_vision/review_data.py`:

```python
"""Review data — load and save match JSON with full deserialization."""

import json
from dataclasses import asdict
from pathlib import Path

from court_vision.shot_classify import MatchData, Point, Shot, ShotPlacement


def load_match_json(path: Path) -> MatchData:
    """Load a match JSON file and deserialize into MatchData.

    Reconstructs the full object hierarchy: MatchData -> Point -> Shot -> ShotPlacement.

    Args:
        path: Path to match JSON file.

    Returns:
        Fully deserialized MatchData.
    """
    with open(path) as f:
        raw = json.load(f)

    points: list[Point] = []
    for p in raw.get("points", []):
        shots: list[Shot] = []
        for s in p.get("shots", []):
            placement = None
            if s.get("placement") is not None:
                pl = s["placement"]
                placement = ShotPlacement(x=pl["x"], y=pl["y"], zone=pl["zone"])

            shots.append(Shot(
                shot_number=s["shot_number"],
                frame=s["frame"],
                time_s=s["time_s"],
                player=s["player"],
                stroke=s["stroke"],
                placement=placement,
                confidence=s["confidence"],
            ))

        points.append(Point(
            point_number=p["point_number"],
            start_frame=p["start_frame"],
            end_frame=p["end_frame"],
            start_time_s=p["start_time_s"],
            end_time_s=p["end_time_s"],
            server=p.get("server"),
            shots=shots,
            outcome=p.get("outcome"),
            outcome_player=p.get("outcome_player"),
            rally_length=p["rally_length"],
            review_status=p.get("review_status", "pending"),
        ))

    return MatchData(
        match_id=raw["match_id"],
        source_url=raw["source_url"],
        metadata=raw.get("metadata", {}),
        points=points,
    )


def save_match_json(match: MatchData, path: Path) -> None:
    """Save MatchData to JSON file.

    Args:
        match: Match data to save.
        path: Output file path.
    """
    data = asdict(match)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_review_data.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/review_data.py tests/test_review_data.py
git commit -m "feat(court-vision): add match data JSON round-trip for review workflow"
```

---

### Task 3: Frame Overlay Rendering (overlay.py)

**Files:**
- Create: `src/court_vision/overlay.py`
- Create: `tests/test_overlay.py`

Renders visual overlays onto video frames using OpenCV: ball position (circle), player bounding boxes (rectangles with labels), and pose skeleton (connected keypoints).

- [ ] **Step 1: Write failing tests**

Create `tests/test_overlay.py`:

```python
"""Tests for frame overlay rendering."""

import numpy as np

from court_vision.ball_tracker import BallDetection
from court_vision.player_detect import PlayerDetection, PoseKeypoints


class TestDrawBall:
    def test_draws_circle_on_frame(self):
        """draw_ball renders a circle at the ball position."""
        from court_vision.overlay import draw_ball

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ball = BallDetection(frame_index=0, x=320.0, y=240.0, confidence=0.9)

        result = draw_ball(frame, ball)

        # Check that pixels near the ball center are non-zero (circle drawn)
        assert result[240, 320].sum() > 0

    def test_returns_copy(self):
        """draw_ball does not modify the original frame."""
        from court_vision.overlay import draw_ball

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ball = BallDetection(frame_index=0, x=320.0, y=240.0, confidence=0.9)

        result = draw_ball(frame, ball)

        assert frame[240, 320].sum() == 0  # Original unchanged
        assert result is not frame


class TestDrawPlayers:
    def test_draws_bounding_box(self):
        """draw_players renders bounding boxes."""
        from court_vision.overlay import draw_players

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        players = [
            PlayerDetection(
                frame_index=0,
                bbox=(100.0, 200.0, 200.0, 400.0),
                confidence=0.9,
                role="near_player",
            ),
        ]

        result = draw_players(frame, players)

        # Check that pixels along the top edge of bbox are non-zero
        assert result[200, 150].sum() > 0


class TestDrawPoses:
    def test_draws_keypoints(self):
        """draw_poses renders keypoint dots."""
        from court_vision.overlay import draw_poses

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = [
            PoseKeypoints(
                frame_index=0,
                role="near_player",
                keypoints={
                    "nose": (320.0, 100.0, 0.9),
                    "left_shoulder": (300.0, 150.0, 0.8),
                    "right_shoulder": (340.0, 150.0, 0.8),
                },
            ),
        ]

        result = draw_poses(frame, poses)

        # Check that pixels at nose position are non-zero
        assert result[100, 320].sum() > 0


class TestRenderOverlay:
    def test_combines_all_overlays(self):
        """render_overlay applies ball, player, and pose overlays."""
        from court_vision.overlay import render_overlay
        from court_vision.player_detect import FrameTrackingResult

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracking = FrameTrackingResult(
            frame_index=0,
            ball=BallDetection(frame_index=0, x=320.0, y=240.0, confidence=0.9),
            players=[
                PlayerDetection(
                    frame_index=0, bbox=(100.0, 200.0, 200.0, 400.0),
                    confidence=0.9, role="near_player",
                ),
            ],
            poses=[
                PoseKeypoints(
                    frame_index=0, role="near_player",
                    keypoints={"nose": (320.0, 100.0, 0.9)},
                ),
            ],
        )

        result = render_overlay(frame, tracking)

        assert result.shape == frame.shape
        assert result.sum() > 0  # Something was drawn

    def test_handles_no_ball(self):
        """render_overlay handles tracking with no ball detection."""
        from court_vision.overlay import render_overlay
        from court_vision.player_detect import FrameTrackingResult

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracking = FrameTrackingResult(
            frame_index=0, ball=None, players=[], poses=[],
        )

        result = render_overlay(frame, tracking)

        assert result.shape == frame.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'court_vision.overlay'`

- [ ] **Step 3: Implement overlay.py**

Create `src/court_vision/overlay.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_overlay.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/overlay.py tests/test_overlay.py
git commit -m "feat(court-vision): add frame overlay rendering for review UI"
```

---

### Task 4: Streamlit Review App — Point List & Frame Viewer

**Files:**
- Create: `src/court_vision/review_app.py`

This is the main Streamlit app. This task builds the core layout: sidebar with point list (color-coded by review status + confidence), and frame viewer with overlays. The app uses `st.session_state` to track the currently selected point and frame.

**Note:** Streamlit apps are not unit-tested with pytest — they require the Streamlit testing framework or manual testing. We'll verify by importing and checking for syntax errors, and test the CLI command that launches it.

- [ ] **Step 1: Create the Streamlit app**

Create `src/court_vision/review_app.py`:

```python
"""Streamlit Review UI for Court Vision match data."""

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from court_vision.overlay import render_overlay
from court_vision.player_detect import FrameTrackingResult
from court_vision.review_data import load_match_json, save_match_json
from court_vision.shot_classify import MatchData

CONFIDENCE_THRESHOLD = 0.7
STROKE_OPTIONS = ["forehand", "backhand", "serve", "volley", "overhead", "slice"]
OUTCOME_OPTIONS = ["winner", "error", "unforced_error"]
STATUS_OPTIONS = ["pending", "approved", "corrected"]
STATUS_COLORS = {
    "pending": "#FFA500",
    "approved": "#00CC00",
    "corrected": "#3399FF",
}


def load_tracking_data(tracking_path: Path) -> dict[int, FrameTrackingResult]:
    """Load per-frame tracking data from JSON if available."""
    if not tracking_path.exists():
        return {}
    with open(tracking_path) as f:
        raw = json.load(f)
    # Tracking data is stored as a list of frame results
    from court_vision.ball_tracker import BallDetection
    from court_vision.player_detect import PlayerDetection, PoseKeypoints

    results: dict[int, FrameTrackingResult] = {}
    for entry in raw:
        ball = None
        if entry.get("ball"):
            b = entry["ball"]
            ball = BallDetection(
                frame_index=b["frame_index"], x=b["x"], y=b["y"],
                confidence=b["confidence"],
                interpolated=b.get("interpolated", False),
            )
        players = [
            PlayerDetection(
                frame_index=p["frame_index"], bbox=tuple(p["bbox"]),
                confidence=p["confidence"],
                court_position=tuple(p["court_position"]) if p.get("court_position") else None,
                role=p.get("role"),
            )
            for p in entry.get("players", [])
        ]
        poses = [
            PoseKeypoints(
                frame_index=pk["frame_index"], role=pk["role"],
                keypoints={k: tuple(v) for k, v in pk["keypoints"].items()},
            )
            for pk in entry.get("poses", [])
        ]
        frame_idx = entry["frame_index"]
        results[frame_idx] = FrameTrackingResult(
            frame_index=frame_idx, ball=ball, players=players, poses=poses,
        )
    return results


def get_frame_image(frames_dir: Path, frame_index: int) -> np.ndarray | None:
    """Load a frame image from the frames directory."""
    frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
    if not frame_path.exists():
        return None
    return cv2.imread(str(frame_path))


def point_confidence(match: MatchData, point_idx: int) -> float:
    """Get the minimum shot confidence for a point."""
    point = match.points[point_idx]
    if not point.shots:
        return 1.0
    return min(s.confidence for s in point.shots)


def run_app() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="Court Vision Review", layout="wide")
    st.title("Court Vision — Match Review")

    # --- Sidebar: File Loading ---
    with st.sidebar:
        st.header("Match Data")
        match_path = st.text_input("Match JSON path", value=st.session_state.get("match_path", ""))
        frames_dir = st.text_input("Frames directory", value=st.session_state.get("frames_dir", ""))
        tracking_path = st.text_input("Tracking JSON (optional)", value=st.session_state.get("tracking_path", ""))

        if st.button("Load Match"):
            if match_path and Path(match_path).exists():
                st.session_state["match_path"] = match_path
                st.session_state["match"] = load_match_json(Path(match_path))
                st.session_state["frames_dir"] = frames_dir
                st.session_state["tracking_path"] = tracking_path
                if tracking_path and Path(tracking_path).exists():
                    st.session_state["tracking"] = load_tracking_data(Path(tracking_path))
                else:
                    st.session_state["tracking"] = {}
                st.session_state["selected_point"] = 0
                st.rerun()
            else:
                st.error("Match JSON file not found.")

    match: MatchData | None = st.session_state.get("match")
    if match is None:
        st.info("Load a match JSON file to begin review.")
        return

    # --- Sidebar: Point List ---
    with st.sidebar:
        st.header("Points")

        # Bulk actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve High-Conf"):
                for point in match.points:
                    if point.review_status == "pending":
                        min_conf = min((s.confidence for s in point.shots), default=1.0)
                        if min_conf >= CONFIDENCE_THRESHOLD:
                            point.review_status = "approved"
                st.rerun()
        with col2:
            if st.button("Save"):
                save_match_json(match, Path(st.session_state["match_path"]))
                st.success("Saved!")

        # Skip to next low-confidence
        if st.button("Next Low-Conf"):
            current = st.session_state.get("selected_point", 0)
            for i in range(current + 1, len(match.points)):
                if point_confidence(match, i) < CONFIDENCE_THRESHOLD:
                    st.session_state["selected_point"] = i
                    st.rerun()
            st.warning("No more low-confidence points.")

        # Status summary
        pending = sum(1 for p in match.points if p.review_status == "pending")
        approved = sum(1 for p in match.points if p.review_status == "approved")
        corrected = sum(1 for p in match.points if p.review_status == "corrected")
        st.caption(f"Pending: {pending} | Approved: {approved} | Corrected: {corrected}")

        # Point list
        for i, point in enumerate(match.points):
            min_conf = point_confidence(match, i)
            color = STATUS_COLORS.get(point.review_status, "#999")
            conf_marker = " ⚠" if min_conf < CONFIDENCE_THRESHOLD else ""
            label = f"Point {point.point_number} [{point.review_status}]{conf_marker}"
            if st.button(label, key=f"point_{i}", use_container_width=True):
                st.session_state["selected_point"] = i
                st.rerun()

    # --- Main Area ---
    selected_idx = st.session_state.get("selected_point", 0)
    if selected_idx >= len(match.points):
        selected_idx = 0
    point = match.points[selected_idx]

    st.subheader(f"Point {point.point_number}")
    st.caption(f"Frames {point.start_frame}–{point.end_frame} | "
               f"{point.start_time_s:.1f}s–{point.end_time_s:.1f}s | "
               f"Rally: {point.rally_length} shots")

    # Frame viewer
    f_dir = st.session_state.get("frames_dir", "")
    tracking_data: dict[int, FrameTrackingResult] = st.session_state.get("tracking", {})

    if f_dir and Path(f_dir).exists():
        frame_num = st.slider(
            "Frame", min_value=point.start_frame, max_value=point.end_frame,
            value=point.start_frame, key=f"frame_slider_{selected_idx}",
        )

        frame_img = get_frame_image(Path(f_dir), frame_num)
        if frame_img is not None:
            # Apply overlay if tracking data exists
            if frame_num in tracking_data:
                frame_img = render_overlay(frame_img, tracking_data[frame_num])
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)
        else:
            st.warning(f"Frame {frame_num} not found in {f_dir}")
    else:
        st.info("Set frames directory to view video frames.")

    # --- Shot Editor ---
    st.subheader("Shots")

    changed = False
    for j, shot in enumerate(point.shots):
        with st.expander(f"Shot {shot.shot_number}: {shot.stroke} by {shot.player} "
                         f"(conf: {shot.confidence:.2f})", expanded=True):
            col_stroke, col_outcome = st.columns(2)
            with col_stroke:
                new_stroke = st.selectbox(
                    "Stroke", STROKE_OPTIONS,
                    index=STROKE_OPTIONS.index(shot.stroke) if shot.stroke in STROKE_OPTIONS else 0,
                    key=f"stroke_{selected_idx}_{j}",
                )
                if new_stroke != shot.stroke:
                    shot.stroke = new_stroke
                    changed = True

            with col_outcome:
                if j == len(point.shots) - 1:  # Last shot determines outcome
                    current_outcome = point.outcome or "winner"
                    new_outcome = st.selectbox(
                        "Point Outcome", OUTCOME_OPTIONS,
                        index=OUTCOME_OPTIONS.index(current_outcome) if current_outcome in OUTCOME_OPTIONS else 0,
                        key=f"outcome_{selected_idx}_{j}",
                    )
                    if new_outcome != point.outcome:
                        point.outcome = new_outcome
                        changed = True

            if shot.placement:
                st.caption(f"Placement: ({shot.placement.x:.1f}, {shot.placement.y:.1f}) — {shot.placement.zone}")

            # Jump to contact frame
            if st.button(f"Go to frame {shot.frame}", key=f"goto_{selected_idx}_{j}"):
                st.session_state[f"frame_slider_{selected_idx}"] = shot.frame
                st.rerun()

    # Review status
    st.subheader("Review Status")
    new_status = st.selectbox(
        "Status", STATUS_OPTIONS,
        index=STATUS_OPTIONS.index(point.review_status) if point.review_status in STATUS_OPTIONS else 0,
        key=f"status_{selected_idx}",
    )
    if new_status != point.review_status:
        point.review_status = new_status
        changed = True

    if changed:
        if point.review_status == "pending":
            point.review_status = "corrected"

    # Navigation
    col_prev, col_next = st.columns(2)
    with col_prev:
        if selected_idx > 0:
            if st.button("← Previous Point"):
                st.session_state["selected_point"] = selected_idx - 1
                st.rerun()
    with col_next:
        if selected_idx < len(match.points) - 1:
            if st.button("Next Point →"):
                st.session_state["selected_point"] = selected_idx + 1
                st.rerun()


if __name__ == "__main__":
    run_app()
```

- [ ] **Step 2: Verify the module imports without errors**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -c "from court_vision.review_app import run_app; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/court_vision/review_app.py
git commit -m "feat(court-vision): add Streamlit review app with point list, frame viewer, and shot editor"
```

---

### Task 5: CLI Review Command

**Files:**
- Modify: `src/court_vision/cli.py:1-91`
- Modify: `tests/test_cli.py`

Add a `review` command to the CLI that launches the Streamlit app with the match JSON and frames directory pre-populated via query params.

- [ ] **Step 1: Write failing test for the review command**

Add to `tests/test_cli.py`:

```python
class TestReviewCommand:
    @patch("court_vision.cli.subprocess.run")
    def test_review_launches_streamlit(self, mock_run: MagicMock, tmp_path: Path):
        """Review command calls streamlit run with the correct arguments."""
        match_file = tmp_path / "match.json"
        match_file.write_text('{"match_id": "test"}')

        result = runner.invoke(app, ["review", str(match_file)])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "streamlit" in call_args[0] or "streamlit" in str(call_args)
```

Also add `import subprocess` to the existing imports and add `subprocess` to the `from unittest.mock import ...` line (it already imports `MagicMock, patch`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_cli.py::TestReviewCommand -v`
Expected: FAIL

- [ ] **Step 3: Add the review command to cli.py**

Add to `src/court_vision/cli.py` before the `version` command:

```python
@app.command()
def review(
    match_json: Path = typer.Argument(help="Path to match data JSON file."),
    frames_dir: Optional[Path] = typer.Option(None, "--frames", help="Path to extracted frames directory."),
    tracking_json: Optional[Path] = typer.Option(None, "--tracking", help="Path to tracking data JSON."),
) -> None:
    """Launch the Streamlit review UI for a match."""
    import subprocess
    import sys

    if not match_json.exists():
        typer.echo(f"Error: {match_json} not found.", err=True)
        raise typer.Exit(1)

    app_path = Path(__file__).parent / "review_app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--", str(match_json),
    ]

    if frames_dir:
        cmd.extend(["--frames-dir", str(frames_dir)])
    if tracking_json:
        cmd.extend(["--tracking", str(tracking_json)])

    typer.echo(f"Launching review UI for {match_json}...")
    subprocess.run(cmd)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_cli.py::TestReviewCommand -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest -v`
Expected: All tests pass (previous 129 + new tests)

- [ ] **Step 6: Commit**

```bash
git add src/court_vision/cli.py tests/test_cli.py
git commit -m "feat(court-vision): add CLI review command to launch Streamlit UI"
```

---

### Task 6: Full Test Suite Verification

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest -v`
Expected: All tests pass.

- [ ] **Step 2: Verify all imports work**

Run:
```bash
cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -c "
from court_vision.review_data import load_match_json, save_match_json
from court_vision.overlay import draw_ball, draw_players, draw_poses, render_overlay
from court_vision.review_app import run_app
print('All Phase 2D imports OK')
"
```
Expected: `All Phase 2D imports OK`

- [ ] **Step 3: Verify streamlit can locate the app**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m streamlit --version`
Expected: Prints streamlit version.

---

## Self-Review Checklist

1. **Spec coverage:** Review data round-trip ✅, frame overlay ✅, point list with confidence highlighting ✅, shot editor ✅, bulk actions (approve high-confidence) ✅, skip to low-confidence ✅, status tracking (pending/approved/corrected) ✅, save edits ✅, CLI review command ✅.

2. **Placeholder scan:** No TBDs, no "implement later", no "similar to Task N". All code blocks are complete.

3. **Type consistency:** `MatchData`, `Point`, `Shot`, `ShotPlacement` used consistently. `load_match_json`/`save_match_json` names consistent across tests and implementation. `render_overlay`/`draw_ball`/`draw_players`/`draw_poses` names consistent.
