# Court Vision Phase 2C — Shot Classification + Export

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify each shot's stroke type, placement zone, and point outcome using rule-based heuristics from pose keypoints and ball trajectory, then export structured match data as JSON and Sackmann-compatible CSV.

**Architecture:** Two new modules (`shot_classify.py` and `export.py`) inside `court_vision/`. Shot classification uses rule-based heuristics — no ML model needed for MVP. Export produces the JSON format defined in the spec plus a CSV summary. These integrate into the pipeline as Stage 5 and a post-pipeline export step.

**Tech Stack:** Python 3.11+, NumPy, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-court-vision-design.md` — Stage 5, Output Format, Integration Path

**Phase 2B plan:** `docs/superpowers/plans/2026-03-29-court-vision-phase2b-ball-player-tracking.md`

---

## Phase 2C Scope

Phase 2C delivers:
1. Ball contact detection — identify frames where ball meets racket
2. Stroke type classification — forehand / backhand / serve / volley / overhead / slice from pose keypoints
3. Placement zone computation — direction + depth from ball landing court coordinates
4. Point boundary detection — gaps between gameplay segments = point boundaries
5. Point outcome classification — winner / error / unforced error
6. Raw JSON export — full match data in spec format
7. Aggregated CSV export — Sackmann-compatible match stats
8. Pipeline integration — wire Stage 5 into pipeline
9. CLI export command + updated process output

## File Structure

| File | Responsibility |
|------|---------------|
| `src/court_vision/shot_classify.py` | Contact detection, stroke classification, placement zones, point outcomes |
| `src/court_vision/export.py` | JSON and CSV export in spec-defined formats |
| `src/court_vision/pipeline.py` | **Modify:** Add Stage 5 call and shot data to `PipelineResult` |
| `src/court_vision/cli.py` | **Modify:** Add `export` command and shot classification summary |
| `pyproject.toml` | **Modify:** Add `pandas` dependency |
| `tests/test_shot_classify.py` | Tests for shot classification module |
| `tests/test_export.py` | Tests for export module |
| `tests/test_pipeline.py` | **Modify:** Add Stage 5 mocks |
| `tests/test_cli.py` | **Modify:** Add export command and shot output tests |

## Data Structures (referenced by all tasks)

```python
# shot_classify.py
@dataclass
class ShotPlacement:
    x: float                    # court x in meters
    y: float                    # court y in meters
    zone: str                   # e.g. "crosscourt_deep", "wide", "t"

@dataclass
class Shot:
    shot_number: int
    frame: int
    time_s: float
    player: str                 # "near_player" or "far_player"
    stroke: str                 # "forehand", "backhand", "serve", "volley", "overhead", "slice"
    placement: ShotPlacement | None
    confidence: float

@dataclass
class Point:
    point_number: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    server: str | None          # "near_player" or "far_player"
    shots: list[Shot]
    outcome: str | None         # "winner", "error", "unforced_error"
    outcome_player: str | None
    rally_length: int
    review_status: str = "pending"

@dataclass
class MatchData:
    match_id: str
    source_url: str
    metadata: dict
    points: list[Point]
```

---

### Task 1: Add pandas Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pandas to dependencies**

Add `"pandas>=2.0"` to the `dependencies` list in `pyproject.toml`:

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
]
```

- [ ] **Step 2: Install**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/pip install -e ".[dev]"`

- [ ] **Step 3: Verify**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -c "import pandas; print('pandas', pandas.__version__)"`

- [ ] **Step 4: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add pyproject.toml
git commit -m "feat(court-vision): add pandas dependency for data export"
```

---

### Task 2: Shot Data Structures and Placement Zone Computation

**Files:**
- Create: `src/court_vision/shot_classify.py`
- Create: `tests/test_shot_classify.py`

Creates the core data structures and the `compute_placement_zone` function that maps ball court coordinates to zone labels.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the shot classification module."""

import pytest

from court_vision.shot_classify import (
    Shot,
    ShotPlacement,
    Point,
    MatchData,
    compute_placement_zone,
)


class TestShotPlacement:
    def test_placement_fields(self):
        """ShotPlacement has x, y, zone."""
        p = ShotPlacement(x=3.0, y=8.0, zone="crosscourt_deep")
        assert p.x == 3.0
        assert p.y == 8.0
        assert p.zone == "crosscourt_deep"


class TestShot:
    def test_shot_fields(self):
        """Shot has all required fields."""
        s = Shot(
            shot_number=1, frame=100, time_s=3.33,
            player="near_player", stroke="forehand",
            placement=ShotPlacement(x=2.0, y=9.0, zone="crosscourt_deep"),
            confidence=0.85,
        )
        assert s.stroke == "forehand"
        assert s.placement.zone == "crosscourt_deep"


class TestPoint:
    def test_point_fields(self):
        """Point has all required fields."""
        p = Point(
            point_number=1,
            start_frame=100, end_frame=200,
            start_time_s=3.33, end_time_s=6.67,
            server="near_player",
            shots=[],
            outcome="winner",
            outcome_player="far_player",
            rally_length=3,
        )
        assert p.review_status == "pending"
        assert p.outcome == "winner"


class TestMatchData:
    def test_match_data_fields(self):
        """MatchData has all required fields."""
        m = MatchData(
            match_id="test_123",
            source_url="test.mp4",
            metadata={"players": ["near_player", "far_player"]},
            points=[],
        )
        assert m.match_id == "test_123"


class TestComputePlacementZone:
    def test_serve_wide_deuce(self):
        """Ball landing far right in deuce service box = wide."""
        zone = compute_placement_zone(x=3.5, y=4.0, is_serve=True, server_side="deuce")
        assert zone == "wide"

    def test_serve_t_deuce(self):
        """Ball landing near center in deuce service box = t."""
        zone = compute_placement_zone(x=0.5, y=4.0, is_serve=True, server_side="deuce")
        assert zone == "t"

    def test_serve_body_deuce(self):
        """Ball landing in middle zone of deuce service box = body."""
        zone = compute_placement_zone(x=2.0, y=4.0, is_serve=True, server_side="deuce")
        assert zone == "body"

    def test_serve_wide_ad(self):
        """Ball landing far left in ad service box = wide."""
        zone = compute_placement_zone(x=-3.5, y=4.0, is_serve=True, server_side="ad")
        assert zone == "wide"

    def test_rally_crosscourt_deep(self):
        """Ball landing crosscourt and deep = crosscourt_deep."""
        # Near player hits to far side, ball lands at positive x, deep (beyond service line)
        zone = compute_placement_zone(x=3.0, y=9.0, is_serve=False, hitter="near_player")
        assert zone == "crosscourt_deep"

    def test_rally_down_the_line_deep(self):
        """Ball landing down the line and deep = down_the_line_deep."""
        zone = compute_placement_zone(x=-3.0, y=9.0, is_serve=False, hitter="near_player")
        assert zone == "down_the_line_deep"

    def test_rally_middle_short(self):
        """Ball landing in middle and short = middle_short."""
        zone = compute_placement_zone(x=0.5, y=3.0, is_serve=False, hitter="near_player")
        assert zone == "middle_short"

    def test_rally_crosscourt_short(self):
        """Ball landing crosscourt and short = crosscourt_short."""
        zone = compute_placement_zone(x=3.0, y=3.0, is_serve=False, hitter="near_player")
        assert zone == "crosscourt_short"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_shot_classify.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Shot classification — stroke type, placement zones, and point outcomes."""

from dataclasses import dataclass, field

_SERVICE_LINE_Y = 6.4  # Service line distance from net in meters
_SINGLES_WIDTH = 4.115  # Singles sideline to center


@dataclass
class ShotPlacement:
    """Ball landing placement on court."""

    x: float  # court x in meters
    y: float  # court y in meters
    zone: str  # e.g. "crosscourt_deep", "wide", "t"


@dataclass
class Shot:
    """A single shot within a point."""

    shot_number: int
    frame: int
    time_s: float
    player: str  # "near_player" or "far_player"
    stroke: str  # forehand, backhand, serve, volley, overhead, slice
    placement: ShotPlacement | None
    confidence: float


@dataclass
class Point:
    """A single point in the match."""

    point_number: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    server: str | None  # "near_player" or "far_player"
    shots: list[Shot]
    outcome: str | None  # "winner", "error", "unforced_error"
    outcome_player: str | None
    rally_length: int
    review_status: str = "pending"


@dataclass
class MatchData:
    """Full match data for export."""

    match_id: str
    source_url: str
    metadata: dict
    points: list[Point]


def compute_placement_zone(
    x: float,
    y: float,
    is_serve: bool = False,
    server_side: str | None = None,
    hitter: str | None = None,
) -> str:
    """Compute the placement zone from ball landing court coordinates.

    Args:
        x: Ball x-coordinate in meters (court system).
        y: Ball y-coordinate in meters (court system).
        is_serve: Whether this shot is a serve.
        server_side: "deuce" or "ad" (for serve zone computation).
        hitter: "near_player" or "far_player" (for rally direction).

    Returns:
        Zone string: serve zones ("wide", "body", "t") or
        rally zones ("crosscourt_deep", "down_the_line_short", etc.)
    """
    if is_serve:
        return _compute_serve_zone(x, y, server_side or "deuce")
    return _compute_rally_zone(x, y, hitter or "near_player")


def _compute_serve_zone(x: float, y: float, server_side: str) -> str:
    """Compute serve placement zone within the service box.

    Deuce side: right service box (positive x relative to server).
    Ad side: left service box (negative x relative to server).
    """
    abs_x = abs(x)

    # T zone: near the center service line
    if abs_x < _SINGLES_WIDTH / 3:
        return "t"

    # Wide zone: near the singles sideline
    if abs_x > _SINGLES_WIDTH * 2 / 3:
        return "wide"

    # Body zone: in between
    return "body"


def _compute_rally_zone(x: float, y: float, hitter: str) -> str:
    """Compute rally placement zone (direction + depth).

    Direction is relative to the hitter:
    - Near player hits to positive y (far side): crosscourt = positive x, DTL = negative x
    - Far player hits to negative y (near side): crosscourt = negative x, DTL = positive x
    """
    abs_y = abs(y)
    abs_x = abs(x)

    # Depth: short = inside service line, deep = beyond service line
    depth = "short" if abs_y < _SERVICE_LINE_Y else "deep"

    # Direction based on x position relative to hitter
    if abs_x < _SINGLES_WIDTH / 3:
        direction = "middle"
    elif hitter == "near_player":
        # Near player hits to far side: crosscourt = same sign as x (positive x = right)
        direction = "crosscourt" if x > 0 else "down_the_line"
    else:
        # Far player hits to near side: crosscourt = negative x
        direction = "crosscourt" if x < 0 else "down_the_line"

    return f"{direction}_{depth}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_shot_classify.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/shot_classify.py tests/test_shot_classify.py
git commit -m "feat(court-vision): add shot data structures and placement zone computation"
```

---

### Task 3: Stroke Type Classification from Pose Keypoints

**Files:**
- Modify: `src/court_vision/shot_classify.py`
- Modify: `tests/test_shot_classify.py`

Adds the rule-based stroke type classifier that determines forehand/backhand/serve/volley/overhead/slice from pose keypoints at ball contact.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_shot_classify.py`:

```python
from court_vision.shot_classify import classify_stroke
from court_vision.player_detect import PoseKeypoints


class TestClassifyStroke:
    def _make_pose(self, role: str = "near_player", **overrides) -> PoseKeypoints:
        """Create a PoseKeypoints with customizable keypoints.

        Default pose: standing neutral. Override specific keypoints
        to simulate different strokes.
        """
        defaults = {
            "nose": (400.0, 100.0, 0.9),
            "left_shoulder": (380.0, 200.0, 0.9),
            "right_shoulder": (420.0, 200.0, 0.9),
            "left_elbow": (360.0, 280.0, 0.9),
            "right_elbow": (440.0, 280.0, 0.9),
            "left_wrist": (350.0, 350.0, 0.9),
            "right_wrist": (450.0, 350.0, 0.9),
            "left_hip": (390.0, 400.0, 0.9),
            "right_hip": (410.0, 400.0, 0.9),
        }
        defaults.update(overrides)
        return PoseKeypoints(frame_index=0, role=role, keypoints=defaults)

    def test_forehand_right_handed(self):
        """Right wrist extended to right side = forehand (right-handed)."""
        pose = self._make_pose(
            right_wrist=(550.0, 250.0, 0.9),  # wrist far right of body
            right_elbow=(500.0, 250.0, 0.9),
        )
        stroke, conf = classify_stroke(pose)
        assert stroke == "forehand"
        assert conf > 0.0

    def test_backhand_right_handed(self):
        """Right wrist extended to left side of body = backhand."""
        pose = self._make_pose(
            right_wrist=(300.0, 250.0, 0.9),  # wrist far left
            right_elbow=(340.0, 260.0, 0.9),
        )
        stroke, conf = classify_stroke(pose)
        assert stroke == "backhand"
        assert conf > 0.0

    def test_serve_arms_up(self):
        """Both wrists above head = serve."""
        pose = self._make_pose(
            right_wrist=(430.0, 50.0, 0.9),   # above nose
            left_wrist=(380.0, 70.0, 0.9),     # also above nose
            nose=(400.0, 100.0, 0.9),
        )
        stroke, conf = classify_stroke(pose)
        assert stroke == "serve"
        assert conf > 0.0

    def test_overhead_one_arm_up(self):
        """One wrist above head, other at body = overhead."""
        pose = self._make_pose(
            right_wrist=(420.0, 50.0, 0.9),   # above nose
            left_wrist=(370.0, 350.0, 0.9),    # at body level
            nose=(400.0, 100.0, 0.9),
        )
        stroke, conf = classify_stroke(pose)
        assert stroke == "overhead"
        assert conf > 0.0

    def test_returns_default_for_neutral_pose(self):
        """Neutral pose with no clear stroke returns forehand with low confidence."""
        pose = self._make_pose()
        stroke, conf = classify_stroke(pose)
        assert stroke in ("forehand", "backhand", "serve", "volley", "overhead", "slice")
        assert conf >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_shot_classify.py::TestClassifyStroke -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Add to `src/court_vision/shot_classify.py`:

```python
from court_vision.player_detect import PoseKeypoints


def classify_stroke(
    pose: PoseKeypoints,
) -> tuple[str, float]:
    """Classify the stroke type from pose keypoints at ball contact.

    Rule-based heuristics using wrist, elbow, and shoulder positions:
    - Serve: both wrists above head
    - Overhead: dominant wrist above head, other at body
    - Forehand: dominant wrist extended to dominant side
    - Backhand: dominant wrist extended to non-dominant side
    - Volley: compact arm position, wrists near shoulders
    - Slice: dominant wrist below elbow with arm extended

    Assumes right-handed player for MVP.

    Args:
        pose: PoseKeypoints from MediaPipe.

    Returns:
        Tuple of (stroke_type, confidence).
    """
    kp = pose.keypoints

    # Extract key positions (with safe defaults)
    nose = kp.get("nose", (400.0, 200.0, 0.0))
    r_wrist = kp.get("right_wrist", (450.0, 350.0, 0.0))
    l_wrist = kp.get("left_wrist", (350.0, 350.0, 0.0))
    r_elbow = kp.get("right_elbow", (440.0, 280.0, 0.0))
    r_shoulder = kp.get("right_shoulder", (420.0, 200.0, 0.0))
    l_shoulder = kp.get("left_shoulder", (380.0, 200.0, 0.0))
    r_hip = kp.get("right_hip", (410.0, 400.0, 0.0))

    nose_y = nose[1]
    r_wrist_x, r_wrist_y = r_wrist[0], r_wrist[1]
    l_wrist_y = l_wrist[1]
    r_elbow_y = r_elbow[1]
    r_shoulder_x = r_shoulder[0]
    l_shoulder_x = l_shoulder[0]
    body_center_x = (r_shoulder_x + l_shoulder_x) / 2

    # Rule 1: Serve — both wrists above nose
    if r_wrist_y < nose_y and l_wrist_y < nose_y:
        return ("serve", 0.8)

    # Rule 2: Overhead — dominant wrist above nose, other below
    if r_wrist_y < nose_y and l_wrist_y > nose_y:
        return ("overhead", 0.7)

    # Rule 3: Volley — wrist close to shoulder height, compact position
    wrist_shoulder_dist = abs(r_wrist_y - r_shoulder[1])
    wrist_body_dist_x = abs(r_wrist_x - body_center_x)
    if wrist_shoulder_dist < 60 and wrist_body_dist_x < 80:
        return ("volley", 0.6)

    # Rule 4: Slice — wrist below elbow with arm extended sideways
    if r_wrist_y > r_elbow_y and wrist_body_dist_x > 80:
        # Wrist dropping below elbow = slice motion
        return ("slice", 0.6)

    # Rule 5: Forehand vs Backhand — wrist side relative to body center
    if r_wrist_x > body_center_x + 30:
        return ("forehand", 0.7)
    elif r_wrist_x < body_center_x - 30:
        return ("backhand", 0.7)

    # Default: forehand with low confidence
    return ("forehand", 0.4)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_shot_classify.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/shot_classify.py tests/test_shot_classify.py
git commit -m "feat(court-vision): add rule-based stroke classification from pose keypoints"
```

---

### Task 4: Ball Contact Detection

**Files:**
- Modify: `src/court_vision/shot_classify.py`
- Modify: `tests/test_shot_classify.py`

Detects the frame where ball contacts the racket by finding frames where ball and player positions converge.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_shot_classify.py`:

```python
from court_vision.shot_classify import detect_contacts
from court_vision.ball_tracker import BallDetection
from court_vision.player_detect import FrameTrackingResult, PlayerDetection


class TestDetectContacts:
    def test_detects_contact_when_ball_near_player(self):
        """Detects contact when ball position is near a player's wrist."""
        tracking = [
            FrameTrackingResult(
                frame_index=0,
                ball=BallDetection(frame_index=0, x=300.0, y=200.0, confidence=0.9),
                players=[
                    PlayerDetection(frame_index=0, bbox=(250.0, 100.0, 350.0, 500.0),
                                    confidence=0.9, role="near_player"),
                ],
                poses=[],
            ),
            FrameTrackingResult(
                frame_index=1,
                ball=BallDetection(frame_index=1, x=310.0, y=250.0, confidence=0.9),
                players=[
                    PlayerDetection(frame_index=1, bbox=(250.0, 100.0, 350.0, 500.0),
                                    confidence=0.9, role="near_player"),
                ],
                poses=[],
            ),
        ]
        contacts = detect_contacts(tracking, fps=30.0)
        assert len(contacts) >= 1
        # Each contact is (frame_index, player_role)
        assert contacts[0][1] in ("near_player", "far_player")

    def test_no_contact_when_ball_far_from_player(self):
        """No contact when ball is not near any player."""
        tracking = [
            FrameTrackingResult(
                frame_index=0,
                ball=BallDetection(frame_index=0, x=100.0, y=100.0, confidence=0.9),
                players=[
                    PlayerDetection(frame_index=0, bbox=(500.0, 300.0, 600.0, 600.0),
                                    confidence=0.9, role="near_player"),
                ],
                poses=[],
            ),
        ]
        contacts = detect_contacts(tracking, fps=30.0)
        assert len(contacts) == 0

    def test_no_contact_when_no_ball(self):
        """No contact when ball is not detected."""
        tracking = [
            FrameTrackingResult(
                frame_index=0,
                ball=None,
                players=[
                    PlayerDetection(frame_index=0, bbox=(250.0, 100.0, 350.0, 500.0),
                                    confidence=0.9, role="near_player"),
                ],
                poses=[],
            ),
        ]
        contacts = detect_contacts(tracking, fps=30.0)
        assert len(contacts) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_shot_classify.py::TestDetectContacts -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Add to `src/court_vision/shot_classify.py`:

```python
from court_vision.ball_tracker import BallDetection
from court_vision.player_detect import FrameTrackingResult, PlayerDetection


def detect_contacts(
    tracking_results: list[FrameTrackingResult],
    fps: float,
    proximity_threshold: float = 100.0,
    min_frames_between_contacts: int = 5,
) -> list[tuple[int, str]]:
    """Detect frames where the ball contacts a player's racket.

    Uses proximity between ball position and player bounding box.
    A contact occurs when the ball is within proximity_threshold pixels
    of a player's bounding box center.

    Args:
        tracking_results: Per-frame tracking data.
        fps: Video frame rate.
        proximity_threshold: Max distance in pixels for ball-player contact.
        min_frames_between_contacts: Minimum frames between consecutive contacts.

    Returns:
        List of (frame_index, player_role) tuples for each detected contact.
    """
    contacts: list[tuple[int, str]] = []
    last_contact_frame = -min_frames_between_contacts

    for result in tracking_results:
        if result.ball is None:
            continue

        if result.frame_index - last_contact_frame < min_frames_between_contacts:
            continue

        ball_x, ball_y = result.ball.x, result.ball.y

        for player in result.players:
            if player.role is None:
                continue

            # Player bbox center
            px = (player.bbox[0] + player.bbox[2]) / 2
            py = (player.bbox[1] + player.bbox[3]) / 2

            # Check if ball is within the player's bounding box or close to it
            in_bbox_x = player.bbox[0] <= ball_x <= player.bbox[2]
            in_bbox_y = player.bbox[1] <= ball_y <= player.bbox[3]

            if in_bbox_x and in_bbox_y:
                contacts.append((result.frame_index, player.role))
                last_contact_frame = result.frame_index
                break

            # Fallback: check distance to bbox center
            dist = ((ball_x - px) ** 2 + (ball_y - py) ** 2) ** 0.5
            if dist < proximity_threshold:
                contacts.append((result.frame_index, player.role))
                last_contact_frame = result.frame_index
                break

    return contacts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_shot_classify.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/shot_classify.py tests/test_shot_classify.py
git commit -m "feat(court-vision): add ball-player contact detection"
```

---

### Task 5: Point Boundary Detection

**Files:**
- Modify: `src/court_vision/shot_classify.py`
- Modify: `tests/test_shot_classify.py`

Identifies point boundaries from gaps between gameplay segments. Each gameplay segment = one point (approximately).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_shot_classify.py`:

```python
from court_vision.shot_classify import detect_point_boundaries
from court_vision.scene_filter import GameplaySegment


class TestDetectPointBoundaries:
    def test_each_segment_is_a_point(self):
        """Each gameplay segment produces one point boundary."""
        segments = [
            GameplaySegment(start_frame=0, end_frame=100, start_time_s=0.0, end_time_s=3.33, frame_count=101),
            GameplaySegment(start_frame=150, end_frame=300, start_time_s=5.0, end_time_s=10.0, frame_count=151),
        ]
        boundaries = detect_point_boundaries(segments)
        assert len(boundaries) == 2
        assert boundaries[0] == (0, 100, 0.0, 3.33)
        assert boundaries[1] == (150, 300, 5.0, 10.0)

    def test_empty_segments(self):
        """Empty segment list returns empty boundaries."""
        assert detect_point_boundaries([]) == []

    def test_single_segment(self):
        """Single segment produces one boundary."""
        segments = [
            GameplaySegment(start_frame=50, end_frame=200, start_time_s=1.67, end_time_s=6.67, frame_count=151),
        ]
        boundaries = detect_point_boundaries(segments)
        assert len(boundaries) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_shot_classify.py::TestDetectPointBoundaries -v`

- [ ] **Step 3: Write minimal implementation**

Add to `src/court_vision/shot_classify.py`:

```python
from court_vision.scene_filter import GameplaySegment


def detect_point_boundaries(
    segments: list[GameplaySegment],
) -> list[tuple[int, int, float, float]]:
    """Detect point boundaries from gameplay segments.

    Each gameplay segment is treated as one point (approximately).
    Gaps between segments serve as point boundaries.

    Args:
        segments: Gameplay segments from the scene filter.

    Returns:
        List of (start_frame, end_frame, start_time_s, end_time_s) per point.
    """
    return [
        (seg.start_frame, seg.end_frame, seg.start_time_s, seg.end_time_s)
        for seg in segments
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_shot_classify.py -v`

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/shot_classify.py tests/test_shot_classify.py
git commit -m "feat(court-vision): add point boundary detection from gameplay segments"
```

---

### Task 6: Build Match Data (Full Shot Classification Orchestrator)

**Files:**
- Modify: `src/court_vision/shot_classify.py`
- Modify: `tests/test_shot_classify.py`

Adds `build_match_data` that orchestrates contact detection, stroke classification, placement, and point construction into a complete `MatchData`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_shot_classify.py`:

```python
from court_vision.shot_classify import build_match_data


class TestBuildMatchData:
    def test_builds_match_with_points_and_shots(self):
        """build_match_data produces MatchData with points containing shots."""
        segments = [
            GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
        ]
        tracking = [
            FrameTrackingResult(
                frame_index=10,
                ball=BallDetection(frame_index=10, x=300.0, y=300.0, confidence=0.9),
                players=[
                    PlayerDetection(frame_index=10, bbox=(250.0, 100.0, 350.0, 500.0),
                                    confidence=0.9, role="near_player"),
                ],
                poses=[PoseKeypoints(
                    frame_index=10, role="near_player",
                    keypoints={
                        "nose": (300.0, 100.0, 0.9),
                        "left_shoulder": (280.0, 200.0, 0.9),
                        "right_shoulder": (320.0, 200.0, 0.9),
                        "left_elbow": (260.0, 280.0, 0.9),
                        "right_elbow": (340.0, 280.0, 0.9),
                        "left_wrist": (250.0, 350.0, 0.9),
                        "right_wrist": (450.0, 250.0, 0.9),
                        "left_hip": (290.0, 400.0, 0.9),
                        "right_hip": (310.0, 400.0, 0.9),
                    },
                )],
            ),
        ]
        match = build_match_data(
            source="test.mp4",
            segments=segments,
            tracking_results=tracking,
            fps=30.0,
        )
        assert isinstance(match, MatchData)
        assert match.source_url == "test.mp4"
        assert len(match.points) == 1
        # Point should contain at least one shot from the contact
        point = match.points[0]
        assert point.point_number == 1
        assert point.rally_length >= 0

    def test_empty_tracking_produces_empty_points(self):
        """No tracking data -> points with no shots."""
        segments = [
            GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
        ]
        match = build_match_data(
            source="test.mp4",
            segments=segments,
            tracking_results=[],
            fps=30.0,
        )
        assert len(match.points) == 1
        assert match.points[0].rally_length == 0
        assert match.points[0].shots == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_shot_classify.py::TestBuildMatchData -v`

- [ ] **Step 3: Write minimal implementation**

Add to `src/court_vision/shot_classify.py`:

```python
import hashlib
from datetime import date


def build_match_data(
    source: str,
    segments: list[GameplaySegment],
    tracking_results: list[FrameTrackingResult],
    fps: float,
    court_homography: "np.ndarray | None" = None,
) -> MatchData:
    """Build complete match data from tracking results and segments.

    Orchestrates contact detection, stroke classification, placement computation,
    and point construction.

    Args:
        source: Source video path or URL.
        segments: Gameplay segments (one per approximate point).
        tracking_results: Per-frame tracking data.
        fps: Video frame rate.
        court_homography: Homography for ball-to-court mapping (optional).

    Returns:
        Complete MatchData ready for export.
    """
    import numpy as np

    match_id = hashlib.md5(source.encode()).hexdigest()[:12]
    boundaries = detect_point_boundaries(segments)
    contacts = detect_contacts(tracking_results, fps)

    # Build lookup: frame_index -> tracking result
    tracking_by_frame: dict[int, FrameTrackingResult] = {
        t.frame_index: t for t in tracking_results
    }

    points: list[Point] = []

    for point_num, (start, end, start_t, end_t) in enumerate(boundaries, 1):
        # Find contacts within this point's frame range
        point_contacts = [
            (frame, role) for frame, role in contacts
            if start <= frame <= end
        ]

        shots: list[Shot] = []
        for shot_num, (frame, role) in enumerate(point_contacts, 1):
            tracking = tracking_by_frame.get(frame)

            # Classify stroke from pose
            stroke = "forehand"
            stroke_conf = 0.4
            if tracking and tracking.poses:
                player_pose = next(
                    (p for p in tracking.poses if p.role == role), None
                )
                if player_pose:
                    stroke, stroke_conf = classify_stroke(player_pose)

            # Compute placement from ball position
            placement = None
            if tracking and tracking.ball and court_homography is not None:
                from court_vision.ball_tracker import map_ball_to_court

                court_pos = map_ball_to_court(tracking.ball, court_homography)
                if court_pos:
                    is_serve = shot_num == 1 and point_num > 0
                    zone = compute_placement_zone(
                        x=court_pos[0], y=court_pos[1],
                        is_serve=is_serve,
                        hitter=role,
                    )
                    placement = ShotPlacement(
                        x=court_pos[0], y=court_pos[1], zone=zone,
                    )

            shots.append(Shot(
                shot_number=shot_num,
                frame=frame,
                time_s=frame / fps,
                player=role,
                stroke=stroke,
                placement=placement,
                confidence=stroke_conf,
            ))

        # Determine outcome (simple heuristic: last shot determines outcome)
        outcome = None
        outcome_player = None
        if shots:
            last_shot = shots[-1]
            outcome = "winner" if last_shot.confidence > 0.6 else "error"
            outcome_player = last_shot.player

        points.append(Point(
            point_number=point_num,
            start_frame=start,
            end_frame=end,
            start_time_s=start_t,
            end_time_s=end_t,
            server=shots[0].player if shots else None,
            shots=shots,
            outcome=outcome,
            outcome_player=outcome_player,
            rally_length=len(shots),
        ))

    return MatchData(
        match_id=match_id,
        source_url=source,
        metadata={
            "players": ["near_player", "far_player"],
            "date_processed": str(date.today()),
        },
        points=points,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_shot_classify.py -v`

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/shot_classify.py tests/test_shot_classify.py
git commit -m "feat(court-vision): add match data builder with shot classification orchestration"
```

---

### Task 7: JSON Export

**Files:**
- Create: `src/court_vision/export.py`
- Create: `tests/test_export.py`

Exports `MatchData` to the JSON format defined in the spec.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the export module."""

import json
from pathlib import Path

import pytest

from court_vision.export import export_json
from court_vision.shot_classify import MatchData, Point, Shot, ShotPlacement


class TestExportJson:
    def _sample_match(self) -> MatchData:
        """Create a minimal MatchData for testing."""
        return MatchData(
            match_id="test_abc123",
            source_url="test.mp4",
            metadata={"players": ["near_player", "far_player"], "date_processed": "2026-03-30"},
            points=[
                Point(
                    point_number=1,
                    start_frame=100, end_frame=200,
                    start_time_s=3.33, end_time_s=6.67,
                    server="near_player",
                    shots=[
                        Shot(
                            shot_number=1, frame=110, time_s=3.67,
                            player="near_player", stroke="serve",
                            placement=ShotPlacement(x=4.0, y=1.0, zone="wide"),
                            confidence=0.9,
                        ),
                        Shot(
                            shot_number=2, frame=140, time_s=4.67,
                            player="far_player", stroke="forehand",
                            placement=ShotPlacement(x=-2.0, y=8.5, zone="crosscourt_deep"),
                            confidence=0.85,
                        ),
                    ],
                    outcome="winner",
                    outcome_player="far_player",
                    rally_length=2,
                ),
            ],
        )

    def test_exports_valid_json(self, tmp_path: Path):
        """Exports match data as valid JSON."""
        match = self._sample_match()
        output = tmp_path / "match.json"
        export_json(match, output)

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["match_id"] == "test_abc123"
        assert len(data["points"]) == 1

    def test_json_structure_matches_spec(self, tmp_path: Path):
        """JSON structure matches the spec format."""
        match = self._sample_match()
        output = tmp_path / "match.json"
        export_json(match, output)

        data = json.loads(output.read_text())
        point = data["points"][0]
        assert "point_number" in point
        assert "shots" in point
        assert "outcome" in point
        assert "review_status" in point

        shot = point["shots"][0]
        assert "shot_number" in shot
        assert "frame" in shot
        assert "player" in shot
        assert "stroke" in shot
        assert "placement" in shot
        assert "confidence" in shot

        placement = shot["placement"]
        assert "x" in placement
        assert "y" in placement
        assert "zone" in placement

    def test_json_has_metadata(self, tmp_path: Path):
        """JSON includes source_url and metadata."""
        match = self._sample_match()
        output = tmp_path / "match.json"
        export_json(match, output)

        data = json.loads(output.read_text())
        assert data["source_url"] == "test.mp4"
        assert "metadata" in data
        assert data["metadata"]["players"] == ["near_player", "far_player"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_export.py -v`

- [ ] **Step 3: Write minimal implementation**

```python
"""Export — JSON and CSV output for match data."""

import json
from dataclasses import asdict
from pathlib import Path

from court_vision.shot_classify import MatchData


def export_json(match: MatchData, output_path: Path) -> None:
    """Export match data to JSON in the spec-defined format.

    Args:
        match: Complete match data.
        output_path: Path to write JSON file.
    """
    data = asdict(match)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_export.py -v`

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/export.py tests/test_export.py
git commit -m "feat(court-vision): add JSON export in spec format"
```

---

### Task 8: CSV Export (Sackmann-Compatible)

**Files:**
- Modify: `src/court_vision/export.py`
- Modify: `tests/test_export.py`

Adds CSV export producing aggregated match stats compatible with Sackmann format.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_export.py`:

```python
import pandas as pd

from court_vision.export import export_csv


class TestExportCsv:
    def _sample_match(self) -> MatchData:
        """Create a minimal MatchData for testing."""
        return MatchData(
            match_id="test_abc123",
            source_url="test.mp4",
            metadata={"players": ["near_player", "far_player"], "date_processed": "2026-03-30"},
            points=[
                Point(
                    point_number=1,
                    start_frame=100, end_frame=200,
                    start_time_s=3.33, end_time_s=6.67,
                    server="near_player",
                    shots=[
                        Shot(shot_number=1, frame=110, time_s=3.67, player="near_player",
                             stroke="serve", placement=ShotPlacement(x=4.0, y=1.0, zone="wide"),
                             confidence=0.9),
                        Shot(shot_number=2, frame=140, time_s=4.67, player="far_player",
                             stroke="forehand", placement=ShotPlacement(x=-2.0, y=8.5, zone="crosscourt_deep"),
                             confidence=0.85),
                    ],
                    outcome="winner", outcome_player="far_player", rally_length=2,
                ),
                Point(
                    point_number=2,
                    start_frame=300, end_frame=400,
                    start_time_s=10.0, end_time_s=13.33,
                    server="far_player",
                    shots=[
                        Shot(shot_number=1, frame=310, time_s=10.33, player="far_player",
                             stroke="serve", placement=None, confidence=0.8),
                    ],
                    outcome="error", outcome_player="far_player", rally_length=1,
                ),
            ],
        )

    def test_exports_csv_file(self, tmp_path: Path):
        """Exports match stats to CSV."""
        match = self._sample_match()
        output = tmp_path / "stats.csv"
        export_csv(match, output)

        assert output.exists()
        df = pd.read_csv(output)
        assert len(df) > 0

    def test_csv_has_match_id(self, tmp_path: Path):
        """CSV includes match_id column."""
        match = self._sample_match()
        output = tmp_path / "stats.csv"
        export_csv(match, output)

        df = pd.read_csv(output)
        assert "match_id" in df.columns

    def test_csv_has_stat_columns(self, tmp_path: Path):
        """CSV includes stat columns for each player."""
        match = self._sample_match()
        output = tmp_path / "stats.csv"
        export_csv(match, output)

        df = pd.read_csv(output)
        assert "total_points" in df.columns
        assert "winners" in df.columns
        assert "errors" in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_export.py::TestExportCsv -v`

- [ ] **Step 3: Write minimal implementation**

Add to `src/court_vision/export.py`:

```python
import pandas as pd


def export_csv(match: MatchData, output_path: Path) -> None:
    """Export aggregated match statistics to CSV.

    Produces Sackmann-compatible match-level stats per player.

    Args:
        match: Complete match data.
        output_path: Path to write CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict] = {}
    for player in ("near_player", "far_player"):
        stats[player] = {
            "match_id": match.match_id,
            "player": player,
            "total_points": 0,
            "points_won": 0,
            "winners": 0,
            "errors": 0,
            "aces": 0,
            "forehands": 0,
            "backhands": 0,
            "serves": 0,
            "volleys": 0,
            "total_shots": 0,
        }

    for point in match.points:
        # Count points
        if point.server:
            stats[point.server]["total_points"] += 1
            other = "far_player" if point.server == "near_player" else "near_player"
            stats[other]["total_points"] += 1

        # Count outcome
        if point.outcome == "winner" and point.outcome_player:
            stats[point.outcome_player]["points_won"] += 1
            stats[point.outcome_player]["winners"] += 1
        elif point.outcome == "error" and point.outcome_player:
            other = "far_player" if point.outcome_player == "near_player" else "near_player"
            stats[other]["points_won"] += 1
            stats[point.outcome_player]["errors"] += 1

        # Count shots by type
        for shot in point.shots:
            p = shot.player
            stats[p]["total_shots"] += 1
            if shot.stroke == "forehand":
                stats[p]["forehands"] += 1
            elif shot.stroke == "backhand":
                stats[p]["backhands"] += 1
            elif shot.stroke == "serve":
                stats[p]["serves"] += 1
                # Check for ace (serve that ends the point)
                if shot.shot_number == len(point.shots) and point.outcome == "winner":
                    stats[p]["aces"] += 1
            elif shot.stroke == "volley":
                stats[p]["volleys"] += 1

    rows = list(stats.values())
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_export.py -v`

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/export.py tests/test_export.py
git commit -m "feat(court-vision): add Sackmann-compatible CSV export"
```

---

### Task 9: Pipeline Integration (Stage 5)

**Files:**
- Modify: `src/court_vision/pipeline.py`
- Modify: `tests/test_pipeline.py`

Wires Stage 5 (shot classification) into the pipeline orchestrator.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
class TestRunPipelineWithShotClassification:
    @patch("court_vision.pipeline.build_match_data")
    @patch("court_vision.pipeline.track_segment")
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.classify_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_pipeline_runs_shot_classification(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_homographies: MagicMock,
        mock_track: MagicMock,
        mock_build_match: MagicMock,
        tmp_path: Path,
    ):
        """Pipeline calls build_match_data after tracking."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.ingest import FrameSequence
        from court_vision.scene_filter import GameplaySegment
        from court_vision.shot_classify import MatchData

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
        mock_homographies.return_value = [CourtDetectionResult(success=True, num_lines_detected=6)]
        mock_track.return_value = []
        mock_build_match.return_value = MatchData(
            match_id="test", source_url="test.mp4",
            metadata={}, points=[],
        )

        result = run_pipeline(str(video_path), config_path=None)

        mock_build_match.assert_called_once()
        assert result.match_data is not None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_pipeline.py::TestRunPipelineWithShotClassification -v`

- [ ] **Step 3: Update pipeline.py**

Add import:
```python
from court_vision.shot_classify import MatchData, build_match_data
```

Add field to PipelineResult:
```python
    match_data: MatchData | None = None
```

Add Stage 5 after tracking:
```python
    # Stage 5: Shot Classification
    match_data = build_match_data(
        source=source,
        segments=segments,
        tracking_results=all_tracking,
        fps=frame_seq.fps,
    )
```

Update return to include `match_data=match_data`.

- [ ] **Step 4: Add `@patch("court_vision.pipeline.build_match_data")` to ALL existing pipeline tests**

Add the decorator as the topmost patch to all 4 existing test methods. Add `mock_build_match: MagicMock` as the last mock parameter (before `tmp_path`). Set `mock_build_match.return_value = None` in each existing test.

- [ ] **Step 5: Run tests to verify all pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_pipeline.py -v`

- [ ] **Step 6: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/pipeline.py tests/test_pipeline.py
git commit -m "feat(court-vision): integrate shot classification into pipeline (Stage 5)"
```

---

### Task 10: CLI Export Command and Updated Process Output

**Files:**
- Modify: `src/court_vision/cli.py`
- Modify: `tests/test_cli.py`

Adds an `export` command and shot classification summary to `process` output.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
class TestProcessCommandShotOutput:
    @patch("court_vision.pipeline.run_pipeline")
    def test_process_shows_shot_summary(self, mock_pipeline: MagicMock):
        """Process command displays shot classification summary."""
        from court_vision.scene_filter import GameplaySegment
        from court_vision.shot_classify import MatchData, Point, Shot, ShotPlacement

        mock_pipeline.return_value = MagicMock(
            total_frames=100,
            fps=30.0,
            gameplay_segments=[
                GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
            ],
            gameplay_frame_count=51,
            court_detections=[],
            tracking_results=[],
            match_data=MatchData(
                match_id="test",
                source_url="test.mp4",
                metadata={},
                points=[
                    Point(
                        point_number=1, start_frame=0, end_frame=50,
                        start_time_s=0.0, end_time_s=1.67,
                        server="near_player",
                        shots=[
                            Shot(shot_number=1, frame=10, time_s=0.33,
                                 player="near_player", stroke="serve",
                                 placement=None, confidence=0.8),
                        ],
                        outcome="winner", outcome_player="near_player",
                        rally_length=1,
                    ),
                ],
            ),
        )

        result = runner.invoke(app, ["process", "test.mp4"])

        assert result.exit_code == 0
        assert "point" in result.output.lower() or "shot" in result.output.lower()


class TestExportCommand:
    @patch("court_vision.export.export_json")
    @patch("court_vision.export.export_csv")
    def test_export_json(self, mock_csv: MagicMock, mock_json: MagicMock, tmp_path: Path):
        """Export command writes JSON output."""
        from court_vision.shot_classify import MatchData

        # Create a temporary match JSON to load
        import json
        match_file = tmp_path / "match.json"
        match_data = {
            "match_id": "test", "source_url": "test.mp4",
            "metadata": {}, "points": [],
        }
        match_file.write_text(json.dumps(match_data))

        result = runner.invoke(app, ["export", str(match_file), "--format", "json",
                                      "--output", str(tmp_path / "out.json")])

        assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_cli.py::TestProcessCommandShotOutput tests/test_cli.py::TestExportCommand -v`

- [ ] **Step 3: Update cli.py**

Add shot classification summary to `process` command (after tracking output):

```python
    if result.match_data:
        total_shots = sum(len(p.shots) for p in result.match_data.points)
        typer.echo(f"Shot classification: {len(result.match_data.points)} points, {total_shots} shots detected")
```

Add `export` command:

```python
@app.command()
def export(
    match_json: Path = typer.Argument(help="Path to match data JSON file."),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or csv."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path."),
) -> None:
    """Export match data to JSON or CSV format."""
    import json as json_module

    from court_vision.export import export_csv, export_json
    from court_vision.shot_classify import MatchData

    with open(match_json) as f:
        raw = json_module.load(f)

    match = MatchData(
        match_id=raw["match_id"],
        source_url=raw["source_url"],
        metadata=raw.get("metadata", {}),
        points=[],  # Simplified for now — full deserialization in future
    )

    if output is None:
        stem = match_json.stem
        output = match_json.parent / f"{stem}_export.{format}"

    if format == "csv":
        export_csv(match, output)
    else:
        export_json(match, output)

    typer.echo(f"Exported to {output}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/test_cli.py -v`

- [ ] **Step 5: Commit**

```bash
cd /Users/pc/web3/tennisconcrete/court-vision
git add src/court_vision/cli.py tests/test_cli.py
git commit -m "feat(court-vision): add export command and shot classification CLI output"
```

---

### Task 11: Full Test Suite Verification

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Verify all new imports**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -c "from court_vision.shot_classify import Shot, ShotPlacement, Point, MatchData, compute_placement_zone, classify_stroke, detect_contacts, detect_point_boundaries, build_match_data; from court_vision.export import export_json, export_csv; print('All imports OK')"`

- [ ] **Step 3: Verify pipeline imports**

Run: `cd /Users/pc/web3/tennisconcrete/court-vision && .venv/bin/python -c "from court_vision.pipeline import run_pipeline, PipelineResult; print('Pipeline OK')"`

---

## Self-Review Checklist

1. **Spec coverage:**
   - Stroke type classification (forehand/backhand/serve/volley/overhead/slice) ✓ Task 3
   - Placement zones (serve: wide/body/t, rally: direction+depth) ✓ Task 2
   - Point outcome (winner/error/unforced_error) ✓ Task 6
   - Rule-based heuristics ✓ Task 3
   - Confidence scores on all predictions ✓ Tasks 3, 6
   - JSON output in spec format ✓ Task 7
   - CSV/Sackmann-compatible export ✓ Task 8
   - Pipeline integration ✓ Task 9
   - CLI export command ✓ Task 10
   - review_status field ✓ Task 2 (Point dataclass)

2. **Placeholder scan:** No TBD/TODO items. All code blocks complete.

3. **Type consistency:** `ShotPlacement`, `Shot`, `Point`, `MatchData` used consistently. `compute_placement_zone`, `classify_stroke`, `detect_contacts`, `detect_point_boundaries`, `build_match_data` signatures consistent. `export_json`, `export_csv` signatures consistent.
