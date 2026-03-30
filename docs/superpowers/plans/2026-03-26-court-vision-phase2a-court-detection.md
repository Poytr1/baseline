# Court Vision Phase 2A — Court Detection & Homography

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect tennis court lines in gameplay frames and compute a homography matrix that maps pixel coordinates to real-world court coordinates (meters), enabling all downstream tracking stages to work in court-space.

**Architecture:** A new `court_detect.py` module inside `court_vision/` that takes gameplay frames (from Phase 1's scene filter output) and produces per-segment homography matrices. Uses OpenCV Hough line detection + geometric constraints to identify court lines, then `cv2.findHomography` to compute the pixel→court transform. Integrates into the existing sequential pipeline.

**Tech Stack:** Python 3.11+, OpenCV (Hough lines, homography), NumPy, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-court-vision-design.md` — Stage 3

**Phase 1 plan:** `docs/superpowers/plans/2026-03-25-court-vision-phase1.md`

---

## Phase 2A Scope

Phase 2A delivers:
1. Court keypoint definition — canonical court coordinates in meters
2. Court line detection — Hough transform with tennis-specific geometric filtering
3. Keypoint extraction — intersect detected lines to find court corner/junction points
4. Homography computation — pixel keypoints → court-space transform matrix
5. Per-segment homography — compute one homography per gameplay segment
6. Pipeline integration — wire `court_detect` into the existing pipeline orchestrator
7. CLI output update — display homography status in pipeline output

Phase 2A does NOT include: ball tracking, player detection, shot classification, export, or review UI. Those are Phase 2B–2D.

**Deferred from spec (Phase 2B+):** The spec mentions "court line detection as secondary signal" for the scene filter. This plan builds the court detection module standalone. Wiring it as a secondary scene-filter signal is deferred until Phase 2B when the integration point is clearer.

---

## Court Coordinate System

All coordinates use meters. Origin at center of net.

```
              Baseline (far)
         y = +11.885
     ┌────────────────────────┐
     │                        │  x = -5.485 (left doubles sideline)
     │    ┌──────────────┐    │
     │    │  Service box  │    │  x = -4.115 (left singles sideline)
     │    │   (far-left)  │    │
     │    ├──────┬───────┤    │  y = +6.4 (service line, far)
     │    │      │       │    │
     │    │      │ center │    │  x = 0 (center service line)
     │    │      │  mark  │    │
  ───┼────┼──────┼───────┼────┼── y = 0 (net)
     │    │      │       │    │
     │    │      │       │    │
     │    ├──────┴───────┤    │  y = -6.4 (service line, near)
     │    │              │    │
     │    │              │    │  x = +4.115 (right singles sideline)
     │    └──────────────┘    │
     │                        │  x = +5.485 (right doubles sideline)
     └────────────────────────┘
         y = -11.885
              Baseline (near)
```

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `court-vision/src/court_vision/court_detect.py` | Court line detection, keypoint extraction, homography computation |
| Create | `court-vision/tests/test_court_detect.py` | Unit tests for all court detection logic |
| Modify | `court-vision/src/court_vision/pipeline.py` | Wire court detection as Stage 3 after scene filter |
| Modify | `court-vision/src/court_vision/cli.py` | Display homography results in process command output |
| Modify | `court-vision/tests/test_pipeline.py` | Update mocks for new court detection stage |

---

### Task 1: Court Keypoint Constants

**Files:**
- Create: `court-vision/src/court_vision/court_detect.py`
- Create: `court-vision/tests/test_court_detect.py`

- [ ] **Step 1: Write the failing test**

```python
# court-vision/tests/test_court_detect.py
"""Tests for court detection and homography."""

import numpy as np
import pytest

from court_vision.court_detect import COURT_KEYPOINTS


class TestCourtKeypoints:
    def test_keypoints_is_dict(self):
        """COURT_KEYPOINTS maps names to (x, y) tuples in meters."""
        assert isinstance(COURT_KEYPOINTS, dict)
        assert len(COURT_KEYPOINTS) >= 12

    def test_net_center_is_origin(self):
        """Net center is at origin (0, 0)."""
        assert COURT_KEYPOINTS["net_center"] == (0.0, 0.0)

    def test_baseline_far_left_singles(self):
        """Far-left singles baseline corner."""
        x, y = COURT_KEYPOINTS["baseline_far_left_singles"]
        assert x == pytest.approx(-4.115)
        assert y == pytest.approx(11.885)

    def test_baseline_near_right_singles(self):
        """Near-right singles baseline corner."""
        x, y = COURT_KEYPOINTS["baseline_near_right_singles"]
        assert x == pytest.approx(4.115)
        assert y == pytest.approx(-11.885)

    def test_service_line_far_center(self):
        """Far service line at center mark."""
        x, y = COURT_KEYPOINTS["service_far_center"]
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(6.4)

    def test_all_keypoints_are_float_tuples(self):
        """Every keypoint is a 2-tuple of floats."""
        for name, (x, y) in COURT_KEYPOINTS.items():
            assert isinstance(x, float), f"{name} x is not float"
            assert isinstance(y, float), f"{name} y is not float"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestCourtKeypoints -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# court-vision/src/court_vision/court_detect.py
"""Court detection — line detection, keypoint extraction, homography."""

# Standard tennis court dimensions in meters.
# Origin (0, 0) = center of net.
# X-axis: parallel to net, positive = right when facing far end.
# Y-axis: perpendicular to net, positive = far side.

_SINGLES_WIDTH_HALF = 4.115  # Singles sideline to center
_DOUBLES_WIDTH_HALF = 5.485  # Doubles sideline to center
_BASELINE_DIST = 11.885  # Baseline to net
_SERVICE_LINE_DIST = 6.4  # Service line to net

COURT_KEYPOINTS: dict[str, tuple[float, float]] = {
    # Far baseline (y = +11.885)
    "baseline_far_left_doubles": (-_DOUBLES_WIDTH_HALF, _BASELINE_DIST),
    "baseline_far_left_singles": (-_SINGLES_WIDTH_HALF, _BASELINE_DIST),
    "baseline_far_center": (0.0, _BASELINE_DIST),
    "baseline_far_right_singles": (_SINGLES_WIDTH_HALF, _BASELINE_DIST),
    "baseline_far_right_doubles": (_DOUBLES_WIDTH_HALF, _BASELINE_DIST),
    # Far service line (y = +6.4)
    "service_far_left": (-_SINGLES_WIDTH_HALF, _SERVICE_LINE_DIST),
    "service_far_center": (0.0, _SERVICE_LINE_DIST),
    "service_far_right": (_SINGLES_WIDTH_HALF, _SERVICE_LINE_DIST),
    # Net (y = 0)
    "net_left_singles": (-_SINGLES_WIDTH_HALF, 0.0),
    "net_center": (0.0, 0.0),
    "net_right_singles": (_SINGLES_WIDTH_HALF, 0.0),
    "net_left_doubles": (-_DOUBLES_WIDTH_HALF, 0.0),
    "net_right_doubles": (_DOUBLES_WIDTH_HALF, 0.0),
    # Near service line (y = -6.4)
    "service_near_left": (-_SINGLES_WIDTH_HALF, -_SERVICE_LINE_DIST),
    "service_near_center": (0.0, -_SERVICE_LINE_DIST),
    "service_near_right": (_SINGLES_WIDTH_HALF, -_SERVICE_LINE_DIST),
    # Near baseline (y = -11.885)
    "baseline_near_left_doubles": (-_DOUBLES_WIDTH_HALF, -_BASELINE_DIST),
    "baseline_near_left_singles": (-_SINGLES_WIDTH_HALF, -_BASELINE_DIST),
    "baseline_near_center": (0.0, -_BASELINE_DIST),
    "baseline_near_right_singles": (_SINGLES_WIDTH_HALF, -_BASELINE_DIST),
    "baseline_near_right_doubles": (_DOUBLES_WIDTH_HALF, -_BASELINE_DIST),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestCourtKeypoints -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/court_detect.py court-vision/tests/test_court_detect.py
git commit -m "feat(court-vision): add court keypoint constants in meters"
```

---

### Task 2: Hough Line Detection with Court Filtering

**Files:**
- Modify: `court-vision/src/court_vision/court_detect.py`
- Modify: `court-vision/tests/test_court_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `court-vision/tests/test_court_detect.py`:

```python
import cv2


def _draw_court_lines(img: np.ndarray) -> np.ndarray:
    """Draw white court lines on a green court image for testing.

    Draws a simplified court: two baselines, two sidelines, net,
    two service lines, center service line. All in white on green.
    Returns the image with lines drawn.
    """
    h, w = img.shape[:2]

    # Approximate court region in pixel space (centered, perspective-ish)
    # Near baseline (bottom)
    cv2.line(img, (200, 650), (1080, 650), (255, 255, 255), 2)
    # Far baseline (top)
    cv2.line(img, (400, 150), (880, 150), (255, 255, 255), 2)
    # Left sideline
    cv2.line(img, (200, 650), (400, 150), (255, 255, 255), 2)
    # Right sideline
    cv2.line(img, (1080, 650), (880, 150), (255, 255, 255), 2)
    # Near service line
    cv2.line(img, (280, 450), (1000, 450), (255, 255, 255), 2)
    # Far service line
    cv2.line(img, (360, 280), (920, 280), (255, 255, 255), 2)
    # Center service line
    cv2.line(img, (640, 280), (640, 450), (255, 255, 255), 2)

    return img


class TestDetectCourtLines:
    def test_detects_lines_on_synthetic_court(self):
        """Detects lines from a synthetic court image."""
        from court_vision.court_detect import detect_court_lines

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)  # green
        img = _draw_court_lines(img)

        lines = detect_court_lines(img)

        # Should detect at least the major horizontal and vertical lines
        assert len(lines) >= 4

    def test_returns_list_of_line_segments(self):
        """Each detected line is a pair of (x, y) endpoints."""
        from court_vision.court_detect import detect_court_lines

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        img = _draw_court_lines(img)

        lines = detect_court_lines(img)

        for line in lines:
            assert len(line) == 2, "Each line should be ((x1,y1), (x2,y2))"
            (x1, y1), (x2, y2) = line
            assert isinstance(x1, (int, float))
            assert isinstance(y1, (int, float))

    def test_no_lines_on_blank_image(self):
        """Returns empty list when no court lines are present."""
        from court_vision.court_detect import detect_court_lines

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)

        lines = detect_court_lines(img)

        assert lines == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestDetectCourtLines -v`
Expected: FAIL with `ImportError: cannot import name 'detect_court_lines'`

- [ ] **Step 3: Write minimal implementation**

Append to `court-vision/src/court_vision/court_detect.py`:

```python
import cv2
import numpy as np


def detect_court_lines(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    min_line_length: int = 100,
    max_line_gap: int = 30,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Detect court lines in a frame using Hough line transform.

    Applies white-pixel masking (court lines are white), Canny edge
    detection, and probabilistic Hough transform. Filters short
    spurious segments.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        canny_low: Lower Canny edge threshold.
        canny_high: Upper Canny edge threshold.
        hough_threshold: Hough accumulator threshold.
        min_line_length: Minimum line length in pixels.
        max_line_gap: Maximum gap between line segments to merge.

    Returns:
        List of line segments as ((x1, y1), (x2, y2)) tuples.
    """
    # Convert to HSV and mask white pixels (court lines)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # White pixels: low saturation, high value
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))

    # Also try grayscale thresholding for robustness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Combine both masks
    combined = cv2.bitwise_or(white_mask, bright_mask)

    # Edge detection on the masked image
    edges = cv2.Canny(combined, canny_low, canny_high)

    # Probabilistic Hough transform
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if raw_lines is None:
        return []

    lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        lines.append(((x1, y1), (x2, y2)))

    return lines
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestDetectCourtLines -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/court_detect.py court-vision/tests/test_court_detect.py
git commit -m "feat(court-vision): add Hough line detection for court lines"
```

---

### Task 3: Line Classification (Horizontal vs Vertical)

**Files:**
- Modify: `court-vision/src/court_vision/court_detect.py`
- Modify: `court-vision/tests/test_court_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `court-vision/tests/test_court_detect.py`:

```python
class TestClassifyLines:
    def test_horizontal_line(self):
        """A nearly-horizontal line is classified as horizontal."""
        from court_vision.court_detect import classify_lines

        lines = [((100, 300), (900, 310))]  # nearly horizontal
        h, v = classify_lines(lines, angle_threshold=30.0)
        assert len(h) == 1
        assert len(v) == 0

    def test_vertical_line(self):
        """A nearly-vertical line is classified as vertical."""
        from court_vision.court_detect import classify_lines

        lines = [((500, 100), (510, 600))]  # nearly vertical
        h, v = classify_lines(lines, angle_threshold=30.0)
        assert len(h) == 0
        assert len(v) == 1

    def test_diagonal_line_excluded(self):
        """A 45-degree line is neither horizontal nor vertical."""
        from court_vision.court_detect import classify_lines

        lines = [((100, 100), (500, 500))]  # 45 degrees
        h, v = classify_lines(lines, angle_threshold=30.0)
        assert len(h) == 0
        assert len(v) == 0

    def test_mixed_lines(self):
        """Correctly separates a mix of horizontal and vertical lines."""
        from court_vision.court_detect import classify_lines

        lines = [
            ((100, 300), (900, 310)),   # horizontal
            ((500, 100), (510, 600)),   # vertical
            ((200, 200), (800, 205)),   # horizontal
        ]
        h, v = classify_lines(lines, angle_threshold=30.0)
        assert len(h) == 2
        assert len(v) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestClassifyLines -v`
Expected: FAIL with `ImportError: cannot import name 'classify_lines'`

- [ ] **Step 3: Write minimal implementation**

Append to `court-vision/src/court_vision/court_detect.py`:

```python
def classify_lines(
    lines: list[tuple[tuple[int, int], tuple[int, int]]],
    angle_threshold: float = 30.0,
) -> tuple[
    list[tuple[tuple[int, int], tuple[int, int]]],
    list[tuple[tuple[int, int], tuple[int, int]]],
]:
    """Classify detected lines as horizontal or vertical.

    Lines within `angle_threshold` degrees of horizontal (0°) are
    classified as horizontal. Lines within `angle_threshold` degrees
    of vertical (90°) are classified as vertical. Diagonal lines
    (between the two thresholds) are discarded.

    Args:
        lines: List of line segments as ((x1, y1), (x2, y2)).
        angle_threshold: Maximum deviation from axis in degrees.

    Returns:
        Tuple of (horizontal_lines, vertical_lines).
    """
    horizontal = []
    vertical = []

    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))

        # Normalize to 0-90 range
        if angle > 90:
            angle = 180 - angle

        if angle <= angle_threshold:
            horizontal.append(((x1, y1), (x2, y2)))
        elif angle >= (90 - angle_threshold):
            vertical.append(((x1, y1), (x2, y2)))
        # else: diagonal — discard

    return horizontal, vertical
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestClassifyLines -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/court_detect.py court-vision/tests/test_court_detect.py
git commit -m "feat(court-vision): add line classification (horizontal/vertical)"
```

---

### Task 4: Keypoint Extraction via Line Intersections

**Files:**
- Modify: `court-vision/src/court_vision/court_detect.py`
- Modify: `court-vision/tests/test_court_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `court-vision/tests/test_court_detect.py`:

```python
class TestLineIntersection:
    def test_perpendicular_lines_intersect(self):
        """Two perpendicular lines intersect at their crossing point."""
        from court_vision.court_detect import find_line_intersection

        # Horizontal line at y=300
        line1 = ((100, 300), (900, 300))
        # Vertical line at x=500
        line2 = ((500, 100), (500, 600))

        point = find_line_intersection(line1, line2)

        assert point is not None
        x, y = point
        assert x == pytest.approx(500.0, abs=1.0)
        assert y == pytest.approx(300.0, abs=1.0)

    def test_parallel_lines_no_intersection(self):
        """Parallel lines return None."""
        from court_vision.court_detect import find_line_intersection

        line1 = ((100, 300), (900, 300))
        line2 = ((100, 400), (900, 400))

        point = find_line_intersection(line1, line2)

        assert point is None

    def test_angled_lines_intersect(self):
        """Two angled lines find their intersection."""
        from court_vision.court_detect import find_line_intersection

        # Line from (0,0) to (100,100) and from (100,0) to (0,100)
        line1 = ((0, 0), (100, 100))
        line2 = ((100, 0), (0, 100))

        point = find_line_intersection(line1, line2)

        assert point is not None
        x, y = point
        assert x == pytest.approx(50.0, abs=1.0)
        assert y == pytest.approx(50.0, abs=1.0)


class TestExtractKeypoints:
    def test_extracts_keypoints_from_court_lines(self):
        """Extracts intersection points from horizontal and vertical lines."""
        from court_vision.court_detect import extract_keypoints

        horizontal = [
            ((200, 150), (880, 150)),   # far baseline
            ((200, 650), (1080, 650)),  # near baseline
        ]
        vertical = [
            ((200, 650), (200, 150)),   # left sideline (extended)
            ((1080, 650), (880, 150)),  # right sideline (extended)
        ]

        keypoints = extract_keypoints(horizontal, vertical)

        # Should find intersections of each horizontal with each vertical
        assert len(keypoints) >= 2

    def test_keypoints_are_float_tuples(self):
        """Each keypoint is a (x, y) tuple of floats."""
        from court_vision.court_detect import extract_keypoints

        horizontal = [((100, 300), (900, 300))]
        vertical = [((500, 100), (500, 600))]

        keypoints = extract_keypoints(horizontal, vertical)

        for x, y in keypoints:
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_no_keypoints_from_empty_lines(self):
        """Returns empty list when no lines provided."""
        from court_vision.court_detect import extract_keypoints

        keypoints = extract_keypoints([], [])

        assert keypoints == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestLineIntersection tests/test_court_detect.py::TestExtractKeypoints -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Append to `court-vision/src/court_vision/court_detect.py`:

```python
def find_line_intersection(
    line1: tuple[tuple[int, int], tuple[int, int]],
    line2: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[float, float] | None:
    """Find the intersection point of two line segments (extended to infinite lines).

    Uses the cross-product method for line-line intersection.

    Args:
        line1: First line as ((x1, y1), (x2, y2)).
        line2: Second line as ((x3, y3), (x4, y4)).

    Returns:
        (x, y) intersection point, or None if lines are parallel.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # parallel or coincident

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)

    return (float(ix), float(iy))


def extract_keypoints(
    horizontal_lines: list[tuple[tuple[int, int], tuple[int, int]]],
    vertical_lines: list[tuple[tuple[int, int], tuple[int, int]]],
    frame_width: int = 1280,
    frame_height: int = 720,
) -> list[tuple[float, float]]:
    """Extract court keypoints as intersections of horizontal and vertical lines.

    Computes all pairwise intersections between horizontal and vertical
    lines and filters to points within the frame bounds.

    Args:
        horizontal_lines: Detected horizontal court lines.
        vertical_lines: Detected vertical court lines.
        frame_width: Image width for bounds checking.
        frame_height: Image height for bounds checking.

    Returns:
        List of (x, y) pixel coordinates for detected keypoints.
    """
    keypoints: list[tuple[float, float]] = []

    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            point = find_line_intersection(h_line, v_line)
            if point is None:
                continue

            x, y = point
            # Keep only points within frame bounds (with small margin)
            margin = 50
            if -margin <= x <= frame_width + margin and -margin <= y <= frame_height + margin:
                keypoints.append(point)

    return keypoints
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestLineIntersection tests/test_court_detect.py::TestExtractKeypoints -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/court_detect.py court-vision/tests/test_court_detect.py
git commit -m "feat(court-vision): add line intersection and keypoint extraction"
```

---

### Task 5: Homography Computation

**Files:**
- Modify: `court-vision/src/court_vision/court_detect.py`
- Modify: `court-vision/tests/test_court_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `court-vision/tests/test_court_detect.py`:

```python
class TestComputeHomography:
    def test_identity_like_homography(self):
        """When pixel and court points are proportional, homography maps correctly."""
        from court_vision.court_detect import compute_homography

        # Use 4 known pixel-court point correspondences
        pixel_points = np.array([
            [200.0, 650.0],   # near-left
            [1080.0, 650.0],  # near-right
            [880.0, 150.0],   # far-right
            [400.0, 150.0],   # far-left
        ], dtype=np.float64)

        court_points = np.array([
            [-4.115, -11.885],   # near-left singles
            [4.115, -11.885],    # near-right singles
            [4.115, 11.885],     # far-right singles
            [-4.115, 11.885],    # far-left singles
        ], dtype=np.float64)

        H = compute_homography(pixel_points, court_points)

        assert H is not None
        assert H.shape == (3, 3)

    def test_homography_transforms_known_point(self):
        """Homography correctly transforms a known pixel point to court coords."""
        from court_vision.court_detect import compute_homography, pixel_to_court

        pixel_points = np.array([
            [200.0, 650.0],
            [1080.0, 650.0],
            [880.0, 150.0],
            [400.0, 150.0],
        ], dtype=np.float64)

        court_points = np.array([
            [-4.115, -11.885],
            [4.115, -11.885],
            [4.115, 11.885],
            [-4.115, 11.885],
        ], dtype=np.float64)

        H = compute_homography(pixel_points, court_points)

        # Transform the first pixel point — should map to first court point
        result = pixel_to_court(np.array([200.0, 650.0]), H)
        assert result[0] == pytest.approx(-4.115, abs=0.5)
        assert result[1] == pytest.approx(-11.885, abs=0.5)

    def test_homography_needs_at_least_4_points(self):
        """Returns None with fewer than 4 point correspondences."""
        from court_vision.court_detect import compute_homography

        pixel_points = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
        court_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        H = compute_homography(pixel_points, court_points)

        assert H is None


class TestPixelToCourt:
    def test_transforms_single_point(self):
        """Transforms a single pixel coordinate to court space."""
        from court_vision.court_detect import pixel_to_court

        # Identity-ish homography (just for shape testing)
        H = np.eye(3, dtype=np.float64)

        result = pixel_to_court(np.array([640.0, 360.0]), H)

        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestComputeHomography tests/test_court_detect.py::TestPixelToCourt -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Append to `court-vision/src/court_vision/court_detect.py`:

```python
def compute_homography(
    pixel_points: np.ndarray,
    court_points: np.ndarray,
) -> np.ndarray | None:
    """Compute the homography matrix from pixel to court coordinates.

    Requires at least 4 point correspondences.

    Args:
        pixel_points: Array of shape (N, 2) — pixel (x, y) coordinates.
        court_points: Array of shape (N, 2) — court (x, y) coordinates in meters.

    Returns:
        3x3 homography matrix, or None if computation fails.
    """
    if len(pixel_points) < 4 or len(court_points) < 4:
        return None

    H, mask = cv2.findHomography(pixel_points, court_points, method=0)

    if H is None:
        return None

    return H


def pixel_to_court(
    pixel_point: np.ndarray,
    homography: np.ndarray,
) -> tuple[float, float]:
    """Transform a pixel coordinate to court coordinates using a homography.

    Args:
        pixel_point: (x, y) pixel coordinate as numpy array.
        homography: 3x3 homography matrix from compute_homography.

    Returns:
        (x, y) court coordinate in meters.
    """
    # Convert to homogeneous coordinates
    px = np.array([pixel_point[0], pixel_point[1], 1.0], dtype=np.float64)

    # Apply homography
    transformed = homography @ px

    # Convert back from homogeneous
    w = transformed[2]
    if abs(w) < 1e-10:
        return (0.0, 0.0)

    return (float(transformed[0] / w), float(transformed[1] / w))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestComputeHomography tests/test_court_detect.py::TestPixelToCourt -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/court_detect.py court-vision/tests/test_court_detect.py
git commit -m "feat(court-vision): add homography computation and pixel-to-court transform"
```

---

### Task 6: Keypoint-to-Court-Point Matching

**Files:**
- Modify: `court-vision/src/court_vision/court_detect.py`
- Modify: `court-vision/tests/test_court_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `court-vision/tests/test_court_detect.py`:

```python
class TestMatchKeypoints:
    def test_matches_four_corners(self):
        """Matches detected pixel keypoints to their nearest court keypoint candidates."""
        from court_vision.court_detect import match_keypoints_to_court

        # Simulated pixel keypoints (4 corners of a trapezoid = court in perspective)
        pixel_keypoints = [
            (200.0, 650.0),   # near-left
            (1080.0, 650.0),  # near-right
            (880.0, 150.0),   # far-right
            (400.0, 150.0),   # far-left
        ]

        pixel_pts, court_pts = match_keypoints_to_court(pixel_keypoints)

        assert len(pixel_pts) >= 4
        assert len(court_pts) >= 4
        assert pixel_pts.shape[1] == 2
        assert court_pts.shape[1] == 2

    def test_returns_none_with_too_few_keypoints(self):
        """Returns None when fewer than 4 keypoints detected."""
        from court_vision.court_detect import match_keypoints_to_court

        pixel_keypoints = [(200.0, 650.0), (1080.0, 650.0)]

        result = match_keypoints_to_court(pixel_keypoints)

        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestMatchKeypoints -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

The matching strategy: sort pixel keypoints by position to identify the court layout (top-left, top-right, bottom-left, bottom-right pattern), then assign them to the corresponding canonical court coordinates.

Append to `court-vision/src/court_vision/court_detect.py`:

```python
def match_keypoints_to_court(
    pixel_keypoints: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Match detected pixel keypoints to canonical court coordinates.

    Uses geometric ordering: sorts keypoints by y-coordinate to separate
    "near" (bottom of image, large y) from "far" (top of image, small y),
    then by x-coordinate within each row to get left-to-right ordering.

    Matches the 4 outermost corners to singles court corners.

    Args:
        pixel_keypoints: List of (x, y) pixel coordinates from keypoint extraction.

    Returns:
        Tuple of (pixel_points, court_points) as (N, 2) arrays, or None
        if fewer than 4 keypoints are available.
    """
    if len(pixel_keypoints) < 4:
        return None

    points = np.array(pixel_keypoints, dtype=np.float64)

    # Sort by y-coordinate: top of image (far court) has small y
    sorted_by_y = points[points[:, 1].argsort()]

    # Split into "far" (top half) and "near" (bottom half)
    mid = len(sorted_by_y) // 2
    far_points = sorted_by_y[:mid]
    near_points = sorted_by_y[mid:]

    # Sort each group by x-coordinate (left to right)
    far_sorted = far_points[far_points[:, 0].argsort()]
    near_sorted = near_points[near_points[:, 0].argsort()]

    # Take outermost corners: far-left, far-right, near-left, near-right
    far_left = far_sorted[0]
    far_right = far_sorted[-1]
    near_left = near_sorted[0]
    near_right = near_sorted[-1]

    pixel_pts = np.array([near_left, near_right, far_right, far_left], dtype=np.float64)

    # Map to singles court corners
    court_pts = np.array([
        [COURT_KEYPOINTS["baseline_near_left_singles"][0],
         COURT_KEYPOINTS["baseline_near_left_singles"][1]],
        [COURT_KEYPOINTS["baseline_near_right_singles"][0],
         COURT_KEYPOINTS["baseline_near_right_singles"][1]],
        [COURT_KEYPOINTS["baseline_far_right_singles"][0],
         COURT_KEYPOINTS["baseline_far_right_singles"][1]],
        [COURT_KEYPOINTS["baseline_far_left_singles"][0],
         COURT_KEYPOINTS["baseline_far_left_singles"][1]],
    ], dtype=np.float64)

    return pixel_pts, court_pts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestMatchKeypoints -v`
Expected: PASS (all 2 tests)

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/court_detect.py court-vision/tests/test_court_detect.py
git commit -m "feat(court-vision): add keypoint-to-court-point matching"
```

---

### Task 7: Top-Level `detect_court` Orchestrator

**Files:**
- Modify: `court-vision/src/court_vision/court_detect.py`
- Modify: `court-vision/tests/test_court_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `court-vision/tests/test_court_detect.py`:

```python
from dataclasses import dataclass


class TestDetectCourt:
    def test_returns_court_detection_result(self):
        """Full detect_court returns a CourtDetectionResult with homography."""
        from court_vision.court_detect import CourtDetectionResult, detect_court

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        img = _draw_court_lines(img)

        result = detect_court(img)

        assert isinstance(result, CourtDetectionResult)
        assert isinstance(result.success, bool)

    def test_successful_detection_has_homography(self):
        """Successful detection includes a 3x3 homography matrix."""
        from court_vision.court_detect import detect_court

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        img = _draw_court_lines(img)

        result = detect_court(img)

        if result.success:
            assert result.homography is not None
            assert result.homography.shape == (3, 3)
            assert result.pixel_keypoints is not None
            assert len(result.pixel_keypoints) >= 4

    def test_failed_detection_on_blank_image(self):
        """Detection fails gracefully on an image with no court lines."""
        from court_vision.court_detect import detect_court

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)

        result = detect_court(img)

        assert result.success is False
        assert result.homography is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestDetectCourt -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Append to `court-vision/src/court_vision/court_detect.py`:

```python
from dataclasses import dataclass, field


@dataclass
class CourtDetectionResult:
    """Result of court detection on a single frame."""

    success: bool
    homography: np.ndarray | None = None
    pixel_keypoints: list[tuple[float, float]] | None = None
    num_lines_detected: int = 0


def detect_court(frame: np.ndarray) -> CourtDetectionResult:
    """Detect the tennis court in a frame and compute the homography.

    Full pipeline: detect lines → classify → extract keypoints →
    match to court → compute homography.

    Args:
        frame: BGR image as numpy array (H, W, 3).

    Returns:
        CourtDetectionResult with homography if successful.
    """
    # Step 1: Detect lines
    lines = detect_court_lines(frame)
    if not lines:
        return CourtDetectionResult(success=False, num_lines_detected=0)

    # Step 2: Classify into horizontal and vertical
    horizontal, vertical = classify_lines(lines)
    if len(horizontal) < 2 or len(vertical) < 2:
        return CourtDetectionResult(success=False, num_lines_detected=len(lines))

    # Step 3: Extract keypoints from intersections
    h, w = frame.shape[:2]
    keypoints = extract_keypoints(horizontal, vertical, frame_width=w, frame_height=h)
    if len(keypoints) < 4:
        return CourtDetectionResult(success=False, num_lines_detected=len(lines))

    # Step 4: Match pixel keypoints to court coordinates
    match_result = match_keypoints_to_court(keypoints)
    if match_result is None:
        return CourtDetectionResult(
            success=False,
            pixel_keypoints=keypoints,
            num_lines_detected=len(lines),
        )

    pixel_pts, court_pts = match_result

    # Step 5: Compute homography
    H = compute_homography(pixel_pts, court_pts)
    if H is None:
        return CourtDetectionResult(
            success=False,
            pixel_keypoints=keypoints,
            num_lines_detected=len(lines),
        )

    return CourtDetectionResult(
        success=True,
        homography=H,
        pixel_keypoints=keypoints,
        num_lines_detected=len(lines),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestDetectCourt -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/court_detect.py court-vision/tests/test_court_detect.py
git commit -m "feat(court-vision): add top-level detect_court orchestrator"
```

---

### Task 8: Per-Segment Homography Computation

**Files:**
- Modify: `court-vision/src/court_vision/court_detect.py`
- Modify: `court-vision/tests/test_court_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `court-vision/tests/test_court_detect.py`:

```python
from pathlib import Path

from court_vision.scene_filter import GameplaySegment


class TestComputeSegmentHomographies:
    def test_returns_one_result_per_segment(self, tmp_path: Path):
        """Computes one CourtDetectionResult per gameplay segment."""
        from court_vision.court_detect import compute_segment_homographies

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        # Create 10 synthetic court frames
        for i in range(10):
            img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
            img = _draw_court_lines(img)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segments = [
            GameplaySegment(start_frame=0, end_frame=4, start_time_s=0.0, end_time_s=0.13, frame_count=5),
            GameplaySegment(start_frame=7, end_frame=9, start_time_s=0.23, end_time_s=0.3, frame_count=3),
        ]

        results = compute_segment_homographies(frames_dir, segments)

        assert len(results) == 2

    def test_uses_middle_frame_of_segment(self, tmp_path: Path):
        """Samples the middle frame of each segment for homography."""
        from unittest.mock import patch as mock_patch
        from court_vision.court_detect import compute_segment_homographies, CourtDetectionResult

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        for i in range(10):
            img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segments = [
            GameplaySegment(start_frame=0, end_frame=8, start_time_s=0.0, end_time_s=0.27, frame_count=9),
        ]

        with mock_patch("court_vision.court_detect.detect_court") as mock_detect:
            mock_detect.return_value = CourtDetectionResult(success=False, num_lines_detected=0)
            compute_segment_homographies(frames_dir, segments)

            # Middle frame of segment 0-8 is frame 4
            call_args = mock_detect.call_args
            assert call_args is not None  # detect_court was called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestComputeSegmentHomographies -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Append to `court-vision/src/court_vision/court_detect.py`:

```python
from court_vision.scene_filter import GameplaySegment


def compute_segment_homographies(
    frames_dir: "Path",
    segments: list[GameplaySegment],
) -> list[CourtDetectionResult]:
    """Compute a homography for each gameplay segment.

    Samples the middle frame of each segment for court detection,
    since the camera position is generally stable within a single point.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        segments: List of gameplay segments from scene filter.

    Returns:
        List of CourtDetectionResult, one per segment.
    """
    from pathlib import Path

    results: list[CourtDetectionResult] = []

    for segment in segments:
        # Sample the middle frame of the segment
        mid_frame = (segment.start_frame + segment.end_frame) // 2
        frame_path = Path(frames_dir) / f"frame_{mid_frame:06d}.jpg"

        frame = cv2.imread(str(frame_path))
        if frame is None:
            results.append(CourtDetectionResult(success=False, num_lines_detected=0))
            continue

        result = detect_court(frame)
        results.append(result)

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd court-vision && python -m pytest tests/test_court_detect.py::TestComputeSegmentHomographies -v`
Expected: PASS (all 2 tests)

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/court_detect.py court-vision/tests/test_court_detect.py
git commit -m "feat(court-vision): add per-segment homography computation"
```

---

### Task 9: Pipeline Integration

**Files:**
- Modify: `court-vision/src/court_vision/pipeline.py`
- Modify: `court-vision/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add a new test to `court-vision/tests/test_pipeline.py`:

```python
class TestRunPipelineWithCourtDetection:
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.classify_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_pipeline_runs_court_detection_after_scene_filter(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_homographies: MagicMock,
        tmp_path: Path,
    ):
        """Pipeline calls compute_segment_homographies after scene filter."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.ingest import FrameSequence
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        mock_load_config.return_value = PipelineConfig()
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        mock_extract.return_value = FrameSequence(
            frames_dir=frames_dir, fps=30.0, total_frames=100, resolution=(1280, 720),
        )
        mock_classify.return_value = []

        segments = [
            GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
        ]
        mock_filter.return_value = segments

        court_result = CourtDetectionResult(success=True, homography=MagicMock(), num_lines_detected=6)
        mock_homographies.return_value = [court_result]

        result = run_pipeline(str(video_path), config_path=None)

        mock_homographies.assert_called_once_with(frames_dir, segments)
        assert result.court_detections is not None
        assert len(result.court_detections) == 1
        assert result.court_detections[0].success is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd court-vision && python -m pytest tests/test_pipeline.py::TestRunPipelineWithCourtDetection -v`
Expected: FAIL — `PipelineResult` has no `court_detections` attribute, `compute_segment_homographies` is not imported

- [ ] **Step 3: Update pipeline.py**

Replace the full contents of `court-vision/src/court_vision/pipeline.py`:

```python
"""Pipeline orchestrator — runs all stages in sequence."""

from dataclasses import dataclass
from pathlib import Path

from court_vision.config import PipelineConfig, load_config
from court_vision.court_detect import CourtDetectionResult, compute_segment_homographies
from court_vision.device import get_device
from court_vision.ingest import (
    FrameSequence,
    download_video,
    extract_frames,
    is_youtube_url,
)
from court_vision.scene_filter import (
    GameplaySegment,
    classify_frames,
    filter_gameplay_segments,
    load_scene_model,
)


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


def run_pipeline(
    source: str,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    scene_weights_path: Path | None = None,
) -> PipelineResult:
    """Run the Court Vision pipeline on a video source.

    Stages: video ingestion -> scene filter -> court detection.

    Args:
        source: YouTube URL or local video file path.
        config_path: Path to court-vision.yaml config. None for defaults.
        output_dir: Directory for pipeline output. None for config default.
        scene_weights_path: Path to fine-tuned scene filter weights.
                            None uses ImageNet pre-trained base.

    Returns:
        PipelineResult with frame data, gameplay segments, and court detections.
    """
    config = load_config(config_path)
    device = get_device(override=config.device)

    if output_dir is None:
        output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Video Ingestion
    if is_youtube_url(source):
        video_path = download_video(source, output_dir)
    else:
        video_path = Path(source)

    resolution = tuple(config.pipeline.target_resolution)
    frame_seq = extract_frames(video_path, target_resolution=resolution)

    # Stage 2: Scene Filter
    model = load_scene_model(scene_weights_path, device)
    results = classify_frames(frame_seq.frames_dir, frame_seq.total_frames, model, device)
    segments = filter_gameplay_segments(results, frame_seq.fps)

    gameplay_frames = sum(seg.frame_count for seg in segments)

    # Stage 3: Court Detection & Homography
    court_detections = compute_segment_homographies(frame_seq.frames_dir, segments)

    return PipelineResult(
        source=source,
        total_frames=frame_seq.total_frames,
        fps=frame_seq.fps,
        gameplay_segments=segments,
        gameplay_frame_count=gameplay_frames,
        frames_dir=frame_seq.frames_dir,
        court_detections=court_detections,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd court-vision && python -m pytest tests/test_pipeline.py -v`
Expected: PASS (all tests including old ones — old tests still pass because `court_detections` defaults to `None` when `compute_segment_homographies` is not mocked, but the old tests mock `load_config` which prevents actual execution. Update the old tests to also mock `compute_segment_homographies` to prevent import errors.)

If old pipeline tests fail because `compute_segment_homographies` is called without being mocked, add `@patch("court_vision.pipeline.compute_segment_homographies")` to the existing test methods in `TestRunPipeline`. The mock parameter goes after the other mocks but before `tmp_path`. Set `mock_homographies.return_value = []`.

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/pipeline.py court-vision/tests/test_pipeline.py
git commit -m "feat(court-vision): integrate court detection into pipeline"
```

---

### Task 10: CLI Output Update

**Files:**
- Modify: `court-vision/src/court_vision/cli.py`
- Modify: `court-vision/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add a new test to `court-vision/tests/test_cli.py`:

```python
from unittest.mock import MagicMock, patch


class TestProcessCommandOutput:
    @patch("court_vision.cli.run_pipeline")
    def test_process_shows_court_detection_summary(self, mock_pipeline: MagicMock):
        """Process command displays court detection results."""
        from court_vision.court_detect import CourtDetectionResult
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
        )

        result = runner.invoke(app, ["process", "test.mp4"])

        assert result.exit_code == 0
        assert "court" in result.output.lower() or "homography" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd court-vision && python -m pytest tests/test_cli.py::TestProcessCommandOutput -v`
Expected: FAIL — current CLI doesn't print court detection info

- [ ] **Step 3: Update cli.py**

Replace the `process` command in `court-vision/src/court_vision/cli.py`:

```python
@app.command()
def process(
    source: str = typer.Argument(help="YouTube URL or path to a local video file."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to court-vision.yaml config file."),
    scene_weights: Optional[Path] = typer.Option(None, "--scene-weights", help="Path to fine-tuned scene filter weights."),
) -> None:
    """Process a tennis match video through the CV pipeline."""
    from court_vision.pipeline import run_pipeline

    result = run_pipeline(
        source=source,
        config_path=config,
        scene_weights_path=scene_weights,
    )

    typer.echo(f"Processed {result.total_frames} frames at {result.fps:.1f} FPS")
    typer.echo(f"Found {len(result.gameplay_segments)} gameplay segments ({result.gameplay_frame_count} frames)")

    for i, seg in enumerate(result.gameplay_segments, 1):
        typer.echo(f"  Segment {i}: frames {seg.start_frame}-{seg.end_frame} ({seg.start_time_s:.1f}s - {seg.end_time_s:.1f}s)")

    if result.court_detections:
        successful = sum(1 for d in result.court_detections if d.success)
        typer.echo(f"Court detection: {successful}/{len(result.court_detections)} segments with homography")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd court-vision && python -m pytest tests/test_cli.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/cli.py court-vision/tests/test_cli.py
git commit -m "feat(court-vision): display court detection results in CLI output"
```

---

### Task 11: Run All Tests & Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `cd court-vision && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Verify imports are clean**

Run: `cd court-vision && python -c "from court_vision.court_detect import detect_court, compute_homography, pixel_to_court, CourtDetectionResult, compute_segment_homographies, COURT_KEYPOINTS; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Verify pipeline runs end-to-end (smoke test)**

Run: `cd court-vision && python -c "from court_vision.pipeline import run_pipeline; print('Pipeline importable')"`
Expected: `Pipeline importable`

- [ ] **Step 4: Final commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix(court-vision): fix any issues found during verification"
```

Only run this step if fixes were needed. If all tests passed, skip.
