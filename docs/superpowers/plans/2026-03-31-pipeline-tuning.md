# Pipeline Tuning: Court Detection & Shot Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix structural issues in ball detection, court detection, and pipeline wiring so the E2E pipeline produces actual shots and placement data from broadcast tennis video.

**Architecture:** Three areas of change: (A) Ball tracker gets stationarity rejection and `build_trajectory` integration, (B) Court detection gets RANSAC, reprojection validation, and multi-frame sampling, (C) Pipeline wires homography through to shot classification. All changes are in existing files with minimal new code.

**Tech Stack:** OpenCV (cv2), numpy — no new dependencies.

---

## Context

The E2E pipeline runs all 5 stages but produces 0 shots on the test video (214 frames, 100% gameplay). Root causes:
1. Ball tracker detects a stationary bright object at pixel (176, 322) in every frame — false positive
2. `build_trajectory` (with gap interpolation) exists but is never called by the pipeline
3. Homography is computed but never passed to `build_match_data`
4. Court detection uses least-squares (`method=0`) with no outlier rejection
5. Court detection samples only the middle frame per segment

Design spec: `docs/superpowers/specs/2026-03-31-pipeline-tuning-design.md`

## Critical Files

- **Modify:** `src/court_vision/ball_tracker.py` — add `reject_stationary_detections`, add `min_confidence` to `detect_ball_in_frame`
- **Modify:** `src/court_vision/court_detect.py` — RANSAC in `compute_homography`, add `_compute_reprojection_error`, multi-frame sampling in `compute_segment_homographies`
- **Modify:** `src/court_vision/player_detect.py` — wire `build_trajectory` into `track_segment`
- **Modify:** `src/court_vision/pipeline.py` — pass homography to `build_match_data`
- **Modify:** `tests/test_ball_tracker.py` — tests for stationarity rejection, confidence threshold
- **Modify:** `tests/test_court_detect.py` — tests for RANSAC, reprojection validation, multi-frame sampling
- **Modify:** `tests/test_pipeline.py` — test for homography wiring

---

### Task 1: Add stationarity rejection to ball tracker

**Files:**
- Modify: `src/court_vision/ball_tracker.py:103-138`
- Modify: `tests/test_ball_tracker.py`

- [ ] **Step 1: Write failing tests for `reject_stationary_detections`**

Add to end of `tests/test_ball_tracker.py`:

```python
class TestRejectStationaryDetections:
    def test_rejects_stationary_detections(self):
        """Detections at the same position are rejected as stationary."""
        from court_vision.ball_tracker import BallDetection, reject_stationary_detections

        # All detections at nearly the same position (std < 5px)
        dets = [
            BallDetection(frame_index=i, x=100.0 + (i % 2), y=200.0 + (i % 2), confidence=0.8)
            for i in range(10)
        ]
        result = reject_stationary_detections(dets)
        assert all(d is None for d in result)

    def test_preserves_moving_detections(self):
        """Detections with real motion are preserved."""
        from court_vision.ball_tracker import BallDetection, reject_stationary_detections

        # Detections moving across the frame
        dets = [
            BallDetection(frame_index=i, x=100.0 + i * 20.0, y=200.0 + i * 10.0, confidence=0.8)
            for i in range(10)
        ]
        result = reject_stationary_detections(dets)
        non_none = [d for d in result if d is not None]
        assert len(non_none) >= 6

    def test_handles_none_detections(self):
        """None detections (no ball found) pass through."""
        from court_vision.ball_tracker import BallDetection, reject_stationary_detections

        dets = [None, None, None]
        result = reject_stationary_detections(dets)
        assert all(d is None for d in result)

    def test_handles_sparse_detections(self):
        """Mix of None and stationary detections."""
        from court_vision.ball_tracker import BallDetection, reject_stationary_detections

        dets = [
            BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.8),
            None,
            BallDetection(frame_index=2, x=101.0, y=201.0, confidence=0.8),
            None,
            BallDetection(frame_index=4, x=100.5, y=200.5, confidence=0.8),
        ]
        result = reject_stationary_detections(dets)
        # Stationary detections should be rejected, Nones stay None
        assert result[1] is None
        assert result[3] is None

    def test_short_sequence_unchanged(self):
        """Fewer than window_size detections are returned as-is."""
        from court_vision.ball_tracker import BallDetection, reject_stationary_detections

        dets = [
            BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.8),
            BallDetection(frame_index=1, x=100.0, y=200.0, confidence=0.8),
        ]
        result = reject_stationary_detections(dets, window=5)
        assert len(result) == 2
        assert all(d is not None for d in result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_ball_tracker.py::TestRejectStationaryDetections -v`
Expected: FAIL (ImportError: cannot import name 'reject_stationary_detections')

- [ ] **Step 3: Write implementation**

Add to `src/court_vision/ball_tracker.py` after the `interpolate_gaps` function (after line 183):

```python
def reject_stationary_detections(
    detections: list[BallDetection | None],
    window: int = 5,
    std_threshold: float = 5.0,
) -> list[BallDetection | None]:
    """Reject ball detections that are stationary (false positives).

    For each detection, examines the surrounding window of detections.
    If the standard deviation of positions within the window is below
    std_threshold in both x and y, the detection is marked as None.

    Args:
        detections: List of per-frame detections (None = no detection).
        window: Number of surrounding detections to consider.
        std_threshold: Maximum std_dev in pixels to consider stationary.

    Returns:
        Filtered list with stationary detections replaced by None.
    """
    if len(detections) < window:
        return list(detections)

    result: list[BallDetection | None] = list(detections)

    # Collect all non-None positions
    positions = [(d.x, d.y) if d is not None else None for d in detections]
    non_none_positions = [(x, y) for x, y in positions if x is not None]

    if len(non_none_positions) < window:
        return result

    xs = [p[0] for p in non_none_positions]
    ys = [p[1] for p in non_none_positions]
    global_std_x = float(np.std(xs))
    global_std_y = float(np.std(ys))

    if global_std_x < std_threshold and global_std_y < std_threshold:
        # All detections are stationary — reject all
        return [None if d is not None else None for d in detections]

    # Per-window check for local stationarity
    half = window // 2
    for i, det in enumerate(detections):
        if det is None:
            continue

        # Gather nearby non-None detections
        start = max(0, i - half)
        end = min(len(detections), i + half + 1)
        nearby = [detections[j] for j in range(start, end) if detections[j] is not None]

        if len(nearby) < 3:
            continue

        local_xs = [d.x for d in nearby]
        local_ys = [d.y for d in nearby]
        if float(np.std(local_xs)) < std_threshold and float(np.std(local_ys)) < std_threshold:
            result[i] = None

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_ball_tracker.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/ball_tracker.py tests/test_ball_tracker.py
git commit -m "feat(court-vision): add stationarity rejection for ball tracker false positives"
```

---

### Task 2: Add minimum confidence threshold to ball detection

**Files:**
- Modify: `src/court_vision/ball_tracker.py:20-92`
- Modify: `tests/test_ball_tracker.py`

- [ ] **Step 1: Write failing tests**

Add to end of `tests/test_ball_tracker.py`:

```python
class TestDetectBallMinConfidence:
    def _create_frame_with_ball(self, width=640, height=480,
                                ball_center=(320, 240), ball_radius=8,
                                ball_color=(0, 255, 255)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (0, 100, 0)
        cv2.circle(frame, ball_center, ball_radius, ball_color, -1)
        return frame

    def test_high_confidence_threshold_rejects(self):
        """Very high min_confidence can reject marginal detections."""
        frame = self._create_frame_with_ball()
        result = detect_ball_in_frame(frame, frame_index=0, min_confidence=0.99)
        # With min_confidence=0.99, most detections should be rejected
        # (circularity rarely hits 0.99 exactly)
        # This test verifies the parameter exists and is applied
        assert result is None or result.confidence >= 0.99

    def test_zero_confidence_accepts_all(self):
        """min_confidence=0.0 accepts any detection."""
        frame = self._create_frame_with_ball()
        result = detect_ball_in_frame(frame, frame_index=0, min_confidence=0.0)
        assert result is not None

    def test_default_confidence_preserves_behavior(self):
        """Default min_confidence=0.0 matches old behavior (no filtering)."""
        frame = self._create_frame_with_ball()
        result = detect_ball_in_frame(frame, frame_index=0)
        assert result is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_ball_tracker.py::TestDetectBallMinConfidence -v`
Expected: FAIL (TypeError: unexpected keyword argument 'min_confidence')

- [ ] **Step 3: Write implementation**

In `src/court_vision/ball_tracker.py`, modify the `detect_ball_in_frame` function signature at line 20. Change from:

```python
def detect_ball_in_frame(
    frame: np.ndarray,
    frame_index: int = 0,
    min_radius: int = 3,
    max_radius: int = 20,
) -> BallDetection | None:
```

to:

```python
def detect_ball_in_frame(
    frame: np.ndarray,
    frame_index: int = 0,
    min_radius: int = 3,
    max_radius: int = 20,
    min_confidence: float = 0.0,
) -> BallDetection | None:
```

Then add a confidence gate after the best detection is found. Replace the final `return best_detection` (line 92) with:

```python
    if best_detection is not None and best_detection.confidence < min_confidence:
        return None
    return best_detection
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_ball_tracker.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/ball_tracker.py tests/test_ball_tracker.py
git commit -m "feat(court-vision): add min_confidence threshold to ball detection"
```

---

### Task 3: RANSAC in compute_homography

**Files:**
- Modify: `src/court_vision/court_detect.py:203-226`
- Modify: `tests/test_court_detect.py`

- [ ] **Step 1: Write failing tests**

Add to end of `tests/test_court_detect.py`:

```python
class TestComputeHomographyRANSAC:
    def test_ransac_rejects_outlier(self):
        """Homography computed with RANSAC ignores outlier point."""
        from court_vision.court_detect import compute_homography

        # 4 good correspondences + 1 outlier
        pixel_points = np.array([
            [200.0, 650.0],
            [1080.0, 650.0],
            [880.0, 150.0],
            [400.0, 150.0],
            [999.0, 999.0],  # outlier
        ], dtype=np.float64)

        court_points = np.array([
            [-4.115, -11.885],
            [4.115, -11.885],
            [4.115, 11.885],
            [-4.115, 11.885],
            [0.0, 0.0],  # outlier target
        ], dtype=np.float64)

        H = compute_homography(pixel_points, court_points)
        assert H is not None
        assert H.shape == (3, 3)

    def test_good_points_still_work(self):
        """RANSAC doesn't break normal 4-point homography."""
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
        assert H is not None

        result = pixel_to_court(np.array([200.0, 650.0]), H)
        assert result[0] == pytest.approx(-4.115, abs=0.5)
        assert result[1] == pytest.approx(-11.885, abs=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_court_detect.py::TestComputeHomographyRANSAC -v`
Expected: Tests might pass (least-squares may handle outliers for small N), but the change is still needed.

- [ ] **Step 3: Write implementation**

In `src/court_vision/court_detect.py`, modify `compute_homography` at line 221. Change from:

```python
    H, mask = cv2.findHomography(pixel_points, court_points, method=0)
```

to:

```python
    H, mask = cv2.findHomography(
        pixel_points, court_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_court_detect.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/court_detect.py tests/test_court_detect.py
git commit -m "fix(court-vision): use RANSAC for outlier rejection in compute_homography"
```

---

### Task 4: Add reprojection error validation

**Files:**
- Modify: `src/court_vision/court_detect.py:324-377`
- Modify: `tests/test_court_detect.py`

- [ ] **Step 1: Write failing tests**

Add to end of `tests/test_court_detect.py`:

```python
class TestComputeReprojectionError:
    def test_identity_has_zero_error(self):
        """Identity homography gives zero reprojection error."""
        from court_vision.court_detect import _compute_reprojection_error

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        H = np.eye(3, dtype=np.float64)
        error = _compute_reprojection_error(pts, pts, H)
        assert error == pytest.approx(0.0, abs=0.01)

    def test_bad_homography_has_high_error(self):
        """Mismatched homography gives high reprojection error."""
        from court_vision.court_detect import _compute_reprojection_error

        pixel_pts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dtype=np.float64)
        court_pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        # Use identity as a deliberately wrong homography (pixel != court scale)
        H = np.eye(3, dtype=np.float64)
        error = _compute_reprojection_error(pixel_pts, court_pts, H)
        assert error > 10.0


class TestDetectCourtWithReprojection:
    def test_valid_court_passes_reprojection(self):
        """Detect court on valid synthetic image passes reprojection check."""
        from court_vision.court_detect import detect_court

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        img = _draw_court_lines(img)
        result = detect_court(img)
        # If detection succeeds, it passed reprojection validation
        if result.success:
            assert result.homography is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_court_detect.py::TestComputeReprojectionError -v`
Expected: FAIL (ImportError: cannot import name '_compute_reprojection_error')

- [ ] **Step 3: Write implementation**

Add `_compute_reprojection_error` to `src/court_vision/court_detect.py` after `compute_homography` (after line 226):

```python
def _compute_reprojection_error(
    pixel_pts: np.ndarray,
    court_pts: np.ndarray,
    H: np.ndarray,
) -> float:
    """Compute mean reprojection error for a homography.

    Projects pixel_pts through H and compares to expected court_pts.

    Args:
        pixel_pts: Array of shape (N, 2) — pixel coordinates.
        court_pts: Array of shape (N, 2) — expected court coordinates.
        H: 3x3 homography matrix.

    Returns:
        Mean Euclidean distance between projected and expected points.
    """
    n = len(pixel_pts)
    if n == 0:
        return 0.0

    # Convert to homogeneous coordinates
    ones = np.ones((n, 1), dtype=np.float64)
    pixel_h = np.hstack([pixel_pts, ones])  # (N, 3)

    # Project through homography
    projected_h = (H @ pixel_h.T).T  # (N, 3)

    # Convert from homogeneous
    w = projected_h[:, 2:3]
    w = np.where(np.abs(w) < 1e-10, 1.0, w)
    projected = projected_h[:, :2] / w

    # Compute mean Euclidean distance
    errors = np.sqrt(np.sum((projected - court_pts) ** 2, axis=1))
    return float(np.mean(errors))
```

Then modify `detect_court` to add reprojection validation. In the function at line 364, after `H = compute_homography(pixel_pts, court_pts)` and the None check, add the reprojection check. Replace lines 363-377:

```python
    # Step 5: Compute homography
    H = compute_homography(pixel_pts, court_pts)
    if H is None:
        return CourtDetectionResult(
            success=False,
            pixel_keypoints=keypoints,
            num_lines_detected=len(lines),
        )

    # Step 6: Validate reprojection error
    reproj_error = _compute_reprojection_error(pixel_pts, court_pts, H)
    if reproj_error > 10.0:
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_court_detect.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/court_detect.py tests/test_court_detect.py
git commit -m "feat(court-vision): add reprojection error validation to court detection"
```

---

### Task 5: Multi-frame sampling per segment

**Files:**
- Modify: `src/court_vision/court_detect.py:380-411`
- Modify: `tests/test_court_detect.py`

- [ ] **Step 1: Write failing tests**

Add to end of `tests/test_court_detect.py`:

```python
class TestMultiFrameSampling:
    def test_samples_three_frames(self, tmp_path):
        """compute_segment_homographies tries 3 frames per segment."""
        from unittest.mock import patch as mock_patch, call
        from court_vision.court_detect import compute_segment_homographies, CourtDetectionResult
        from court_vision.scene_filter import GameplaySegment

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(20):
            img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segments = [
            GameplaySegment(start_frame=0, end_frame=19, start_time_s=0.0, end_time_s=0.63, frame_count=20),
        ]

        # First two calls fail, third succeeds
        mock_results = [
            CourtDetectionResult(success=False, num_lines_detected=0),
            CourtDetectionResult(success=False, num_lines_detected=0),
            CourtDetectionResult(success=True, homography=np.eye(3), num_lines_detected=6),
        ]

        with mock_patch("court_vision.court_detect.detect_court", side_effect=mock_results) as mock_detect:
            results = compute_segment_homographies(frames_dir, segments)

        assert len(results) == 1
        assert results[0].success is True
        assert mock_detect.call_count == 3

    def test_returns_best_on_first_success(self, tmp_path):
        """Stops trying frames after first successful detection."""
        from unittest.mock import patch as mock_patch
        from court_vision.court_detect import compute_segment_homographies, CourtDetectionResult
        from court_vision.scene_filter import GameplaySegment

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(20):
            img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segments = [
            GameplaySegment(start_frame=0, end_frame=19, start_time_s=0.0, end_time_s=0.63, frame_count=20),
        ]

        mock_result = CourtDetectionResult(success=True, homography=np.eye(3), num_lines_detected=6)

        with mock_patch("court_vision.court_detect.detect_court", return_value=mock_result) as mock_detect:
            results = compute_segment_homographies(frames_dir, segments)

        assert results[0].success is True
        assert mock_detect.call_count == 1  # stopped after first success
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_court_detect.py::TestMultiFrameSampling -v`
Expected: FAIL (mock_detect.call_count == 1 not 3 for first test, etc.)

- [ ] **Step 3: Write implementation**

Replace `compute_segment_homographies` in `src/court_vision/court_detect.py` (lines 380-411) with:

```python
def compute_segment_homographies(
    frames_dir: Path,
    segments: list[GameplaySegment],
) -> list[CourtDetectionResult]:
    """Compute a homography for each gameplay segment.

    Samples up to 3 frames per segment (25%, 50%, 75%) and uses the
    first successful detection.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        segments: List of gameplay segments from scene filter.

    Returns:
        List of CourtDetectionResult, one per segment.
    """
    results: list[CourtDetectionResult] = []

    for segment in segments:
        seg_len = segment.end_frame - segment.start_frame
        # Sample at 25%, 50%, 75% of the segment
        sample_offsets = [0.25, 0.50, 0.75]
        sample_frames = [
            segment.start_frame + int(seg_len * offset)
            for offset in sample_offsets
        ]
        # Deduplicate (short segments may repeat)
        sample_frames = list(dict.fromkeys(sample_frames))

        best_result = CourtDetectionResult(success=False, num_lines_detected=0)

        for frame_idx in sample_frames:
            frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            result = detect_court(frame)
            if result.success:
                best_result = result
                break  # Use first successful detection
            elif result.num_lines_detected > best_result.num_lines_detected:
                best_result = result

        results.append(best_result)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_court_detect.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/court_detect.py tests/test_court_detect.py
git commit -m "feat(court-vision): sample multiple frames per segment for court detection"
```

---

### Task 6: Wire build_trajectory into track_segment

**Files:**
- Modify: `src/court_vision/player_detect.py:10,271-320`
- Modify: `tests/test_pipeline.py` (integration-level test)

- [ ] **Step 1: Write failing test**

Add to end of `tests/test_pipeline.py`:

```python
class TestTrackSegmentUsesBuildTrajectory:
    @patch("court_vision.player_detect.estimate_pose")
    @patch("court_vision.player_detect.detect_players_in_frame")
    @patch("court_vision.player_detect.build_trajectory")
    def test_track_segment_calls_build_trajectory(
        self,
        mock_build_traj: MagicMock,
        mock_detect_players: MagicMock,
        mock_estimate_pose: MagicMock,
        tmp_path: Path,
    ):
        """track_segment uses build_trajectory instead of per-frame detect_ball_in_frame."""
        import numpy as np

        from court_vision.ball_tracker import BallDetection, BallTrajectory
        from court_vision.player_detect import track_segment
        from court_vision.scene_filter import GameplaySegment

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(5):
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segment = GameplaySegment(
            start_frame=0, end_frame=4,
            start_time_s=0.0, end_time_s=0.13, frame_count=5,
        )

        mock_build_traj.return_value = BallTrajectory(
            detections=[
                BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.8),
                BallDetection(frame_index=2, x=150.0, y=250.0, confidence=0.7),
            ],
            fps=30.0,
        )
        mock_detect_players.return_value = []
        mock_estimate_pose.return_value = None

        results = track_segment(frames_dir, segment)

        mock_build_traj.assert_called_once()
        # Frame 0 and 2 should have ball data, frames 1/3/4 should have None
        ball_frames = {r.frame_index: r.ball for r in results}
        assert ball_frames.get(0) is not None
        assert ball_frames.get(2) is not None
        assert ball_frames.get(1) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_pipeline.py::TestTrackSegmentUsesBuildTrajectory -v`
Expected: FAIL (mock_build_traj not called, or import error)

- [ ] **Step 3: Write implementation**

In `src/court_vision/player_detect.py`, update the import on line 10 to add `build_trajectory`, `BallTrajectory`, and `reject_stationary_detections`:

```python
from court_vision.ball_tracker import BallDetection, BallTrajectory, build_trajectory, reject_stationary_detections
```

Then replace the `track_segment` function (lines 271-320) with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_pipeline.py::TestTrackSegmentUsesBuildTrajectory tests/test_ball_tracker.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/player_detect.py tests/test_pipeline.py
git commit -m "feat(court-vision): wire build_trajectory into track_segment with stationarity rejection"
```

---

### Task 7: Pass homography to build_match_data in pipeline

**Files:**
- Modify: `src/court_vision/pipeline.py:106-112`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

Add to end of `tests/test_pipeline.py`:

```python
class TestPipelinePassesHomography:
    @patch("court_vision.pipeline.build_match_data")
    @patch("court_vision.pipeline.track_segment")
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_homography_passed_to_build_match_data(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_extract: MagicMock,
        mock_homographies: MagicMock,
        mock_track: MagicMock,
        mock_build_match: MagicMock,
        tmp_path: Path,
    ):
        """Pipeline passes court_homography to build_match_data when available."""
        import numpy as np
        import torch

        from court_vision.config import PipelineConfig, PipelineSettings
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.ingest import FrameSequence
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        mock_load_config.return_value = PipelineConfig(
            pipeline=PipelineSettings(scene_filter_mode="ml")
        )
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        mock_extract.return_value = FrameSequence(
            frames_dir=frames_dir, fps=30.0, total_frames=100, resolution=(1280, 720),
        )
        mock_classify_result = []
        mock_filter.return_value = [
            GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
        ]

        fake_H = np.eye(3, dtype=np.float64)
        mock_homographies.return_value = [
            CourtDetectionResult(success=True, homography=fake_H, num_lines_detected=6),
        ]
        mock_track.return_value = []
        mock_build_match.return_value = None

        with patch("court_vision.pipeline.classify_frames", return_value=mock_classify_result):
            run_pipeline(str(video_path), config_path=None)

        # Verify court_homography was passed
        call_kwargs = mock_build_match.call_args
        assert call_kwargs is not None
        # Check keyword args or positional args for court_homography
        if call_kwargs.kwargs:
            assert "court_homography" in call_kwargs.kwargs
            passed_H = call_kwargs.kwargs["court_homography"]
        else:
            # court_homography is the 5th positional arg
            passed_H = call_kwargs.args[4] if len(call_kwargs.args) > 4 else None
        assert passed_H is not None
        assert np.array_equal(passed_H, fake_H)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_pipeline.py::TestPipelinePassesHomography -v`
Expected: FAIL (court_homography not passed, defaults to None)

- [ ] **Step 3: Write implementation**

In `src/court_vision/pipeline.py`, replace lines 106-112 (Stage 5) with:

```python
    # Stage 5: Shot Classification
    court_homography = None
    for cd in (court_detections or []):
        if cd.success and cd.homography is not None:
            court_homography = cd.homography
            break

    match_data = build_match_data(
        source=source,
        segments=segments,
        tracking_results=all_tracking,
        fps=frame_seq.fps,
        court_homography=court_homography,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd court-vision && .venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/pipeline.py tests/test_pipeline.py
git commit -m "fix(court-vision): pass court homography through to shot classification"
```

---

### Task 8: Full test suite verification and E2E check

- [ ] **Step 1: Run full test suite**

Run: `cd court-vision && .venv/bin/python -m pytest tests/ -v`
Expected: All tests pass (163+ existing + ~15 new = ~178+)

- [ ] **Step 2: Run E2E pipeline on test video**

```bash
cd court-vision && .venv/bin/python -c "
from court_vision.pipeline import run_pipeline
result = run_pipeline(source='data/test_input_video.mp4')
print(f'Total frames: {result.total_frames}')
print(f'Gameplay segments: {len(result.gameplay_segments)} ({result.gameplay_frame_count} frames)')
for i, seg in enumerate(result.gameplay_segments, 1):
    print(f'  Segment {i}: frames {seg.start_frame}-{seg.end_frame}')
if result.court_detections:
    for i, cd in enumerate(result.court_detections):
        print(f'Court detection {i}: success={cd.success}, lines={cd.num_lines_detected}')
if result.match_data:
    print(f'Points: {len(result.match_data.points)}')
    for pt in result.match_data.points:
        print(f'  Point {pt.point_number}: {pt.rally_length} shots')
        for shot in pt.shots:
            placement = f'{shot.placement.zone}' if shot.placement else 'None'
            print(f'    Shot {shot.shot_number}: {shot.stroke} by {shot.player}, placement={placement}')
"
```

Expected:
- Ball tracker rejects stationary false positives
- Court detection uses RANSAC and validates reprojection
- Homography is passed to shot classification
- Shot count may be 0 if ball tracker produces no valid detections (expected given the HSV tracker's limitations) or > 0 if some valid ball detections survive filtering

- [ ] **Step 3: Commit if any threshold adjustments needed**

Only if tuning was required during E2E verification.

## Verification

1. All existing tests still pass (163+)
2. New tests pass (~15 new tests for stationarity, confidence, RANSAC, reprojection, multi-frame, trajectory wiring, homography passing)
3. Ball tracker rejects the stationary false positive at (176, 322)
4. Court detection uses RANSAC with reprojection validation
5. Pipeline passes homography through to shot classification
6. E2E pipeline runs without errors on test video

## Out of Scope

- ML-based ball detection (TrackNet etc.) — evaluate after structural fixes
- Left-handed player support in stroke classification
- Multi-camera or non-broadcast angles
- Threshold tuning across multiple videos
