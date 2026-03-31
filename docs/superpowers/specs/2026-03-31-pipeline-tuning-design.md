# Pipeline Tuning: Court Detection & Shot Classification

**Date:** 2026-03-31
**Goal:** Fix structural issues in ball detection, court detection, and shot classification so the E2E pipeline produces actual shots and placement data from broadcast tennis video.

## Problem Statement

The E2E pipeline runs all 5 stages but produces 0 shots on the test video (214 frames, 100% gameplay). Root cause analysis:

1. **Ball detection is a false positive.** The HSV+contour ball tracker latches onto a fixed bright object at pixel (176, 322) in every frame. No frame-to-frame motion = not a real ball.
2. **`build_trajectory` is never called.** The pipeline calls `detect_ball_in_frame` per-frame but skips `build_trajectory`, losing gap interpolation that could fill in sparse real detections.
3. **Homography is never passed to shot classification.** `build_match_data` receives `court_homography=None` even when court detection succeeds, so placement is always `None`.
4. **Court detection uses no outlier rejection.** `compute_homography` uses `method=0` (least squares). One bad keypoint corrupts the entire homography.
5. **No reprojection validation.** A geometrically invalid homography propagates silently.
6. **Single frame sampled per segment.** If the middle frame is blurred or occluded, the entire segment has no homography.

## Diagnostic Evidence

From the test video (`data/test_input_video.mp4`, 214 frames at 30fps):

| Metric | Value |
|--------|-------|
| Ball detected | 214/214 frames (100%) |
| Unique ball X positions | 2 (166, 176) |
| Unique ball Y positions | 3 (321, 322, 646) |
| Ball near player (dist<150) | 0 frames |
| Players detected (2+) | 213/214 frames |
| Court detection | Success, 41 lines |
| Contacts found | 0 |
| Shots classified | 0 |

The ball tracker has near-zero position variance across 214 frames — it is detecting a stationary scene element, not a moving tennis ball.

## Design

### A. Ball Detection Fixes

**File:** `src/court_vision/ball_tracker.py`

#### A1. Motion-based false positive rejection

Add a `_is_stationary` check: track the last N ball positions (default N=5). If the standard deviation of positions over the window is below a threshold (default 5px), reject the detection as stationary.

This requires `detect_ball_in_frame` to accept optional previous detections, or — better — move the stationarity check into `build_trajectory` which already has the full sequence.

**Approach:** Add a post-processing step `reject_stationary_detections` that runs after all per-frame detections. For each detection, if the surrounding window (±2 frames) has std_dev < 5px in both x and y, mark it as `None`. This keeps `detect_ball_in_frame` stateless and puts the filtering in `build_trajectory`.

#### A2. Minimum confidence threshold

Add `min_confidence: float = 0.5` parameter to `detect_ball_in_frame`. Detections below this confidence are returned as `None`. The current code has no confidence gate.

#### A3. Wire `build_trajectory` into pipeline

In `player_detect.py`, `track_segment` currently calls `detect_ball_in_frame` per frame. Change it to:
1. Collect raw per-frame ball detections
2. Call `build_trajectory` (which does gap interpolation)
3. Apply `reject_stationary_detections`
4. Return the filtered trajectory

`build_trajectory` already reads frames from disk and calls `detect_ball_in_frame` internally. `track_segment` should call `build_trajectory` once for the segment's frame range, then zip the ball results with player detections, instead of calling `detect_ball_in_frame` per frame separately. Add `reject_stationary_detections` as a post-processing step inside `build_trajectory` after raw detection and before interpolation.

### B. Court Detection Robustness

**File:** `src/court_vision/court_detect.py`

#### B1. RANSAC in compute_homography

Change `method=0` to `method=cv2.RANSAC` with `ransacReprojThreshold=5.0`. This rejects outlier keypoint matches.

```python
H, mask = cv2.findHomography(pixel_points, court_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)
```

#### B2. Reprojection error validation

After computing H, project the matched pixel points through the homography and compare to expected court points. If mean reprojection error > 10px, mark `success=False`.

Add a helper `_compute_reprojection_error(pixel_pts, court_pts, H) -> float` and call it in `detect_court`.

#### B3. Multi-frame sampling per segment

In `compute_segment_homographies`, instead of sampling only the middle frame, sample 3 frames (25%, 50%, 75% of the segment). Run `detect_court` on each. Pick the result with the lowest reprojection error (or the first successful one).

### C. Pipeline Wiring

**File:** `src/court_vision/pipeline.py`

#### C1. Pass homography to build_match_data

The pipeline already computes `court_detections` but never passes the homography to `build_match_data`. Fix:

```python
# Find the best homography from successful court detections
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

#### C2. Use build_trajectory in track_segment

Wire `build_trajectory` (with stationarity rejection) into `track_segment` so the pipeline gets interpolated, filtered ball data.

## Files Changed

| File | Change |
|------|--------|
| `src/court_vision/ball_tracker.py` | Add `reject_stationary_detections`, add `min_confidence` param |
| `src/court_vision/court_detect.py` | RANSAC in `compute_homography`, add `_compute_reprojection_error`, multi-frame sampling |
| `src/court_vision/player_detect.py` | Wire `build_trajectory` into `track_segment` |
| `src/court_vision/pipeline.py` | Pass homography to `build_match_data` |
| `tests/test_ball_tracker.py` | Tests for stationarity rejection, confidence threshold |
| `tests/test_court_detect.py` | Tests for RANSAC, reprojection validation, multi-frame sampling |
| `tests/test_pipeline.py` | Update pipeline test for homography wiring |

## Success Criteria

1. All existing tests still pass (163+)
2. Ball tracker rejects stationary false positives on test video
3. Court detection uses RANSAC and validates reprojection error
4. Pipeline passes homography through to shot classification
5. E2E run on test video produces > 0 shots (exact count depends on video content)
6. Placement data is populated for detected shots (not `None`)

## Out of Scope

- ML-based ball detection (TrackNet etc.) — evaluate after structural fixes
- Left-handed player support in stroke classification
- Multi-camera or non-broadcast angles
- Threshold tuning across multiple videos (will need additional test data)
