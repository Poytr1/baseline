# TrackNet v2 Ball Detection

**Date:** 2026-04-01
**Goal:** Replace the HSV+contour ball detector with TrackNet v2 (pretrained) so the pipeline produces real ball detections that survive stationarity filtering and enable shot classification.

## Problem Statement

The current HSV+contour ball tracker produces only false positives on broadcast tennis video. After stationarity filtering, 0 usable detections remain. The pipeline therefore finds 0 ball-player contacts and classifies 0 shots.

Diagnostic evidence from the test video (214 frames at 30fps):

| Metric | Value |
|--------|-------|
| Raw ball detections (HSV) | 214/214 (100%) |
| After stationarity filter | 59/214 (27%) |
| Real ball-player contacts | 0 |
| Shots classified | 0 |

The 59 surviving detections are non-stationary noise, not real ball tracks. The HSV detector cannot distinguish a tennis ball from other bright circular objects in broadcast footage.

## Design

### A. TrackNet v2 Model Module

**File:** `src/court_vision/tracknet.py`

TrackNet v2 is a U-Net variant designed for tennis ball tracking. It takes 3 consecutive RGB frames (concatenated along the channel dimension, 9 channels total), resized to 640x360, and outputs a 640x360 probability heatmap. The heatmap peak indicates ball position.

#### A1. Model Architecture

The TrackNet v2 architecture is a compact encoder-decoder network (~150 lines of PyTorch):

- **Encoder**: VGG-style convolutional blocks with batch normalization and max pooling. Input is 9-channel (3 frames x 3 RGB channels). Progressively downsamples spatial resolution while increasing channel depth.
- **Decoder**: Transposed convolutions with skip connections from encoder layers. Upsamples back to input resolution (640x360). Final layer outputs a single-channel heatmap via sigmoid activation.
- **Output**: 640x360 heatmap where values represent probability of ball presence. Peak location = ball position, peak value = confidence.

#### A2. Weight Management

Pretrained weights stored at `~/.cache/court-vision/models/tracknet_v2.pt`. Auto-downloaded on first use via `_download_tracknet_weights()`. This matches the existing MediaPipe pose model pattern.

The `_get_tracknet_model()` function is a singleton loader (matching `_get_yolo_model()` and `_get_pose_estimator()` patterns). It:

1. Checks cache directory for weights file
2. Downloads if missing (with progress logging)
3. Instantiates model architecture
4. Loads state_dict
5. Sets eval mode
6. Moves to device via existing `get_device()`
7. Returns cached model instance on subsequent calls

If download fails, raises an error. The pipeline handles this by falling back to HSV detection with a warning log.

#### A3. Inference Function

```python
def detect_ball_tracknet(
    frames: list[np.ndarray],
    frame_index: int,
    confidence_threshold: float = 0.5,
) -> BallDetection | None:
```

- Takes exactly 3 frames (BGR, any resolution — internally resized to 640x360)
- Concatenates along channel dimension → 9-channel tensor
- Runs inference → 640x360 heatmap
- Finds peak via `np.unravel_index(np.argmax(heatmap))`
- If peak value < confidence_threshold → returns None
- Scales peak coordinates back to original frame resolution
- Returns `BallDetection(frame_index, x, y, confidence=peak_value)`

#### A4. Heatmap Extraction

```python
def _extract_ball_position(
    heatmap: np.ndarray,
    original_width: int,
    original_height: int,
    confidence_threshold: float = 0.5,
) -> tuple[float, float, float] | None:
```

Separated from inference for testability. Takes raw heatmap, returns (x, y, confidence) in original image coordinates, or None if below threshold.

### B. Integration into build_trajectory

**File:** `src/court_vision/ball_tracker.py`

#### B1. Method Parameter

`build_trajectory` gains a `method: str = "tracknet"` parameter:

```python
def build_trajectory(
    frames_dir: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    max_gap_s: float = 0.5,
    method: str = "tracknet",
) -> BallTrajectory:
```

When `method="tracknet"`, uses the 3-frame sliding window with `detect_ball_tracknet`. When `method="hsv"`, uses the existing `detect_ball_in_frame`. Default is `"tracknet"`.

#### B2. 3-Frame Sliding Window

The existing sequential frame loop reads frames from disk one by one. For TrackNet, we maintain a 3-frame buffer:

- Frame buffer: `[frame_{i-2}, frame_{i-1}, frame_i]`
- For the first 2 frames (i=0, i=1), pad missing earlier frames with black frames (np.zeros). TrackNet produces low-confidence output for these, effectively returning None.
- For each frame position, pass the 3-frame buffer to `detect_ball_tracknet()`

The HSV path (`method="hsv"`) bypasses the buffer and calls `detect_ball_in_frame` per frame as before.

#### B3. No Other Changes

`interpolate_gaps`, `reject_stationary_detections`, and `map_ball_to_court` remain unchanged. They operate on `BallDetection` objects regardless of source.

### C. Config & Pipeline Threading

**File:** `src/court_vision/config.py`

Add to `PipelineSettings`:

```python
ball_detection_method: str = "tracknet"  # "tracknet" or "hsv"
```

**File:** `src/court_vision/pipeline.py`

Pass `config.pipeline.ball_detection_method` through to `track_segment`.

**File:** `src/court_vision/player_detect.py`

`track_segment` gains a `ball_method: str = "tracknet"` parameter, threaded to `build_trajectory(method=ball_method)`.

### D. Device Handling

TrackNet uses the existing `get_device()` utility from `device.py`. On M2 Pro Macs, this returns `mps` (Metal Performance Shaders) when available, falling back to `cpu`. The model and input tensors are moved to the selected device.

## Files Changed

| File | Change |
|------|--------|
| `src/court_vision/tracknet.py` | New. Model architecture, weight loading, `detect_ball_tracknet()`, heatmap extraction |
| `src/court_vision/ball_tracker.py` | Add `method` param to `build_trajectory`, 3-frame sliding window |
| `src/court_vision/config.py` | Add `ball_detection_method` to `PipelineSettings` |
| `src/court_vision/pipeline.py` | Thread `ball_detection_method` to `track_segment` |
| `src/court_vision/player_detect.py` | Thread `ball_method` param to `build_trajectory` |
| `tests/test_tracknet.py` | New. Model shape tests, heatmap extraction, confidence thresholding |
| `tests/test_ball_tracker.py` | Tests for method param, sliding window, HSV fallback |
| `tests/test_pipeline.py` | Test config threading for `ball_detection_method` |

## Success Criteria

1. All existing tests still pass (180+)
2. TrackNet model loads and runs inference on M2 Pro (MPS or CPU)
3. `build_trajectory(method="tracknet")` produces ball detections on the test video
4. Stationarity filter preserves real ball detections (non-zero after filtering)
5. Pipeline produces > 0 shots on the test video
6. `method="hsv"` fallback preserves existing behavior
7. Config `ball_detection_method` threads through pipeline correctly

## Out of Scope

- Training infrastructure or custom weight fine-tuning
- Trajectory interpolation upgrades (polynomial, physics-aware)
- Multi-ball detection (single ball per frame assumption preserved)
- Left-handed player support in stroke classification
- Batch GPU inference optimization (sequential frame processing is sufficient)
