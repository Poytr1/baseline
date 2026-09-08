# TrackNet v2 Ball Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the HSV+contour ball detector with TrackNet v2 (pretrained) so the pipeline produces real ball detections that survive stationarity filtering and enable shot classification.

**Architecture:** New `tracknet.py` module implements the TrackNet v2 encoder-decoder architecture, singleton weight loading with auto-download, and a `detect_ball_tracknet()` function. `build_trajectory` gains a `method` parameter to switch between `"tracknet"` (3-frame sliding window) and `"hsv"` (existing per-frame). Config and pipeline thread the method parameter end-to-end.

**Tech Stack:** PyTorch (model architecture, inference), OpenCV + NumPy (frame I/O, preprocessing), existing `device.py` for MPS/CUDA/CPU selection.

---

## Context

The current HSV+contour ball tracker produces only false positives on broadcast tennis video. After stationarity filtering, 0 usable detections remain and the pipeline classifies 0 shots. TrackNet v2 is a U-Net variant designed for tennis ball tracking that takes 3 consecutive frames and outputs a probability heatmap.

## Critical Files

- **Create:** `src/court_vision/tracknet.py` — model architecture, weight management, inference
- **Create:** `tests/test_tracknet.py` — model shape tests, heatmap extraction, inference
- **Modify:** `src/court_vision/ball_tracker.py:107-142` — add `method` param, 3-frame sliding window
- **Modify:** `src/court_vision/config.py:10-16` — add `ball_detection_method` field
- **Modify:** `src/court_vision/pipeline.py:98-104` — thread `ball_detection_method` to `track_segment`
- **Modify:** `src/court_vision/player_detect.py:271-294` — add `ball_method` param, thread to `build_trajectory`
- **Modify:** `tests/test_ball_tracker.py` — tests for method param, sliding window
- **Modify:** `tests/test_pipeline.py` — test config threading for ball_detection_method

---

### Task 1: Create tracknet.py with model architecture and heatmap extraction

**Files:**
- Create: `src/court_vision/tracknet.py`
- Create: `tests/test_tracknet.py`

- [ ] **Step 1: Write failing tests for model architecture and heatmap extraction**

Create `tests/test_tracknet.py`:

```python
"""Tests for TrackNet v2 ball detection module."""

import numpy as np
import pytest
import torch

from court_vision.tracknet import TrackNetV2, _extract_ball_position


class TestTrackNetV2Architecture:
    def test_model_accepts_9_channel_input(self):
        """TrackNet v2 takes 9-channel input (3 frames x 3 RGB)."""
        model = TrackNetV2()
        model.eval()
        x = torch.randn(1, 9, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 360, 640)

    def test_output_range_is_0_to_1(self):
        """Output heatmap values are in [0, 1] range (sigmoid)."""
        model = TrackNetV2()
        model.eval()
        x = torch.randn(1, 9, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_model_handles_batch_size_1(self):
        """Model works with single-image batch."""
        model = TrackNetV2()
        model.eval()
        x = torch.randn(1, 9, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1


class TestExtractBallPosition:
    def test_extracts_peak_from_heatmap(self):
        """Peak in heatmap is returned as ball position."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        heatmap[180, 320] = 0.95  # peak at center
        result = _extract_ball_position(heatmap, original_width=1280, original_height=720)
        assert result is not None
        x, y, conf = result
        # 320/640 * 1280 = 640, 180/360 * 720 = 360
        assert abs(x - 640.0) < 1.0
        assert abs(y - 360.0) < 1.0
        assert abs(conf - 0.95) < 0.01

    def test_returns_none_below_threshold(self):
        """Returns None when peak value is below confidence threshold."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        heatmap[180, 320] = 0.3  # below default 0.5 threshold
        result = _extract_ball_position(heatmap, original_width=1280, original_height=720)
        assert result is None

    def test_custom_threshold(self):
        """Custom confidence threshold is respected."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        heatmap[180, 320] = 0.3
        result = _extract_ball_position(
            heatmap, original_width=1280, original_height=720, confidence_threshold=0.2,
        )
        assert result is not None
        _, _, conf = result
        assert abs(conf - 0.3) < 0.01

    def test_scales_to_original_resolution(self):
        """Coordinates are scaled from 640x360 heatmap to original frame resolution."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        heatmap[90, 160] = 0.9  # top-left quadrant
        result = _extract_ball_position(heatmap, original_width=1920, original_height=1080)
        assert result is not None
        x, y, _ = result
        # 160/640 * 1920 = 480, 90/360 * 1080 = 270
        assert abs(x - 480.0) < 1.0
        assert abs(y - 270.0) < 1.0

    def test_all_zeros_returns_none(self):
        """All-zero heatmap returns None (peak=0 < threshold)."""
        heatmap = np.zeros((360, 640), dtype=np.float32)
        result = _extract_ball_position(heatmap, original_width=1280, original_height=720)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tracknet.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write implementation for model architecture and heatmap extraction**

Create `src/court_vision/tracknet.py`:

```python
"""TrackNet v2 ball detection — model architecture, weight loading, and inference."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# TrackNet v2 input/output resolution
TRACKNET_WIDTH = 640
TRACKNET_HEIGHT = 360


class TrackNetV2(nn.Module):
    """TrackNet v2 encoder-decoder for tennis ball detection.

    Input: 9-channel tensor (3 consecutive RGB frames concatenated).
    Output: 1-channel 640x360 heatmap (ball probability).
    """

    def __init__(self):
        super().__init__()

        # Encoder (VGG-style blocks with batch norm)
        self.encoder1 = self._conv_block(9, 64, 2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = self._conv_block(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = self._conv_block(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Decoder (transposed convolutions with skip connections)
        self.decoder3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.decoder3_conv = self._conv_block(256 + 128, 128, 2)

        self.decoder2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.decoder2_conv = self._conv_block(128 + 64, 64, 2)

        self.decoder1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1_conv = self._conv_block(64, 32, 2)

        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_ch: int, out_ch: int, num_layers: int) -> nn.Sequential:
        """VGG-style conv block: (Conv3x3 + BN + ReLU) x num_layers."""
        layers: list[nn.Module] = []
        for i in range(num_layers):
            ch_in = in_ch if i == 0 else out_ch
            layers.extend([
                nn.Conv2d(ch_in, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.encoder1(x)        # (B, 64, H, W)
        e2 = self.encoder2(self.pool1(e1))  # (B, 128, H/2, W/2)
        e3 = self.encoder3(self.pool2(e2))  # (B, 256, H/4, W/4)

        # Decoder with skip connections
        d3 = self.decoder3(e3)        # (B, 256, H/2, W/2)
        d3 = torch.cat([d3, e2], dim=1)  # (B, 384, H/2, W/2)
        d3 = self.decoder3_conv(d3)   # (B, 128, H/2, W/2)

        d2 = self.decoder2(d3)        # (B, 128, H, W)
        d2 = torch.cat([d2, e1], dim=1)  # (B, 192, H, W)
        d2 = self.decoder2_conv(d2)   # (B, 64, H, W)

        d1 = self.decoder1(d2)        # (B, 64, 2H, 2W) — but we started at H, so this is H*2
        d1 = self.decoder1_conv(d1)   # (B, 32, 2H, 2W)

        out = self.sigmoid(self.final(d1))  # (B, 1, 2H, 2W)

        # Resize back to input spatial dims if needed
        if out.shape[2:] != x.shape[2:]:
            out = nn.functional.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


def _extract_ball_position(
    heatmap: np.ndarray,
    original_width: int,
    original_height: int,
    confidence_threshold: float = 0.5,
) -> tuple[float, float, float] | None:
    """Extract ball position from a TrackNet heatmap.

    Args:
        heatmap: 2D array (H, W) with values in [0, 1].
        original_width: Width of the original video frame.
        original_height: Height of the original video frame.
        confidence_threshold: Minimum peak value to accept.

    Returns:
        (x, y, confidence) in original frame coordinates, or None.
    """
    peak_value = float(np.max(heatmap))
    if peak_value < confidence_threshold:
        return None

    peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    heatmap_y, heatmap_x = peak_idx

    # Scale from heatmap resolution to original frame resolution
    x = float(heatmap_x) / heatmap.shape[1] * original_width
    y = float(heatmap_y) / heatmap.shape[0] * original_height

    return (x, y, peak_value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tracknet.py -v`
Expected: PASS (8/8)

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/tracknet.py tests/test_tracknet.py
git commit -m "feat(court-vision): add TrackNet v2 model architecture and heatmap extraction"
```

---

### Task 2: Add weight management and detect_ball_tracknet inference function

**Files:**
- Modify: `src/court_vision/tracknet.py`
- Modify: `tests/test_tracknet.py`

- [ ] **Step 1: Write failing tests for weight loading and inference**

Add to `tests/test_tracknet.py`:

```python
from unittest.mock import MagicMock, patch

from court_vision.ball_tracker import BallDetection
from court_vision.tracknet import _get_tracknet_model, detect_ball_tracknet


class TestGetTracknetModel:
    @patch("court_vision.tracknet._download_tracknet_weights")
    def test_returns_model_instance(self, mock_download):
        """Returns a TrackNetV2 model in eval mode."""
        mock_download.return_value = None
        # Clear the lru_cache so our mock takes effect
        _get_tracknet_model.cache_clear()
        with patch("court_vision.tracknet.Path.exists", return_value=False), \
             patch("court_vision.tracknet.torch.load", return_value={}), \
             patch.object(TrackNetV2, "load_state_dict"):
            model = _get_tracknet_model()
        assert isinstance(model, TrackNetV2)
        assert not model.training
        _get_tracknet_model.cache_clear()

    def test_caches_model_on_second_call(self):
        """Second call returns same model instance (singleton)."""
        _get_tracknet_model.cache_clear()
        with patch("court_vision.tracknet._download_tracknet_weights"), \
             patch("court_vision.tracknet.Path.exists", return_value=False), \
             patch("court_vision.tracknet.torch.load", return_value={}), \
             patch.object(TrackNetV2, "load_state_dict"):
            model1 = _get_tracknet_model()
            model2 = _get_tracknet_model()
        assert model1 is model2
        _get_tracknet_model.cache_clear()


class TestDetectBallTracknet:
    def test_returns_ball_detection_on_strong_signal(self):
        """Returns BallDetection when model produces high-confidence peak."""
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

        fake_heatmap = torch.zeros(1, 1, 360, 640)
        fake_heatmap[0, 0, 180, 320] = 0.9

        mock_model = MagicMock()
        mock_model.return_value = fake_heatmap
        mock_model.eval = MagicMock(return_value=mock_model)

        with patch("court_vision.tracknet._get_tracknet_model", return_value=mock_model), \
             patch("court_vision.tracknet.get_device", return_value=torch.device("cpu")):
            result = detect_ball_tracknet(frames, frame_index=5)

        assert result is not None
        assert isinstance(result, BallDetection)
        assert result.frame_index == 5
        assert result.confidence > 0.5

    def test_returns_none_on_low_confidence(self):
        """Returns None when model output is below threshold."""
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

        fake_heatmap = torch.zeros(1, 1, 360, 640)
        fake_heatmap[0, 0, 180, 320] = 0.2  # below threshold

        mock_model = MagicMock()
        mock_model.return_value = fake_heatmap

        with patch("court_vision.tracknet._get_tracknet_model", return_value=mock_model), \
             patch("court_vision.tracknet.get_device", return_value=torch.device("cpu")):
            result = detect_ball_tracknet(frames, frame_index=0)

        assert result is None

    def test_requires_exactly_3_frames(self):
        """Raises ValueError if not exactly 3 frames provided."""
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(2)]
        with pytest.raises(ValueError, match="3 frames"):
            detect_ball_tracknet(frames, frame_index=0)

    def test_custom_confidence_threshold(self):
        """Custom confidence_threshold is respected."""
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

        fake_heatmap = torch.zeros(1, 1, 360, 640)
        fake_heatmap[0, 0, 180, 320] = 0.3

        mock_model = MagicMock()
        mock_model.return_value = fake_heatmap

        with patch("court_vision.tracknet._get_tracknet_model", return_value=mock_model), \
             patch("court_vision.tracknet.get_device", return_value=torch.device("cpu")):
            result = detect_ball_tracknet(frames, frame_index=0, confidence_threshold=0.2)

        assert result is not None
        assert result.confidence > 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tracknet.py::TestGetTracknetModel tests/test_tracknet.py::TestDetectBallTracknet -v`
Expected: FAIL (functions not found)

- [ ] **Step 3: Write implementation for weight loading and inference**

Add to `src/court_vision/tracknet.py` (after the existing code):

```python
import logging

import cv2

from court_vision.ball_tracker import BallDetection
from court_vision.device import get_device

logger = logging.getLogger(__name__)

_WEIGHTS_URL = "https://github.com/yastrebksv/TrackNet/releases/download/v2.0/tracknet_v2.pt"
_CACHE_DIR = Path.home() / ".cache" / "court-vision" / "models"
_WEIGHTS_FILENAME = "tracknet_v2.pt"


def _download_tracknet_weights() -> Path:
    """Download pretrained TrackNet v2 weights if not cached.

    Returns:
        Path to the weights file.
    """
    import urllib.request

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = _CACHE_DIR / _WEIGHTS_FILENAME

    if weights_path.exists():
        return weights_path

    logger.info("Downloading TrackNet v2 weights to %s...", weights_path)
    urllib.request.urlretrieve(_WEIGHTS_URL, str(weights_path))
    logger.info("TrackNet v2 weights downloaded.")

    return weights_path


@lru_cache(maxsize=1)
def _get_tracknet_model() -> TrackNetV2:
    """Load the TrackNet v2 model (cached singleton).

    Downloads weights on first use. Moves model to best available device.
    """
    weights_path = _download_tracknet_weights()
    device = get_device()

    model = TrackNetV2()

    if weights_path.exists():
        try:
            state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            logger.info("TrackNet v2 weights loaded from %s", weights_path)
        except Exception as e:
            logger.warning("Failed to load TrackNet weights: %s. Using random initialization.", e)

    model = model.to(device)
    model.eval()
    return model


def detect_ball_tracknet(
    frames: list[np.ndarray],
    frame_index: int,
    confidence_threshold: float = 0.5,
) -> BallDetection | None:
    """Detect the tennis ball using TrackNet v2.

    Takes exactly 3 consecutive BGR frames, runs TrackNet inference,
    and returns ball position from the heatmap peak.

    Args:
        frames: Exactly 3 BGR frames (any resolution, internally resized to 640x360).
        frame_index: Frame index for the detection result (typically the last frame).
        confidence_threshold: Minimum heatmap peak value to accept.

    Returns:
        BallDetection with pixel coordinates in original resolution, or None.

    Raises:
        ValueError: If not exactly 3 frames provided.
    """
    if len(frames) != 3:
        raise ValueError(f"detect_ball_tracknet requires exactly 3 frames, got {len(frames)}")

    original_height, original_width = frames[0].shape[:2]
    device = get_device()
    model = _get_tracknet_model()

    # Preprocess: resize to 640x360, convert BGR->RGB, concatenate along channels
    processed: list[np.ndarray] = []
    for frame in frames:
        resized = cv2.resize(frame, (TRACKNET_WIDTH, TRACKNET_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed.append(rgb)

    # Stack: (3, H, W, 3) -> concatenate channel dim -> (H, W, 9) -> (9, H, W)
    concatenated = np.concatenate(processed, axis=2)  # (360, 640, 9)
    tensor = torch.from_numpy(concatenated).permute(2, 0, 1).float() / 255.0  # (9, 360, 640)
    tensor = tensor.unsqueeze(0).to(device)  # (1, 9, 360, 640)

    # Inference
    with torch.no_grad():
        heatmap_tensor = model(tensor)

    heatmap = heatmap_tensor[0, 0].cpu().numpy()  # (360, 640)

    result = _extract_ball_position(
        heatmap, original_width, original_height, confidence_threshold,
    )
    if result is None:
        return None

    x, y, confidence = result
    return BallDetection(
        frame_index=frame_index,
        x=x,
        y=y,
        confidence=confidence,
    )
```

Note: The imports `logging`, `cv2`, `BallDetection`, and `get_device` should be added at the top of the file alongside the existing imports. The final file should have all imports at the top.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tracknet.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/tracknet.py tests/test_tracknet.py
git commit -m "feat(court-vision): add TrackNet v2 weight loading and inference function"
```

---

### Task 3: Add `method` parameter to `build_trajectory` with 3-frame sliding window

**Files:**
- Modify: `src/court_vision/ball_tracker.py:107-142`
- Modify: `tests/test_ball_tracker.py`

- [ ] **Step 1: Write failing tests for method parameter and sliding window**

Add to end of `tests/test_ball_tracker.py`:

```python
from unittest.mock import MagicMock, patch


class TestBuildTrajectoryMethod:
    def test_hsv_method_uses_detect_ball_in_frame(self, tmp_path):
        """method='hsv' calls detect_ball_in_frame per frame."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        with patch("court_vision.ball_tracker.detect_ball_in_frame", return_value=None) as mock_hsv:
            build_trajectory(frames_dir, 0, 2, fps=30.0, method="hsv")

        assert mock_hsv.call_count == 3

    def test_tracknet_method_uses_detect_ball_tracknet(self, tmp_path):
        """method='tracknet' calls detect_ball_tracknet with 3-frame windows."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(5):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        with patch("court_vision.ball_tracker.detect_ball_tracknet", return_value=None) as mock_tn:
            build_trajectory(frames_dir, 0, 4, fps=30.0, method="tracknet")

        # Should be called once per frame (5 times)
        assert mock_tn.call_count == 5

    def test_tracknet_passes_3_frame_buffer(self, tmp_path):
        """TrackNet receives exactly 3 frames per call."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(4):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        call_args = []

        def capture_call(frames, frame_index, **kwargs):
            call_args.append((len(frames), frame_index))
            return None

        with patch("court_vision.ball_tracker.detect_ball_tracknet", side_effect=capture_call):
            build_trajectory(frames_dir, 0, 3, fps=30.0, method="tracknet")

        # Each call should pass exactly 3 frames
        for num_frames, _ in call_args:
            assert num_frames == 3

    def test_default_method_is_tracknet(self, tmp_path):
        """Default method is 'tracknet'."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        with patch("court_vision.ball_tracker.detect_ball_tracknet", return_value=None) as mock_tn:
            build_trajectory(frames_dir, 0, 2, fps=30.0)

        assert mock_tn.call_count == 3

    def test_early_frames_padded_with_black(self, tmp_path):
        """First 2 frames are padded with black frames for TrackNet."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            # Create non-black frames so we can verify padding
            img = np.ones((480, 640, 3), dtype=np.uint8) * 128
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        call_frames = []

        def capture_frames(frames, frame_index, **kwargs):
            # Check if any frame is all-black (padding)
            has_black = any(np.all(f == 0) for f in frames)
            call_frames.append((frame_index, has_black))
            return None

        with patch("court_vision.ball_tracker.detect_ball_tracknet", side_effect=capture_frames):
            build_trajectory(frames_dir, 0, 2, fps=30.0, method="tracknet")

        # Frame 0: needs 2 black padding frames
        assert call_frames[0] == (0, True)
        # Frame 1: needs 1 black padding frame
        assert call_frames[1] == (1, True)
        # Frame 2: has all 3 real frames, no padding
        assert call_frames[2] == (2, False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ball_tracker.py::TestBuildTrajectoryMethod -v`
Expected: FAIL (`build_trajectory() got an unexpected keyword argument 'method'`)

- [ ] **Step 3: Write implementation — update `build_trajectory`**

Replace the `build_trajectory` function in `src/court_vision/ball_tracker.py` (lines 107-142):

```python
def build_trajectory(
    frames_dir: "Path",
    start_frame: int,
    end_frame: int,
    fps: float,
    max_gap_s: float = 0.5,
    method: str = "tracknet",
) -> BallTrajectory:
    """Build a ball trajectory by detecting the ball in each frame.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        start_frame: First frame index to process.
        end_frame: Last frame index to process (inclusive).
        fps: Video frame rate.
        max_gap_s: Maximum gap in seconds to interpolate through.
        method: Detection method — "tracknet" (3-frame sliding window)
                or "hsv" (per-frame color+contour).

    Returns:
        BallTrajectory with detections and interpolated positions.
    """
    from pathlib import Path

    frames_dir = Path(frames_dir)
    raw_detections: list[BallDetection] = []

    if method == "tracknet":
        from court_vision.tracknet import detect_ball_tracknet

        # Read all frames into memory for sliding window
        frames_cache: dict[int, np.ndarray] = {}
        for i in range(start_frame, end_frame + 1):
            frame_path = frames_dir / f"frame_{i:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                frames_cache[i] = frame

        for i in range(start_frame, end_frame + 1):
            if i not in frames_cache:
                continue

            # Build 3-frame buffer: [i-2, i-1, i]
            current = frames_cache[i]
            h, w = current.shape[:2]
            black = np.zeros((h, w, 3), dtype=np.uint8)

            frame_minus2 = frames_cache.get(i - 2, black) if i - 2 >= start_frame else black
            frame_minus1 = frames_cache.get(i - 1, black) if i - 1 >= start_frame else black

            buffer = [frame_minus2, frame_minus1, current]
            det = detect_ball_tracknet(buffer, frame_index=i)
            if det is not None:
                raw_detections.append(det)
    else:
        # HSV method: per-frame detection
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ball_tracker.py -v`
Expected: PASS (all tests including new ones)

- [ ] **Step 5: Commit**

```bash
git add src/court_vision/ball_tracker.py tests/test_ball_tracker.py
git commit -m "feat(court-vision): add method param to build_trajectory with 3-frame sliding window"
```

---

### Task 4: Add `ball_detection_method` to config and thread through pipeline

**Files:**
- Modify: `src/court_vision/config.py:10-16`
- Modify: `src/court_vision/player_detect.py:271-294`
- Modify: `src/court_vision/pipeline.py:98-104`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for config and pipeline threading**

Add to end of `tests/test_pipeline.py`:

```python
class TestPipelineThreadsBallMethod:
    @patch("court_vision.pipeline.build_match_data")
    @patch("court_vision.pipeline.track_segment")
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_ball_detection_method_threaded_to_track_segment(
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
        """ball_detection_method from config is passed to track_segment."""
        import torch

        from court_vision.config import PipelineConfig, PipelineSettings
        from court_vision.ingest import FrameSequence
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        config = PipelineConfig(
            pipeline=PipelineSettings(
                scene_filter_mode="ml",
                ball_detection_method="hsv",
            )
        )
        mock_load_config.return_value = config
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        mock_extract.return_value = FrameSequence(
            frames_dir=frames_dir, fps=30.0, total_frames=100, resolution=(1280, 720),
        )
        mock_filter.return_value = [
            GameplaySegment(
                start_frame=0, end_frame=50,
                start_time_s=0.0, end_time_s=1.67,
                frame_count=51,
            ),
        ]
        mock_homographies.return_value = []
        mock_track.return_value = []
        mock_build_match.return_value = None

        with patch("court_vision.pipeline.classify_frames", return_value=[]):
            run_pipeline(str(video_path), config_path=None)

        # Verify track_segment was called with ball_method="hsv"
        call_kwargs = mock_track.call_args
        assert call_kwargs is not None
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("ball_method") == "hsv"
        else:
            # ball_method is the 5th positional arg
            assert call_kwargs.args[4] == "hsv"


class TestTrackSegmentThreadsBallMethod:
    @patch("court_vision.player_detect.estimate_pose")
    @patch("court_vision.player_detect.detect_players_in_frame")
    @patch("court_vision.player_detect.build_trajectory")
    def test_track_segment_passes_ball_method_to_build_trajectory(
        self,
        mock_build_traj: MagicMock,
        mock_detect_players: MagicMock,
        mock_estimate_pose: MagicMock,
        tmp_path: Path,
    ):
        """track_segment passes ball_method to build_trajectory."""
        import numpy as np

        from court_vision.ball_tracker import BallTrajectory
        from court_vision.player_detect import track_segment
        from court_vision.scene_filter import GameplaySegment

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segment = GameplaySegment(
            start_frame=0, end_frame=2,
            start_time_s=0.0, end_time_s=0.07, frame_count=3,
        )

        mock_build_traj.return_value = BallTrajectory(detections=[], fps=30.0)
        mock_detect_players.return_value = []
        mock_estimate_pose.return_value = None

        track_segment(frames_dir, segment, ball_method="hsv")

        call_kwargs = mock_build_traj.call_args
        assert call_kwargs is not None
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("method") == "hsv"
        else:
            # method would be in kwargs since it's a keyword arg
            assert False, "Expected method='hsv' in build_trajectory kwargs"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestPipelineThreadsBallMethod tests/test_pipeline.py::TestTrackSegmentThreadsBallMethod -v`
Expected: FAIL (unexpected keyword argument)

- [ ] **Step 3: Write implementation — add config field**

In `src/court_vision/config.py`, add `ball_detection_method` to `PipelineSettings` (line 16, after `gameplay_threshold`):

```python
class PipelineSettings(BaseModel):
    target_resolution: list[int] = [1280, 720]
    fps_override: int | None = None
    confidence_threshold: float = 0.7
    max_interpolation_gap_s: float = 0.5
    scene_filter_mode: Literal["heuristic", "ml"] = "heuristic"
    gameplay_threshold: float = 0.45
    ball_detection_method: Literal["tracknet", "hsv"] = "tracknet"
```

- [ ] **Step 4: Write implementation — update `track_segment` in `player_detect.py`**

Update the `track_segment` function signature in `src/court_vision/player_detect.py` (line 271) to accept and thread `ball_method`:

```python
def track_segment(
    frames_dir: Path,
    segment: GameplaySegment,
    homography: np.ndarray | None = None,
    fps: float = 30.0,
    ball_method: str = "tracknet",
) -> list[FrameTrackingResult]:
```

Update the `build_trajectory` call at line 292-294 to pass the method:

```python
    trajectory = build_trajectory(
        frames_dir, segment.start_frame, segment.end_frame, fps=fps,
        method=ball_method,
    )
```

- [ ] **Step 5: Write implementation — update pipeline.py to thread config**

In `src/court_vision/pipeline.py`, update line 103 where `track_segment` is called:

```python
        segment_tracking = track_segment(
            frame_seq.frames_dir, segment, homography,
            fps=frame_seq.fps,
            ball_method=config.pipeline.ball_detection_method,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add src/court_vision/config.py src/court_vision/player_detect.py src/court_vision/pipeline.py tests/test_pipeline.py
git commit -m "feat(court-vision): thread ball_detection_method through config, pipeline, and track_segment"
```

---

### Task 5: Full test suite verification

**Files:**
- No new files — verification only

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests pass (180+ existing + ~20 new = 200+)

- [ ] **Step 2: Verify no import cycles or runtime errors**

Run:
```bash
.venv/bin/python -c "from court_vision.tracknet import TrackNetV2, detect_ball_tracknet, _extract_ball_position; print('TrackNet imports OK')"
.venv/bin/python -c "from court_vision.ball_tracker import build_trajectory; print('ball_tracker imports OK')"
.venv/bin/python -c "from court_vision.config import PipelineConfig; c = PipelineConfig(); print(f'ball_detection_method={c.pipeline.ball_detection_method}')"
.venv/bin/python -c "from court_vision.pipeline import run_pipeline; print('pipeline imports OK')"
```

Expected:
```
TrackNet imports OK
ball_tracker imports OK
ball_detection_method=tracknet
pipeline imports OK
```

## Verification

1. All existing tests still pass (180+)
2. New TrackNet tests pass (~13 new tests)
3. `build_trajectory(method="tracknet")` uses 3-frame sliding window with `detect_ball_tracknet`
4. `build_trajectory(method="hsv")` preserves existing HSV behavior
5. Config `ball_detection_method` threads: `config.py` → `pipeline.py` → `track_segment` → `build_trajectory`
6. Default is `"tracknet"` everywhere

## Out of Scope

- Training infrastructure or custom weight fine-tuning
- E2E test on real video (requires downloaded weights — manual verification)
- Trajectory interpolation upgrades
- Multi-ball detection
- Batch GPU inference optimization
