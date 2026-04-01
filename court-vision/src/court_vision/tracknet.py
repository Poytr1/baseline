"""TrackNet v2 ball detection — model architecture, weight loading, and inference."""

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from court_vision.ball_tracker import BallDetection
from court_vision.device import get_device


# TrackNet v2 input/output resolution
TRACKNET_WIDTH = 640
TRACKNET_HEIGHT = 360


class _Conv(nn.Module):
    """Single conv layer: Conv2d -> ReLU -> BatchNorm."""

    def __init__(self, ic: int, oc: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, kernel_size=(3, 3), padding="same")
        self.bn = nn.BatchNorm2d(oc)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.act(self.conv(x)))


class TrackNetV2(nn.Module):
    """TrackNet v2 encoder-decoder for tennis ball detection.

    Architecture matches ChgygLin/TrackNetV2-pytorch for pretrained weight
    compatibility.  Input: 9-channel tensor (3 consecutive RGB frames
    concatenated).  Output: 3-channel heatmap (one per input frame).
    """

    def __init__(self) -> None:
        super().__init__()

        # --- Encoder (VGG16-style) ---

        # Block 1: 9 -> 64
        self.conv2d_1 = _Conv(9, 64)
        self.conv2d_2 = _Conv(64, 64)
        self.max_pooling_1 = nn.MaxPool2d((2, 2), stride=(2, 2))

        # Block 2: 64 -> 128
        self.conv2d_3 = _Conv(64, 128)
        self.conv2d_4 = _Conv(128, 128)
        self.max_pooling_2 = nn.MaxPool2d((2, 2), stride=(2, 2))

        # Block 3: 128 -> 256
        self.conv2d_5 = _Conv(128, 256)
        self.conv2d_6 = _Conv(256, 256)
        self.conv2d_7 = _Conv(256, 256)
        self.max_pooling_3 = nn.MaxPool2d((2, 2), stride=(2, 2))

        # Block 4 (bottleneck): 256 -> 512
        self.conv2d_8 = _Conv(256, 512)
        self.conv2d_9 = _Conv(512, 512)
        self.conv2d_10 = _Conv(512, 512)

        # --- Decoder (U-Net style with skip connections) ---

        # Upsample + skip from conv2d_7 (512+256=768)
        self.up_sampling_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_11 = _Conv(768, 256)
        self.conv2d_12 = _Conv(256, 256)
        self.conv2d_13 = _Conv(256, 256)

        # Upsample + skip from conv2d_4 (256+128=384)
        self.up_sampling_2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_14 = _Conv(384, 128)
        self.conv2d_15 = _Conv(128, 128)

        # Upsample + skip from conv2d_2 (128+64=192)
        self.up_sampling_3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_16 = _Conv(192, 64)
        self.conv2d_17 = _Conv(64, 64)

        # Final 1x1 conv: 64 -> 3 output channels (one heatmap per frame)
        self.conv2d_18 = nn.Conv2d(64, 3, kernel_size=(1, 1), padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder ---
        x = self.conv2d_1(x)
        x1 = self.conv2d_2(x)  # skip connection 1 (64 ch)
        x = self.max_pooling_1(x1)

        x = self.conv2d_3(x)
        x2 = self.conv2d_4(x)  # skip connection 2 (128 ch)
        x = self.max_pooling_2(x2)

        x = self.conv2d_5(x)
        x = self.conv2d_6(x)
        x3 = self.conv2d_7(x)  # skip connection 3 (256 ch)
        x = self.max_pooling_3(x3)

        x = self.conv2d_8(x)
        x = self.conv2d_9(x)
        x = self.conv2d_10(x)  # bottleneck (512 ch)

        # --- Decoder ---
        x = self.up_sampling_1(x)
        x = torch.cat([x, x3], dim=1)  # 512+256=768
        x = self.conv2d_11(x)
        x = self.conv2d_12(x)
        x = self.conv2d_13(x)

        x = self.up_sampling_2(x)
        x = torch.cat([x, x2], dim=1)  # 256+128=384
        x = self.conv2d_14(x)
        x = self.conv2d_15(x)

        x = self.up_sampling_3(x)
        x = torch.cat([x, x1], dim=1)  # 128+64=192
        x = self.conv2d_16(x)
        x = self.conv2d_17(x)
        x = self.conv2d_18(x)

        return torch.sigmoid(x)


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

    x = float(heatmap_x) / heatmap.shape[1] * original_width
    y = float(heatmap_y) / heatmap.shape[0] * original_height

    return (x, y, peak_value)


logger = logging.getLogger(__name__)

_WEIGHTS_URL = "https://github.com/ChgygLin/TrackNetV2-pytorch/releases/download/v0.1/last.pt"
_CACHE_DIR = Path.home() / ".cache" / "court-vision" / "models"
_WEIGHTS_FILENAME = "tracknet_v2.pt"


def _download_tracknet_weights() -> Path:
    """Download pretrained TrackNet v2 weights if not cached."""
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
    """Load the TrackNet v2 model (cached singleton)."""
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
        frame_index: Frame index for the detection result.
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

    # Concatenate channel dim: (360, 640, 9) -> (9, 360, 640)
    concatenated = np.concatenate(processed, axis=2)
    tensor = torch.from_numpy(concatenated).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap_tensor = model(tensor)

    # Model outputs 3 channels (one per frame); use channel 2 (last/current frame)
    heatmap = heatmap_tensor[0, 2].cpu().numpy()

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
