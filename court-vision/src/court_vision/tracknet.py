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
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))

        # Decoder with skip connections
        d3 = self.decoder3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3_conv(d3)

        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2_conv(d2)

        d1 = self.decoder1(d2)
        d1 = self.decoder1_conv(d1)

        out = self.sigmoid(self.final(d1))

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

    x = float(heatmap_x) / heatmap.shape[1] * original_width
    y = float(heatmap_y) / heatmap.shape[0] * original_height

    return (x, y, peak_value)


logger = logging.getLogger(__name__)

_WEIGHTS_URL = "https://github.com/yastrebksv/TrackNet/releases/download/v2.0/tracknet_v2.pt"
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

    if weights_path is not None and weights_path.exists():
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

    heatmap = heatmap_tensor[0, 0].cpu().numpy()

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
