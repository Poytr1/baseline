"""Court keypoint detection using a pretrained TrackNet-style CNN.

Model from https://github.com/yastrebksv/TennisCourtDetector
Predicts 14 court keypoints from a single broadcast tennis frame.
"""

from functools import lru_cache
from pathlib import Path
import logging
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn

from court_vision.device import get_device

logger = logging.getLogger(__name__)

# Pretrained weights URL (Google Drive direct download)
_WEIGHTS_URL = (
    "https://drive.google.com/uc?export=download"
    "&id=1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"
)
_CACHE_DIR = Path.home() / ".cache" / "court-vision" / "models"
_WEIGHTS_FILENAME = "court_keypoint_net.pt"

# Model input resolution
INPUT_WIDTH = 640
INPUT_HEIGHT = 360

# The 14 predicted keypoints mapped to court coordinates (meters).
# Origin = net center, x = parallel to net (+ right), y = perpendicular (+ far).
# Ordered to match the model's output channels 0-13.
KEYPOINT_COURT_COORDS: list[tuple[float, float]] = [
    (-5.485, 11.885),   # kp0:  far baseline left doubles
    (5.485, 11.885),    # kp1:  far baseline right doubles
    (-5.485, -11.885),  # kp2:  near baseline left doubles
    (5.485, -11.885),   # kp3:  near baseline right doubles
    (-4.115, 11.885),   # kp4:  far baseline left singles
    (-4.115, -11.885),  # kp5:  near baseline left singles
    (4.115, 11.885),    # kp6:  far baseline right singles
    (4.115, -11.885),   # kp7:  near baseline right singles
    (-4.115, 6.4),      # kp8:  far service line left
    (4.115, 6.4),       # kp9:  far service line right
    (-4.115, -6.4),     # kp10: near service line left
    (4.115, -6.4),      # kp11: near service line right
    (0.0, 6.4),         # kp12: far service center
    (0.0, -6.4),        # kp13: near service center
]


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 pad: int = 1, stride: int = 1, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CourtKeypointNet(nn.Module):
    """TrackNet-style encoder-decoder for court keypoint heatmap prediction.

    Input: (B, 3, 360, 640) RGB image tensor in [0, 1].
    Output: (B, 15, 360, 640) heatmap logits (channels 0-13 = keypoints, 14 = center).
    """

    def __init__(self, out_channels: int = 15):
        super().__init__()
        self.out_channels = out_channels

        # Encoder
        self.conv1 = _ConvBlock(3, 64)
        self.conv2 = _ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = _ConvBlock(64, 128)
        self.conv4 = _ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = _ConvBlock(128, 256)
        self.conv6 = _ConvBlock(256, 256)
        self.conv7 = _ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = _ConvBlock(256, 512)
        self.conv9 = _ConvBlock(512, 512)
        self.conv10 = _ConvBlock(512, 512)

        # Decoder
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = _ConvBlock(512, 256)
        self.conv12 = _ConvBlock(256, 256)
        self.conv13 = _ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = _ConvBlock(256, 128)
        self.conv15 = _ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = _ConvBlock(128, 64)
        self.conv17 = _ConvBlock(64, 64)
        self.conv18 = _ConvBlock(64, self.out_channels)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def _download_weights() -> Path:
    """Download pretrained weights if not cached."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = _CACHE_DIR / _WEIGHTS_FILENAME
    if weights_path.exists():
        return weights_path
    logger.info("Downloading court keypoint model weights...")
    try:
        urllib.request.urlretrieve(_WEIGHTS_URL, str(weights_path))
    except Exception:
        logger.warning("Failed to download court keypoint weights.")
        raise
    return weights_path


@lru_cache(maxsize=1)
def _get_model(weights_path: str | None = None) -> tuple[CourtKeypointNet, torch.device]:
    """Load the court keypoint model (singleton)."""
    device = get_device()
    model = CourtKeypointNet(out_channels=15)

    if weights_path is not None:
        path = Path(weights_path)
    else:
        path = _download_weights()

    state_dict = torch.load(str(path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device


def _postprocess_heatmap(
    heatmap: np.ndarray, low_thresh: int = 170, max_radius: int = 25,
) -> tuple[float | None, float | None]:
    """Extract keypoint (x, y) from a single heatmap channel via HoughCircles."""
    _, binary = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=2, minRadius=10, maxRadius=max_radius,
    )
    if circles is not None:
        return float(circles[0][0][0]), float(circles[0][0][1])
    return None, None


def detect_keypoints(
    frame: np.ndarray,
    weights_path: str | None = None,
    confidence_threshold: int = 170,
) -> list[tuple[float | None, float | None]]:
    """Detect 14 court keypoints in a single frame.

    Args:
        frame: BGR image at any resolution.
        weights_path: Optional path to model weights (auto-downloads if None).
        confidence_threshold: Heatmap threshold for peak extraction.

    Returns:
        List of 14 (x, y) tuples in original frame pixel coordinates.
        (None, None) for keypoints not detected.
    """
    model, device = _get_model(weights_path)
    h_orig, w_orig = frame.shape[:2]

    # Preprocess: resize to 640x360, normalize to [0, 1], CHW tensor
    img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    inp = img.astype(np.float32) / 255.0
    inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0)  # (1, 3, 360, 640)

    with torch.no_grad():
        out = model(inp.float().to(device))[0]
    pred = torch.sigmoid(out).detach().cpu().numpy()

    # Extract peaks from channels 0-13 (skip channel 14 = court center aux)
    scale_x = w_orig / INPUT_WIDTH
    scale_y = h_orig / INPUT_HEIGHT
    points: list[tuple[float | None, float | None]] = []

    for kp_idx in range(14):
        heatmap = (pred[kp_idx] * 255).astype(np.uint8)
        x, y = _postprocess_heatmap(heatmap, low_thresh=confidence_threshold)
        if x is not None and y is not None:
            points.append((x * scale_x, y * scale_y))
        else:
            points.append((None, None))

    return points
