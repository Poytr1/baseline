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
