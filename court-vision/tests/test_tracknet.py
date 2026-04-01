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
