"""Tests for TrackNet v2 ball detection module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from court_vision.ball_tracker import BallDetection
from court_vision.tracknet import TrackNetV2, _extract_ball_position, _get_tracknet_model, detect_ball_tracknet


class TestTrackNetV2Architecture:
    def test_model_accepts_9_channel_input(self):
        """TrackNet v2 takes 9-channel input (3 frames x 3 RGB)."""
        model = TrackNetV2()
        model.eval()
        x = torch.randn(1, 9, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 360, 640)

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


class TestGetTracknetModel:
    @patch("court_vision.tracknet._download_tracknet_weights")
    def test_returns_model_instance(self, mock_download):
        """Returns a TrackNetV2 model in eval mode."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        mock_download.return_value = mock_path
        _get_tracknet_model.cache_clear()
        with patch("court_vision.tracknet.torch.load", return_value={}), \
             patch.object(TrackNetV2, "load_state_dict"):
            model = _get_tracknet_model()
        assert isinstance(model, TrackNetV2)
        assert not model.training
        _get_tracknet_model.cache_clear()

    def test_caches_model_on_second_call(self):
        """Second call returns same model instance (singleton)."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        _get_tracknet_model.cache_clear()
        with patch("court_vision.tracknet._download_tracknet_weights", return_value=mock_path), \
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

        fake_heatmap = torch.zeros(1, 3, 360, 640)
        fake_heatmap[0, 2, 180, 320] = 0.9

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

        fake_heatmap = torch.zeros(1, 3, 360, 640)
        fake_heatmap[0, 2, 180, 320] = 0.2

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

        fake_heatmap = torch.zeros(1, 3, 360, 640)
        fake_heatmap[0, 2, 180, 320] = 0.3

        mock_model = MagicMock()
        mock_model.return_value = fake_heatmap

        with patch("court_vision.tracknet._get_tracknet_model", return_value=mock_model), \
             patch("court_vision.tracknet.get_device", return_value=torch.device("cpu")):
            result = detect_ball_tracknet(frames, frame_index=0, confidence_threshold=0.2)

        assert result is not None
        assert result.confidence > 0.2
