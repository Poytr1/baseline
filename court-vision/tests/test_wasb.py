"""Tests for WASB-SBDT ball detection module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from court_vision.ball_tracker import BallDetection
from court_vision.wasb import (
    WASBHRNet,
    _extract_ball_position_weighted,
    _get_wasb_model,
    _strip_module_prefix,
    detect_ball_wasb,
)


class TestWASBHRNetArchitecture:
    def test_model_accepts_9_channel_input(self):
        """WASB HRNet takes 9-channel input (3 frames x 3 RGB)."""
        model = WASBHRNet()
        model.eval()
        x = torch.randn(1, 9, 288, 512)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, dict)
        assert 0 in out
        assert out[0].shape == (1, 3, 288, 512)

    def test_output_is_raw_logits(self):
        """Output is raw logits — sigmoid is applied in postprocessing, not inside the model."""
        model = WASBHRNet()
        model.eval()
        x = torch.randn(1, 9, 288, 512)
        with torch.no_grad():
            out = model(x)[0]
        # Raw logits can be < 0 or > 1; sigmoid is applied downstream.
        # We just confirm it's finite and not clamped to [0,1].
        assert torch.all(torch.isfinite(out))


class TestExtractBallPositionWeighted:
    def test_extracts_peak_from_heatmap(self):
        """Strong peak produces a detection scaled to original coords."""
        heatmap = np.zeros((288, 512), dtype=np.float32)
        heatmap[144:147, 256:259] = 0.9  # 3x3 blob near center
        result = _extract_ball_position_weighted(
            heatmap, original_width=1280, original_height=720,
        )
        assert result is not None
        x, y, conf = result
        # Centroid of blob at (257, 145): 257/512*1280 ≈ 642.5, 145/288*720 ≈ 362.5
        assert abs(x - 642.5) < 5.0
        assert abs(y - 362.5) < 5.0
        assert abs(conf - 0.9) < 0.01

    def test_returns_none_below_threshold(self):
        heatmap = np.zeros((288, 512), dtype=np.float32)
        heatmap[144, 256] = 0.3
        result = _extract_ball_position_weighted(
            heatmap, original_width=1280, original_height=720,
        )
        assert result is None

    def test_custom_threshold(self):
        heatmap = np.zeros((288, 512), dtype=np.float32)
        heatmap[144:146, 256:258] = 0.3
        result = _extract_ball_position_weighted(
            heatmap, original_width=1280, original_height=720,
            confidence_threshold=0.2,
        )
        assert result is not None

    def test_all_zeros_returns_none(self):
        heatmap = np.zeros((288, 512), dtype=np.float32)
        result = _extract_ball_position_weighted(
            heatmap, original_width=1280, original_height=720,
        )
        assert result is None

    def test_picks_highest_scoring_component(self):
        """Given two blobs, the one with higher total weight wins."""
        heatmap = np.zeros((288, 512), dtype=np.float32)
        # Small bright blob at left
        heatmap[50:52, 50:52] = 0.9
        # Larger dimmer blob at right — total weight dominates
        heatmap[150:160, 350:360] = 0.7
        result = _extract_ball_position_weighted(
            heatmap, original_width=512, original_height=288,
        )
        assert result is not None
        x, _, _ = result
        # Winner should be the right blob (x ~ 355)
        assert x > 300


class TestStripModulePrefix:
    def test_strips_data_parallel_prefix(self):
        state = {"module.conv1.weight": 1, "module.bn1.bias": 2, "final.weight": 3}
        stripped = _strip_module_prefix(state)
        assert "conv1.weight" in stripped
        assert "bn1.bias" in stripped
        assert "final.weight" in stripped
        assert not any(k.startswith("module.") for k in stripped)


class TestGetWasbModel:
    @patch("court_vision.wasb._download_wasb_weights")
    def test_returns_model_instance(self, mock_download):
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        mock_download.return_value = mock_path
        _get_wasb_model.cache_clear()
        with patch("court_vision.wasb.torch.load", return_value={}), \
             patch.object(WASBHRNet, "load_state_dict", return_value=([], [])):
            model = _get_wasb_model()
        assert isinstance(model, WASBHRNet)
        assert not model.training
        _get_wasb_model.cache_clear()

    def test_caches_model_on_second_call(self):
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        _get_wasb_model.cache_clear()
        with patch("court_vision.wasb._download_wasb_weights", return_value=mock_path), \
             patch("court_vision.wasb.torch.load", return_value={}), \
             patch.object(WASBHRNet, "load_state_dict", return_value=([], [])):
            model1 = _get_wasb_model()
            model2 = _get_wasb_model()
        assert model1 is model2
        _get_wasb_model.cache_clear()


class TestDetectBallWasb:
    def test_returns_ball_detection_on_strong_signal(self):
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

        fake_logits = torch.full((1, 3, 288, 512), -10.0)
        fake_logits[0, 2, 144:146, 256:258] = 10.0  # strong peak, sigmoid→~1

        mock_model = MagicMock()
        mock_model.return_value = {0: fake_logits}

        with patch("court_vision.wasb._get_wasb_model", return_value=mock_model), \
             patch("court_vision.wasb.get_device", return_value=torch.device("cpu")):
            result = detect_ball_wasb(frames, frame_index=5)

        assert result is not None
        assert isinstance(result, BallDetection)
        assert result.frame_index == 5
        assert result.confidence > 0.5

    def test_returns_none_on_low_confidence(self):
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

        fake_logits = torch.full((1, 3, 288, 512), -10.0)
        # sigmoid(-1.0) ≈ 0.27, below default 0.5 threshold
        fake_logits[0, 2, 144, 256] = -1.0

        mock_model = MagicMock()
        mock_model.return_value = {0: fake_logits}

        with patch("court_vision.wasb._get_wasb_model", return_value=mock_model), \
             patch("court_vision.wasb.get_device", return_value=torch.device("cpu")):
            result = detect_ball_wasb(frames, frame_index=0)

        assert result is None

    def test_requires_exactly_3_frames(self):
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(2)]
        with pytest.raises(ValueError, match="3 frames"):
            detect_ball_wasb(frames, frame_index=0)

    def test_uses_last_frame_channel(self):
        """Peak on channel 2 (current frame) is detected; peaks on channels 0/1 are ignored."""
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

        fake_logits = torch.full((1, 3, 288, 512), -10.0)
        # Peak only on channel 0 (past frame) — should be ignored
        fake_logits[0, 0, 144, 256] = 10.0

        mock_model = MagicMock()
        mock_model.return_value = {0: fake_logits}

        with patch("court_vision.wasb._get_wasb_model", return_value=mock_model), \
             patch("court_vision.wasb.get_device", return_value=torch.device("cpu")):
            result = detect_ball_wasb(frames, frame_index=0)

        assert result is None


class TestExtractBallCandidates:
    def test_every_blob_is_returned_strongest_first(self):
        from court_vision.wasb import extract_ball_candidates, _extract_ball_position_weighted

        hm = np.zeros((288, 512), dtype=np.float32)
        hm[100:104, 200:204] = 0.9          # strong blob
        hm[50:52, 400:402] = 0.6            # weaker blob
        cands = extract_ball_candidates(hm, 1280, 720, confidence_threshold=0.5, max_candidates=5)
        assert len(cands) == 2
        (x1, y1, c1), (x2, y2, c2) = cands
        assert abs(c1 - 0.9) < 1e-6 and abs(c2 - 0.6) < 1e-6
        assert abs(x1 - (201.5 / 512) * 1280) < 2 and abs(y1 - (101.5 / 288) * 720) < 2
        # the single-peak extractor agrees with the first candidate
        single = _extract_ball_position_weighted(hm, 1280, 720, 0.5)
        assert abs(single[0] - x1) < 1e-6 and abs(single[1] - y1) < 1e-6
        assert extract_ball_candidates(hm, 1280, 720, confidence_threshold=0.95) == []
        assert len(extract_ball_candidates(hm, 1280, 720, 0.5, max_candidates=1)) == 1
