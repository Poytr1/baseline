"""Tests for court keypoint neural network module."""

import cv2
import numpy as np
import pytest


class TestKeypointCourtCoords:
    def test_has_14_entries(self):
        from court_vision.court_keypoint_net import KEYPOINT_COURT_COORDS
        assert len(KEYPOINT_COURT_COORDS) == 14

    def test_coords_are_float_tuples(self):
        from court_vision.court_keypoint_net import KEYPOINT_COURT_COORDS
        for x, y in KEYPOINT_COURT_COORDS:
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_symmetric_pairs(self):
        """Left/right keypoints should be symmetric about x=0."""
        from court_vision.court_keypoint_net import KEYPOINT_COURT_COORDS
        # kp0 and kp1: far baseline doubles
        assert KEYPOINT_COURT_COORDS[0][0] == -KEYPOINT_COURT_COORDS[1][0]
        assert KEYPOINT_COURT_COORDS[0][1] == KEYPOINT_COURT_COORDS[1][1]
        # kp2 and kp3: near baseline doubles
        assert KEYPOINT_COURT_COORDS[2][0] == -KEYPOINT_COURT_COORDS[3][0]
        assert KEYPOINT_COURT_COORDS[2][1] == KEYPOINT_COURT_COORDS[3][1]


class TestCourtKeypointNet:
    def test_model_output_shape(self):
        import torch
        from court_vision.court_keypoint_net import CourtKeypointNet
        model = CourtKeypointNet(out_channels=15)
        model.eval()
        inp = torch.rand(1, 3, 360, 640)
        with torch.no_grad():
            out = model(inp)
        assert out.shape == (1, 15, 360, 640)

    def test_model_output_channels_default(self):
        from court_vision.court_keypoint_net import CourtKeypointNet
        model = CourtKeypointNet()
        assert model.out_channels == 15


class TestPostprocessHeatmap:
    def test_returns_none_for_blank(self):
        from court_vision.court_keypoint_net import _postprocess_heatmap
        blank = np.zeros((360, 640), dtype=np.uint8)
        x, y = _postprocess_heatmap(blank)
        assert x is None
        assert y is None

    def test_finds_bright_spot(self):
        from court_vision.court_keypoint_net import _postprocess_heatmap
        heatmap = np.zeros((360, 640), dtype=np.uint8)
        cv2.circle(heatmap, (320, 180), 12, 255, -1)
        x, y = _postprocess_heatmap(heatmap)
        assert x is not None and y is not None
        assert abs(x - 320) < 5
        assert abs(y - 180) < 5


class TestDetectKeypoints:
    def test_returns_14_points(self):
        """With random weights, should still return 14 entries."""
        from unittest.mock import patch
        from court_vision.court_keypoint_net import detect_keypoints, CourtKeypointNet
        import torch

        # Use a fresh model with random weights (no download)
        model = CourtKeypointNet(out_channels=15)
        model.eval()
        device = torch.device("cpu")

        with patch("court_vision.court_keypoint_net._get_model", return_value=(model, device)):
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            points = detect_keypoints(frame)
            assert len(points) == 14
            for pt in points:
                assert len(pt) == 2
