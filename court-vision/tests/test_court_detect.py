"""Tests for court detection and homography."""

import numpy as np
import pytest

from court_vision.court_detect import COURT_KEYPOINTS


class TestCourtKeypoints:
    def test_keypoints_is_dict(self):
        """COURT_KEYPOINTS maps names to (x, y) tuples in meters."""
        assert isinstance(COURT_KEYPOINTS, dict)
        assert len(COURT_KEYPOINTS) >= 12

    def test_net_center_is_origin(self):
        """Net center is at origin (0, 0)."""
        assert COURT_KEYPOINTS["net_center"] == (0.0, 0.0)

    def test_baseline_far_left_singles(self):
        """Far-left singles baseline corner."""
        x, y = COURT_KEYPOINTS["baseline_far_left_singles"]
        assert x == pytest.approx(-4.115)
        assert y == pytest.approx(11.885)

    def test_baseline_near_right_singles(self):
        """Near-right singles baseline corner."""
        x, y = COURT_KEYPOINTS["baseline_near_right_singles"]
        assert x == pytest.approx(4.115)
        assert y == pytest.approx(-11.885)

    def test_service_line_far_center(self):
        """Far service line at center mark."""
        x, y = COURT_KEYPOINTS["service_far_center"]
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(6.4)

    def test_all_keypoints_are_float_tuples(self):
        """Every keypoint is a 2-tuple of floats."""
        for name, (x, y) in COURT_KEYPOINTS.items():
            assert isinstance(x, float), f"{name} x is not float"
            assert isinstance(y, float), f"{name} y is not float"
