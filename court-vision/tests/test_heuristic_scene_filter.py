"""Tests for heuristic scene filter."""

import cv2
import numpy as np
import pytest

from court_vision.heuristic_scene_filter import compute_court_color_ratio


class TestComputeCourtColorRatio:
    def test_green_court_has_high_ratio(self):
        """Frame filled with green court color has high ratio."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)  # BGR forest green
        ratio = compute_court_color_ratio(frame)
        assert ratio > 0.8

    def test_black_image_has_zero_ratio(self):
        """All-black frame has zero court color."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        ratio = compute_court_color_ratio(frame)
        assert ratio == 0.0

    def test_blue_court_has_high_ratio(self):
        """Frame filled with blue hard court color has high ratio."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (180, 120, 40)  # BGR blue court
        ratio = compute_court_color_ratio(frame)
        assert ratio > 0.8

    def test_half_green_has_partial_ratio(self):
        """Half green, half black gives ~0.5 ratio."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:360, :] = (34, 139, 34)  # top half green
        ratio = compute_court_color_ratio(frame)
        assert 0.3 < ratio < 0.7
