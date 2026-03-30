"""Tests for heuristic scene filter."""

import cv2
import numpy as np
import pytest

from court_vision.heuristic_scene_filter import compute_court_color_ratio, compute_line_score


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


class TestComputeLineScore:
    def test_frame_with_white_lines_scores_high(self):
        """Frame with white lines on dark background scores > 0."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (34, 100, 34)  # dark green background
        # Draw horizontal white lines
        cv2.line(frame, (100, 200), (1100, 200), (255, 255, 255), 3)
        cv2.line(frame, (100, 500), (1100, 500), (255, 255, 255), 3)
        # Draw vertical white lines
        cv2.line(frame, (200, 100), (200, 600), (255, 255, 255), 3)
        cv2.line(frame, (1000, 100), (1000, 600), (255, 255, 255), 3)
        score = compute_line_score(frame)
        assert score > 0.3

    def test_blank_frame_has_zero_score(self):
        """All-black frame has no lines."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        score = compute_line_score(frame)
        assert score == 0.0

    def test_grid_pattern_gets_bonus(self):
        """Lines in both H and V directions get grid bonus."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(frame, (100, 300), (1100, 300), (255, 255, 255), 3)
        cv2.line(frame, (640, 50), (640, 670), (255, 255, 255), 3)
        score = compute_line_score(frame)
        # Should have grid bonus since lines in both directions
        assert score > 0.0
