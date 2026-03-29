"""Tests for court detection and homography."""

import cv2
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


def _draw_court_lines(img: np.ndarray) -> np.ndarray:
    """Draw white court lines on a green court image for testing."""
    h, w = img.shape[:2]
    cv2.line(img, (200, 650), (1080, 650), (255, 255, 255), 2)
    cv2.line(img, (400, 150), (880, 150), (255, 255, 255), 2)
    cv2.line(img, (200, 650), (400, 150), (255, 255, 255), 2)
    cv2.line(img, (1080, 650), (880, 150), (255, 255, 255), 2)
    cv2.line(img, (280, 450), (1000, 450), (255, 255, 255), 2)
    cv2.line(img, (360, 280), (920, 280), (255, 255, 255), 2)
    cv2.line(img, (640, 280), (640, 450), (255, 255, 255), 2)
    return img


class TestDetectCourtLines:
    def test_detects_lines_on_synthetic_court(self):
        """Detects lines from a synthetic court image."""
        from court_vision.court_detect import detect_court_lines

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)  # green
        img = _draw_court_lines(img)
        lines = detect_court_lines(img)
        assert len(lines) >= 4

    def test_returns_list_of_line_segments(self):
        """Each detected line is a pair of (x, y) endpoints."""
        from court_vision.court_detect import detect_court_lines

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        img = _draw_court_lines(img)
        lines = detect_court_lines(img)
        for line in lines:
            assert len(line) == 2, "Each line should be ((x1,y1), (x2,y2))"
            (x1, y1), (x2, y2) = line
            assert isinstance(x1, (int, float))
            assert isinstance(y1, (int, float))

    def test_no_lines_on_blank_image(self):
        """Returns empty list when no court lines are present."""
        from court_vision.court_detect import detect_court_lines

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        lines = detect_court_lines(img)
        assert lines == []
