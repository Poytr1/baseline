"""Tests for the ball tracker module."""

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from court_vision.ball_tracker import BallDetection, detect_ball_in_frame


class TestBallDetection:
    def test_ball_detection_fields(self):
        """BallDetection has required fields."""
        det = BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.9)
        assert det.frame_index == 0
        assert det.x == 100.0
        assert det.y == 200.0
        assert det.confidence == 0.9
        assert det.interpolated is False

    def test_ball_detection_interpolated_flag(self):
        """BallDetection supports interpolated flag."""
        det = BallDetection(frame_index=5, x=50.0, y=60.0, confidence=0.5, interpolated=True)
        assert det.interpolated is True


class TestDetectBallInFrame:
    def _create_frame_with_ball(self, width: int = 640, height: int = 480,
                                  ball_center: tuple[int, int] = (320, 240),
                                  ball_radius: int = 8,
                                  ball_color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """Create a synthetic frame with a bright circular ball."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw green court background
        frame[:, :] = (0, 100, 0)
        cv2.circle(frame, ball_center, ball_radius, ball_color, -1)
        return frame

    def test_detects_ball_in_frame(self):
        """Detects a bright ball on a dark/green background."""
        frame = self._create_frame_with_ball(ball_center=(200, 150), ball_radius=8)
        result = detect_ball_in_frame(frame, frame_index=0)
        assert result is not None
        assert abs(result.x - 200) < 20
        assert abs(result.y - 150) < 20
        assert result.confidence > 0.0

    def test_returns_none_for_empty_frame(self):
        """Returns None when no ball-like object is found."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (0, 100, 0)  # uniform green
        result = detect_ball_in_frame(frame, frame_index=0)
        assert result is None

    def test_frame_index_propagated(self):
        """frame_index is passed through to the result."""
        frame = self._create_frame_with_ball()
        result = detect_ball_in_frame(frame, frame_index=42)
        if result is not None:
            assert result.frame_index == 42
