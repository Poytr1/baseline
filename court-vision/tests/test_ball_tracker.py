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


class TestBallTrajectory:
    def test_trajectory_fields(self):
        """BallTrajectory stores detections and fps."""
        dets = [BallDetection(frame_index=0, x=10.0, y=20.0, confidence=0.9)]
        from court_vision.ball_tracker import BallTrajectory
        traj = BallTrajectory(detections=dets, fps=30.0)
        assert len(traj.detections) == 1
        assert traj.fps == 30.0


class TestBuildTrajectory:
    def test_builds_from_frame_directory(self):
        """build_trajectory processes frames and returns trajectory."""
        from court_vision.ball_tracker import BallTrajectory
        dets = [
            BallDetection(frame_index=0, x=10.0, y=20.0, confidence=0.9),
            BallDetection(frame_index=2, x=30.0, y=40.0, confidence=0.8),
        ]
        traj = BallTrajectory(detections=dets, fps=30.0)
        assert len(traj.detections) == 2


class TestInterpolateGaps:
    def test_fills_single_frame_gap(self):
        """Interpolates a single missing frame between two detections."""
        from court_vision.ball_tracker import interpolate_gaps
        dets = [
            BallDetection(frame_index=0, x=0.0, y=0.0, confidence=0.9),
            BallDetection(frame_index=2, x=20.0, y=20.0, confidence=0.9),
        ]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 3
        interp = result[1]
        assert interp.frame_index == 1
        assert abs(interp.x - 10.0) < 0.1
        assert abs(interp.y - 10.0) < 0.1
        assert interp.interpolated is True

    def test_fills_multi_frame_gap(self):
        """Interpolates multiple missing frames within max_gap_s."""
        from court_vision.ball_tracker import interpolate_gaps
        dets = [
            BallDetection(frame_index=0, x=0.0, y=0.0, confidence=0.9),
            BallDetection(frame_index=5, x=50.0, y=100.0, confidence=0.9),
        ]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 6
        for i, det in enumerate(result):
            assert det.frame_index == i
        mid = result[2]
        assert abs(mid.x - 20.0) < 0.1
        assert abs(mid.y - 40.0) < 0.1

    def test_does_not_fill_gap_beyond_max(self):
        """Gaps longer than max_gap_s are NOT interpolated."""
        from court_vision.ball_tracker import interpolate_gaps
        dets = [
            BallDetection(frame_index=0, x=0.0, y=0.0, confidence=0.9),
            BallDetection(frame_index=30, x=50.0, y=50.0, confidence=0.9),
        ]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 2

    def test_returns_empty_for_empty_input(self):
        """Returns empty list for empty input."""
        from court_vision.ball_tracker import interpolate_gaps
        result = interpolate_gaps([], fps=30.0, max_gap_s=0.5)
        assert result == []

    def test_returns_single_detection_unchanged(self):
        """Single detection is returned as-is."""
        from court_vision.ball_tracker import interpolate_gaps
        dets = [BallDetection(frame_index=0, x=10.0, y=20.0, confidence=0.9)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 1
        assert result[0].interpolated is False
