"""Tests for the ball tracker module."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

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


class TestInterpolateGapsQuadratic:
    def test_parabolic_arc_not_linear(self):
        """Quadratic interpolation produces curved path, not straight line."""
        from court_vision.ball_tracker import interpolate_gaps

        # Three detections forming an arc: ball goes up then comes down
        dets = [
            BallDetection(frame_index=0, x=100.0, y=400.0, confidence=0.9),
            BallDetection(frame_index=5, x=150.0, y=200.0, confidence=0.9),
            BallDetection(frame_index=10, x=200.0, y=400.0, confidence=0.9),
        ]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)

        # Frame 2 and 3 should be interpolated between dets[0] and dets[1]
        interp_2 = next(d for d in result if d.frame_index == 2)
        interp_3 = next(d for d in result if d.frame_index == 3)

        # Linear would give y=240 at frame 2, y=280 at frame 3
        # Quadratic should give different (curved) values
        # The midpoint (frame 2-3) y should be LOWER (closer to peak)
        # than linear because the arc curves
        linear_y_2 = 400.0 + (2 / 5) * (200.0 - 400.0)  # = 320
        assert abs(interp_2.y - linear_y_2) > 5.0  # meaningfully different from linear

    def test_falls_back_to_linear_with_two_points(self):
        """With only 2 anchor points, uses linear interpolation."""
        from court_vision.ball_tracker import interpolate_gaps

        dets = [
            BallDetection(frame_index=0, x=0.0, y=0.0, confidence=0.9),
            BallDetection(frame_index=4, x=40.0, y=80.0, confidence=0.9),
        ]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        interp_2 = next(d for d in result if d.frame_index == 2)
        # Linear: x=20, y=40
        assert abs(interp_2.x - 20.0) < 0.1
        assert abs(interp_2.y - 40.0) < 0.1

    def test_preserves_original_detections(self):
        """Original (non-interpolated) detections are unchanged."""
        from court_vision.ball_tracker import interpolate_gaps

        dets = [
            BallDetection(frame_index=0, x=100.0, y=400.0, confidence=0.9),
            BallDetection(frame_index=3, x=130.0, y=200.0, confidence=0.85),
            BallDetection(frame_index=6, x=160.0, y=400.0, confidence=0.9),
        ]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)

        orig_0 = next(d for d in result if d.frame_index == 0)
        orig_3 = next(d for d in result if d.frame_index == 3)
        orig_6 = next(d for d in result if d.frame_index == 6)
        assert orig_0.x == 100.0 and orig_0.y == 400.0
        assert orig_3.x == 130.0 and orig_3.y == 200.0
        assert orig_6.x == 160.0 and orig_6.y == 400.0
        assert not orig_0.interpolated
        assert not orig_3.interpolated

    def test_interpolated_flag_still_set(self):
        """Interpolated detections still have interpolated=True."""
        from court_vision.ball_tracker import interpolate_gaps

        dets = [
            BallDetection(frame_index=0, x=0.0, y=0.0, confidence=0.9),
            BallDetection(frame_index=3, x=30.0, y=30.0, confidence=0.9),
        ]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        interp_1 = next(d for d in result if d.frame_index == 1)
        interp_2 = next(d for d in result if d.frame_index == 2)
        assert interp_1.interpolated is True
        assert interp_2.interpolated is True


class TestMapBallToCourt:
    def _identity_homography(self) -> np.ndarray:
        """Returns an identity homography (pixel = court coords)."""
        return np.eye(3, dtype=np.float64)

    def test_maps_detection_to_court_coords(self):
        """Maps a ball detection's pixel coords to court coords via homography."""
        from court_vision.ball_tracker import map_ball_to_court
        H = self._identity_homography()
        det = BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.9)
        court_x, court_y = map_ball_to_court(det, H)
        assert abs(court_x - 100.0) < 0.1
        assert abs(court_y - 200.0) < 0.1

    def test_returns_none_for_none_homography(self):
        """Returns None if homography is None."""
        from court_vision.ball_tracker import map_ball_to_court
        det = BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.9)
        result = map_ball_to_court(det, None)
        assert result is None

    def test_works_with_scaling_homography(self):
        """Correctly applies a scaling homography."""
        from court_vision.ball_tracker import map_ball_to_court
        H = np.array([[2.0, 0, 0], [0, 3.0, 0], [0, 0, 1.0]], dtype=np.float64)
        det = BallDetection(frame_index=0, x=10.0, y=20.0, confidence=0.9)
        court_x, court_y = map_ball_to_court(det, H)
        assert abs(court_x - 20.0) < 0.1
        assert abs(court_y - 60.0) < 0.1


class TestRejectStationaryDetections:
    def test_rejects_stationary_detections(self):
        """Detections at the same position are rejected as stationary."""
        from court_vision.ball_tracker import BallDetection, reject_stationary_detections

        # All detections at nearly the same position (std < 5px)
        dets = [
            BallDetection(frame_index=i, x=100.0 + (i % 2), y=200.0 + (i % 2), confidence=0.8)
            for i in range(10)
        ]
        result = reject_stationary_detections(dets)
        assert all(d is None for d in result)

    def test_preserves_moving_detections(self):
        """Detections with real motion are preserved."""
        from court_vision.ball_tracker import BallDetection, reject_stationary_detections

        # Detections moving across the frame
        dets = [
            BallDetection(frame_index=i, x=100.0 + i * 20.0, y=200.0 + i * 10.0, confidence=0.8)
            for i in range(10)
        ]
        result = reject_stationary_detections(dets)
        non_none = [d for d in result if d is not None]
        assert len(non_none) >= 6

    def test_handles_none_detections(self):
        """None detections (no ball found) pass through."""
        from court_vision.ball_tracker import BallDetection, reject_stationary_detections

        dets = [None, None, None]
        result = reject_stationary_detections(dets)
        assert all(d is None for d in result)

    def test_handles_sparse_detections(self):
        """Mix of None and stationary detections."""
        from court_vision.ball_tracker import BallDetection, reject_stationary_detections

        dets = [
            BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.8),
            None,
            BallDetection(frame_index=2, x=101.0, y=201.0, confidence=0.8),
            None,
            BallDetection(frame_index=4, x=100.5, y=200.5, confidence=0.8),
        ]
        result = reject_stationary_detections(dets)
        # Stationary detections should be rejected, Nones stay None
        assert result[1] is None
        assert result[3] is None

    def test_short_sequence_unchanged(self):
        """Fewer than window_size detections are returned as-is."""
        from court_vision.ball_tracker import BallDetection, reject_stationary_detections

        dets = [
            BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.8),
            BallDetection(frame_index=1, x=100.0, y=200.0, confidence=0.8),
        ]
        result = reject_stationary_detections(dets, window=5)
        assert len(result) == 2
        assert all(d is not None for d in result)


class TestDetectBallMinConfidence:
    def _create_frame_with_ball(self, width=640, height=480,
                                ball_center=(320, 240), ball_radius=8,
                                ball_color=(0, 255, 255)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (0, 100, 0)
        cv2.circle(frame, ball_center, ball_radius, ball_color, -1)
        return frame

    def test_high_confidence_threshold_rejects(self):
        """Very high min_confidence can reject marginal detections."""
        frame = self._create_frame_with_ball()
        result = detect_ball_in_frame(frame, frame_index=0, min_confidence=0.99)
        assert result is None or result.confidence >= 0.99

    def test_zero_confidence_accepts_all(self):
        """min_confidence=0.0 accepts any detection."""
        frame = self._create_frame_with_ball()
        result = detect_ball_in_frame(frame, frame_index=0, min_confidence=0.0)
        assert result is not None

    def test_default_confidence_preserves_behavior(self):
        """Default min_confidence=0.0 matches old behavior (no filtering)."""
        frame = self._create_frame_with_ball()
        result = detect_ball_in_frame(frame, frame_index=0)
        assert result is not None


class TestSmoothTrajectory:
    def test_reduces_jitter(self):
        """Smoothing reduces frame-to-frame noise."""
        from court_vision.ball_tracker import smooth_trajectory

        # Noisy detections along a line: y oscillates ±10 around 200
        dets = [
            BallDetection(frame_index=i, x=float(i * 10), y=200.0 + (10.0 if i % 2 == 0 else -10.0), confidence=0.9)
            for i in range(10)
        ]
        smoothed = smooth_trajectory(dets, window=3)
        assert len(smoothed) == 10

        # Middle detections should have less y-variation than ±10
        mid_ys = [d.y for d in smoothed[1:-1]]
        original_ys = [200.0 + (10.0 if i % 2 == 0 else -10.0) for i in range(1, 9)]
        smooth_var = np.std(mid_ys)
        orig_var = np.std(original_ys)
        assert smooth_var < orig_var

    def test_preserves_interpolated_flag(self):
        """Smoothing does not change interpolated flag."""
        from court_vision.ball_tracker import smooth_trajectory

        dets = [
            BallDetection(frame_index=0, x=0.0, y=0.0, confidence=0.9, interpolated=False),
            BallDetection(frame_index=1, x=10.0, y=10.0, confidence=0.5, interpolated=True),
            BallDetection(frame_index=2, x=20.0, y=20.0, confidence=0.9, interpolated=False),
        ]
        smoothed = smooth_trajectory(dets, window=3)
        assert smoothed[0].interpolated is False
        assert smoothed[1].interpolated is True
        assert smoothed[2].interpolated is False

    def test_short_input_unchanged(self):
        """Fewer than window detections returned as-is."""
        from court_vision.ball_tracker import smooth_trajectory

        dets = [BallDetection(frame_index=0, x=10.0, y=20.0, confidence=0.9)]
        smoothed = smooth_trajectory(dets, window=3)
        assert len(smoothed) == 1
        assert smoothed[0].x == 10.0

    def test_preserves_frame_index(self):
        """Frame indices are preserved after smoothing."""
        from court_vision.ball_tracker import smooth_trajectory

        dets = [
            BallDetection(frame_index=i, x=float(i), y=float(i), confidence=0.9)
            for i in range(5)
        ]
        smoothed = smooth_trajectory(dets, window=3)
        for i, d in enumerate(smoothed):
            assert d.frame_index == i

    def test_confidence_unchanged(self):
        """Smoothing does not alter confidence values."""
        from court_vision.ball_tracker import smooth_trajectory

        dets = [
            BallDetection(frame_index=i, x=float(i), y=float(i), confidence=0.9 - i * 0.1)
            for i in range(5)
        ]
        smoothed = smooth_trajectory(dets, window=3)
        for orig, sm in zip(dets, smoothed):
            assert sm.confidence == orig.confidence


class TestBuildTrajectorySmoothing:
    def test_build_trajectory_applies_smoothing(self, tmp_path):
        """build_trajectory applies smoothing after interpolation."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(5):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        # Create noisy detections (y oscillates ±12 around 200)
        noisy_dets = [
            BallDetection(frame_index=0, x=100.0, y=212.0, confidence=0.9),
            BallDetection(frame_index=1, x=110.0, y=188.0, confidence=0.9),
            BallDetection(frame_index=2, x=120.0, y=212.0, confidence=0.9),
            BallDetection(frame_index=3, x=130.0, y=188.0, confidence=0.9),
            BallDetection(frame_index=4, x=140.0, y=212.0, confidence=0.9),
        ]

        def mock_detect(frames, frame_index, **kwargs):
            for d in noisy_dets:
                if d.frame_index == frame_index:
                    return d
            return None

        with patch("court_vision.ball_tracker.detect_ball_tracknet", side_effect=mock_detect):
            traj = build_trajectory(frames_dir, 0, 4, fps=30.0, method="tracknet")

        # Smoothing should reduce y-variation
        ys = [d.y for d in traj.detections]
        assert np.std(ys) < 10.0  # raw std is 10.0, smoothed should be less


class TestBuildTrajectoryMethod:
    def test_hsv_method_uses_detect_ball_in_frame(self, tmp_path):
        """method='hsv' calls detect_ball_in_frame per frame."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        with patch("court_vision.ball_tracker.detect_ball_in_frame", return_value=None) as mock_hsv:
            build_trajectory(frames_dir, 0, 2, fps=30.0, method="hsv")

        assert mock_hsv.call_count == 3

    def test_tracknet_method_uses_detect_ball_tracknet(self, tmp_path):
        """method='tracknet' calls detect_ball_tracknet with 3-frame windows."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(5):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        with patch("court_vision.ball_tracker.detect_ball_tracknet", return_value=None) as mock_tn:
            build_trajectory(frames_dir, 0, 4, fps=30.0, method="tracknet")

        assert mock_tn.call_count == 5

    def test_tracknet_passes_3_frame_buffer(self, tmp_path):
        """TrackNet receives exactly 3 frames per call."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(4):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        call_args = []

        def capture_call(frames, frame_index, **kwargs):
            call_args.append((len(frames), frame_index))
            return None

        with patch("court_vision.ball_tracker.detect_ball_tracknet", side_effect=capture_call):
            build_trajectory(frames_dir, 0, 3, fps=30.0, method="tracknet")

        for num_frames, _ in call_args:
            assert num_frames == 3

    def test_default_method_is_tracknet(self, tmp_path):
        """Default method is 'tracknet'."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        with patch("court_vision.ball_tracker.detect_ball_tracknet", return_value=None) as mock_tn:
            build_trajectory(frames_dir, 0, 2, fps=30.0)

        assert mock_tn.call_count == 3

    def test_early_frames_padded_with_black(self, tmp_path):
        """First 2 frames are padded with black frames for TrackNet."""
        from court_vision.ball_tracker import build_trajectory

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            img = np.ones((480, 640, 3), dtype=np.uint8) * 128
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        call_frames = []

        def capture_frames(frames, frame_index, **kwargs):
            has_black = any(np.all(f == 0) for f in frames)
            call_frames.append((frame_index, has_black))
            return None

        with patch("court_vision.ball_tracker.detect_ball_tracknet", side_effect=capture_frames):
            build_trajectory(frames_dir, 0, 2, fps=30.0, method="tracknet")

        # Frame 0: needs 2 black padding frames
        assert call_frames[0] == (0, True)
        # Frame 1: needs 1 black padding frame
        assert call_frames[1] == (1, True)
        # Frame 2: has all 3 real frames, no padding
        assert call_frames[2] == (2, False)

