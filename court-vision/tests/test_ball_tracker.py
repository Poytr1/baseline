"""Tests for the ball tracker module (neural detection driver + trajectory post-processing).

The neural detectors themselves are never run: every test that drives
``detect_ball_sequence`` / ``build_trajectory`` swaps the detector for a fake.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from court_vision import ball_tracker
from court_vision.ball_tracker import (
    BallDetection,
    BallTrajectory,
    build_trajectory,
    detect_ball_sequence,
    far_ball_roi,
    interpolate_gaps,
    map_ball_to_court,
    postprocess_trajectory,
    reject_stationary_detections,
    reject_velocity_outliers,
    smooth_trajectory,
    trim_weak_edges,
)

# Linear pixel->court homography: court_x = (px - 640) / 50, court_y = -(py - 360) / 25
# (larger image y == closer to the camera == negative court y).
H_LINEAR = np.array([[1 / 50, 0.0, -12.8], [0.0, -1 / 25, 14.4], [0.0, 0.0, 1.0]])
FRAME_SHAPE = (720, 1280, 3)

# Frame i is filled with this gray level so a fake detector can tell frames apart.
_GRAY = [20, 60, 100, 140, 180, 220]


def _write_frames(frames_dir: Path, n: int, shape=(48, 64, 3)) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full(shape, _GRAY[i], dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)


def _which(frame: np.ndarray) -> int:
    """Index of the written frame whose gray level is closest to ``frame``'s mean."""
    return int(np.argmin([abs(float(frame.mean()) - g) for g in _GRAY]))


@contextmanager
def _patched_detector(method: str, fn):
    """Replace the detector for ``method`` with ``fn`` for the duration of the block.

    ``detect_ball_sequence`` resolves the wrapper by name at call time, so
    patching the public wrapper on the module is enough to keep the real
    model out of the tests.
    """
    mock = MagicMock(side_effect=fn)
    attr = "detect_ball_wasb" if method == "wasb" else "detect_ball_tracknet"
    with patch.object(ball_tracker, attr, mock):
        yield mock


def _det(frame: int, x: float, y: float, conf: float = 0.9, interpolated: bool = False) -> BallDetection:
    return BallDetection(frame_index=frame, x=x, y=y, confidence=conf, interpolated=interpolated)


def _line(frames, x0: float = 100.0, dx: float = 10.0, y: float = 200.0, conf: float = 0.9) -> list[BallDetection]:
    """Detections along a horizontal line, ``dx`` px per frame, at the given frame indices."""
    return [_det(f, x0 + dx * f, y, conf) for f in frames]


class TestBallDetection:
    def test_ball_detection_fields(self):
        det = BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.9)
        assert det.frame_index == 0
        assert det.x == 100.0
        assert det.y == 200.0
        assert det.confidence == 0.9
        assert det.interpolated is False

    def test_ball_detection_interpolated_flag(self):
        det = BallDetection(frame_index=5, x=50.0, y=60.0, confidence=0.5, interpolated=True)
        assert det.interpolated is True


class TestBallTrajectory:
    def test_trajectory_fields(self):
        dets = [_det(0, 10.0, 20.0)]
        traj = BallTrajectory(detections=dets, fps=30.0)
        assert len(traj.detections) == 1
        assert traj.fps == 30.0


class TestDetectBallSequence:
    def test_calls_detector_once_per_frame_with_3_frame_buffers(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 5)
        calls = []

        def capture(frames, frame_index, **kwargs):
            calls.append((frame_index, [len(frames)]))
            return None

        with _patched_detector("tracknet", capture):
            out = detect_ball_sequence(frames_dir, 0, 4, method="tracknet")

        assert out == []
        assert [c[0] for c in calls] == [0, 1, 2, 3, 4]
        assert all(c[1] == [3] for c in calls)

    def test_default_method_is_wasb(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 3)
        with _patched_detector("wasb", lambda frames, frame_index, **kw: None) as wasb, \
                _patched_detector("tracknet", lambda frames, frame_index, **kw: None) as tracknet:
            detect_ball_sequence(frames_dir, 0, 2)
        assert wasb.call_count == 3
        assert tracknet.call_count == 0

    def test_unknown_method_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="hsv"):
            detect_ball_sequence(tmp_path, 0, 2, method="hsv")

    def test_early_frames_repeat_the_earliest_frame(self, tmp_path: Path):
        """Before the window is full the earliest frame is repeated (no black padding)."""
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 3)
        buffers = {}

        def capture(frames, frame_index, **kwargs):
            buffers[frame_index] = [_which(f) for f in frames]
            assert not any(np.all(f == 0) for f in frames)
            return None

        with _patched_detector("tracknet", capture):
            detect_ball_sequence(frames_dir, 0, 2, method="tracknet")

        assert buffers == {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 2]}

    def test_frame_step_spaces_the_window(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 6)
        buffers = {}

        def capture(frames, frame_index, **kwargs):
            buffers[frame_index] = [_which(f) for f in frames]
            return None

        with _patched_detector("wasb", capture):
            detect_ball_sequence(frames_dir, 0, 5, method="wasb", frame_step=2)

        assert buffers[5] == [1, 3, 5]
        assert buffers[4] == [0, 2, 4]
        assert buffers[1] == [0, 0, 1]  # clamped to the first frame of the range

    def test_start_frame_bounds_the_window(self, tmp_path: Path):
        """The window never reaches before ``start_frame`` even if earlier frames exist."""
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 5)
        buffers = {}

        def capture(frames, frame_index, **kwargs):
            buffers[frame_index] = [_which(f) for f in frames]
            return None

        with _patched_detector("wasb", capture):
            detect_ball_sequence(frames_dir, 2, 4, method="wasb")

        assert buffers == {2: [2, 2, 2], 3: [2, 2, 3], 4: [2, 3, 4]}

    def test_confidence_threshold_and_frame_index_forwarded(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 2)
        with _patched_detector("wasb", lambda frames, frame_index, **kw: None) as mock:
            detect_ball_sequence(frames_dir, 0, 1, method="wasb", confidence_threshold=0.42)
        for call, expected_index in zip(mock.call_args_list, [0, 1]):
            assert call.kwargs["frame_index"] == expected_index
            assert call.kwargs["confidence_threshold"] == 0.42

    def test_collects_raw_detections_without_interpolation(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 5)
        found = {0: _det(0, 10.0, 10.0), 2: _det(2, 30.0, 30.0), 4: _det(4, 50.0, 50.0)}

        with _patched_detector("wasb", lambda frames, frame_index, **kw: found.get(frame_index)):
            out = detect_ball_sequence(frames_dir, 0, 4, method="wasb")

        assert [d.frame_index for d in out] == [0, 2, 4]
        assert all(not d.interpolated for d in out)

    def test_progress_callback_reports_current_and_total(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 4)
        progress = []
        with _patched_detector("wasb", lambda frames, frame_index, **kw: None):
            detect_ball_sequence(frames_dir, 0, 3, method="wasb", progress_callback=lambda c, t: progress.append((c, t)))
        assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]

    def test_missing_frame_files_are_skipped(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 4)
        (frames_dir / "frame_000002.jpg").unlink()
        seen = []

        def capture(frames, frame_index, **kwargs):
            seen.append(frame_index)
            return _det(frame_index, 1.0, 1.0)

        with _patched_detector("wasb", capture):
            out = detect_ball_sequence(frames_dir, 0, 3, method="wasb")

        assert seen == [0, 1, 3]
        assert [d.frame_index for d in out] == [0, 1, 3]

    def test_far_roi_runs_a_second_pass_on_the_crop(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 2)
        roi = (10, 4, 40, 24)  # (x1, y1, x2, y2) inside the 64x48 frames
        shapes = []

        def capture(frames, frame_index, **kwargs):
            shapes.append(frames[-1].shape[:2])
            if frames[-1].shape[:2] == (20, 30):  # the crop
                return _det(frame_index, 5.0, 6.0, conf=0.9)
            return _det(frame_index, 50.0, 40.0, conf=0.4)

        with _patched_detector("wasb", capture) as mock:
            out = detect_ball_sequence(frames_dir, 0, 1, method="wasb", far_roi=roi)

        assert mock.call_count == 4  # full frame + crop, per frame
        assert shapes.count((48, 64)) == 2 and shapes.count((20, 30)) == 2
        # The crop detection is more confident and is mapped back to full-frame pixels.
        assert [(d.x, d.y, d.confidence) for d in out] == [(15.0, 10.0, 0.9), (15.0, 10.0, 0.9)]

    def test_far_roi_full_frame_detection_wins_when_more_confident(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 1)

        def capture(frames, frame_index, **kwargs):
            if frames[-1].shape[:2] == (20, 30):
                return _det(frame_index, 5.0, 6.0, conf=0.3)
            return _det(frame_index, 50.0, 40.0, conf=0.8)

        with _patched_detector("wasb", capture):
            out = detect_ball_sequence(frames_dir, 0, 0, method="wasb", far_roi=(10, 4, 40, 24))
        assert [(d.x, d.y, d.confidence) for d in out] == [(50.0, 40.0, 0.8)]

    def test_far_roi_crop_used_when_full_frame_misses(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 1)

        def capture(frames, frame_index, **kwargs):
            if frames[-1].shape[:2] == (20, 30):
                return _det(frame_index, 1.0, 2.0, conf=0.5)
            return None

        with _patched_detector("wasb", capture):
            out = detect_ball_sequence(frames_dir, 0, 0, method="wasb", far_roi=(10, 4, 40, 24))
        assert [(d.x, d.y) for d in out] == [(11.0, 6.0)]


class TestFarBallRoi:
    def test_none_without_homography(self):
        assert far_ball_roi(None, FRAME_SHAPE) is None

    def test_roi_covers_far_half_and_stays_inside_the_frame(self):
        roi = far_ball_roi(H_LINEAR, FRAME_SHAPE)
        assert roi is not None
        x1, y1, x2, y2 = roi
        assert 0 <= x1 < x2 <= 1280
        assert 0 <= y1 < y2 <= 720
        # far baseline corners (px 366..914, py ~63) and the net line (py 360) are inside
        assert x1 <= 365 and x2 >= 915
        assert y1 <= 62 and y2 > 360
        # ...but the near half of the court is not
        assert y2 < 600
        # horizontally centred on the court
        assert abs((x1 + x2) / 2 - 640) < 5

    def test_returns_ints(self):
        roi = far_ball_roi(H_LINEAR, FRAME_SHAPE)
        assert all(isinstance(v, int) for v in roi)

    def test_singular_homography_gives_none(self):
        assert far_ball_roi(np.zeros((3, 3)), FRAME_SHAPE) is None

    def test_tiny_frame_gives_none(self):
        assert far_ball_roi(H_LINEAR, (20, 30, 3)) is None


class TestRejectVelocityOutliers:
    def test_straight_track_unchanged(self):
        dets = _line(range(10))
        assert reject_velocity_outliers(dets, fps=30.0, max_speed_px_per_frame=150.0, min_run_s=0.0) == dets

    def test_fewer_than_two_detections_returned_as_is(self):
        assert reject_velocity_outliers([], fps=30.0) == []
        one = [_det(0, 1.0, 1.0)]
        assert reject_velocity_outliers(one, fps=30.0) == one

    def test_isolated_blip_inside_a_track_is_dropped(self):
        dets = _line(range(10))
        dets[5] = _det(5, 900.0, 900.0)  # false peak far off the track
        result = reject_velocity_outliers(dets, fps=30.0, max_speed_px_per_frame=150.0, min_run_s=0.0)
        assert [d.frame_index for d in result] == [0, 1, 2, 3, 4, 6, 7, 8, 9]

    def test_two_frame_blip_is_dropped(self):
        dets = _line(range(12))
        dets[5] = _det(5, 900.0, 900.0)
        dets[6] = _det(6, 905.0, 905.0)  # consistent with the first blip, not with the track
        result = reject_velocity_outliers(dets, fps=30.0, max_speed_px_per_frame=150.0, min_run_s=0.0)
        assert [d.frame_index for d in result] == [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]

    def test_isolated_short_run_at_the_start_is_dropped(self):
        dets = [_det(0, 900.0, 900.0)] + _line(range(1, 8))
        result = reject_velocity_outliers(dets, fps=30.0, max_speed_px_per_frame=150.0, min_run_s=0.0)
        assert [d.frame_index for d in result] == [1, 2, 3, 4, 5, 6, 7]

    def test_isolated_short_run_at_the_end_is_dropped(self):
        dets = _line(range(7)) + [_det(7, 900.0, 900.0), _det(8, 902.0, 902.0)]
        result = reject_velocity_outliers(dets, fps=30.0, max_speed_px_per_frame=150.0, min_run_s=0.0)
        assert [d.frame_index for d in result] == [0, 1, 2, 3, 4, 5, 6]

    def test_speed_cap_scales_with_the_frame_gap(self):
        # 600 px over 5 frames == 120 px/frame, under a 150 px/frame cap
        dets = [_det(0, 0.0, 0.0), _det(5, 600.0, 0.0), _det(10, 1200.0, 0.0)]
        assert reject_velocity_outliers(dets, fps=30.0, max_speed_px_per_frame=150.0, min_run_s=0.0) == dets
        # the same displacement in one frame is an outlier chain -> everything goes
        dets = [_det(0, 0.0, 0.0), _det(1, 600.0, 0.0), _det(2, 1200.0, 0.0)]
        assert reject_velocity_outliers(dets, fps=30.0, max_speed_px_per_frame=150.0, min_run_s=0.0) == []

    def test_min_run_one_keeps_every_run(self):
        dets = _line(range(10))
        dets[5] = _det(5, 900.0, 900.0)
        result = reject_velocity_outliers(dets, fps=30.0, max_speed_px_per_frame=150.0, min_run=1, min_run_s=0.0)
        assert result == dets

    def test_generous_cap_keeps_the_blip(self):
        dets = _line(range(10))
        dets[5] = _det(5, 900.0, 900.0)
        assert reject_velocity_outliers(dets, fps=30.0, max_speed_px_per_frame=5000.0, min_run_s=0.0) == dets

    def test_output_preserves_order_and_objects(self):
        dets = _line(range(5))
        result = reject_velocity_outliers(dets, fps=30.0, min_run_s=0.0)
        assert [d is o for d, o in zip(result, dets)] == [True] * 5


class TestInterpolateGaps:
    def test_fills_single_frame_gap(self):
        dets = [_det(0, 0.0, 0.0), _det(2, 20.0, 20.0)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 3
        interp = result[1]
        assert interp.frame_index == 1
        assert abs(interp.x - 10.0) < 0.1
        assert abs(interp.y - 10.0) < 0.1
        assert interp.interpolated is True

    def test_fills_multi_frame_gap(self):
        dets = [_det(0, 0.0, 0.0), _det(5, 50.0, 100.0)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 6
        for i, det in enumerate(result):
            assert det.frame_index == i
        mid = result[2]
        assert abs(mid.x - 20.0) < 0.1
        assert abs(mid.y - 40.0) < 0.1

    def test_interpolated_confidence_is_half_the_weaker_anchor(self):
        dets = [_det(0, 0.0, 0.0, conf=0.8), _det(2, 20.0, 20.0, conf=0.6)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert result[1].confidence == pytest.approx(0.3)

    def test_does_not_fill_gap_beyond_max(self):
        dets = [_det(0, 0.0, 0.0), _det(30, 50.0, 50.0)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 2

    def test_max_gap_scales_with_fps(self):
        dets = [_det(0, 0.0, 0.0), _det(20, 20.0, 20.0)]
        assert len(interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)) == 2  # 15-frame limit
        assert len(interpolate_gaps(dets, fps=60.0, max_gap_s=0.5)) == 21  # 30-frame limit

    def test_returns_empty_for_empty_input(self):
        assert interpolate_gaps([], fps=30.0, max_gap_s=0.5) == []

    def test_returns_single_detection_unchanged(self):
        dets = [_det(0, 10.0, 20.0)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        assert len(result) == 1
        assert result[0].interpolated is False

    def test_speed_cap_skips_gaps_between_different_tracks(self):
        # 900 px over 3 frames == 300 px/frame: not the same ball
        dets = [_det(0, 0.0, 0.0), _det(3, 900.0, 0.0)]
        assert len(interpolate_gaps(dets, fps=30.0, max_gap_s=0.5, max_speed_px_per_frame=150.0)) == 2
        assert len(interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)) == 4  # no cap
        assert len(interpolate_gaps(dets, fps=30.0, max_gap_s=0.5, max_speed_px_per_frame=400.0)) == 4

    def test_speed_cap_uses_the_gap_length(self):
        # 300 px over 3 frames == 100 px/frame, within a 150 px/frame cap
        dets = [_det(0, 0.0, 0.0), _det(3, 300.0, 0.0)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5, max_speed_px_per_frame=150.0)
        assert [d.frame_index for d in result] == [0, 1, 2, 3]
        assert [d.x for d in result] == pytest.approx([0.0, 100.0, 200.0, 300.0])


class TestInterpolateGapsQuadratic:
    def test_parabolic_arc_not_linear(self):
        """Quadratic interpolation produces a curved path, not a straight line."""
        dets = [_det(0, 100.0, 400.0), _det(5, 150.0, 200.0), _det(10, 200.0, 400.0)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        interp_2 = next(d for d in result if d.frame_index == 2)
        linear_y_2 = 400.0 + (2 / 5) * (200.0 - 400.0)  # = 320
        assert abs(interp_2.y - linear_y_2) > 5.0

    def test_falls_back_to_linear_with_two_points(self):
        dets = [_det(0, 0.0, 0.0), _det(4, 40.0, 80.0)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        interp_2 = next(d for d in result if d.frame_index == 2)
        assert abs(interp_2.x - 20.0) < 0.1
        assert abs(interp_2.y - 40.0) < 0.1

    def test_preserves_original_detections(self):
        dets = [_det(0, 100.0, 400.0), _det(3, 130.0, 200.0, conf=0.85), _det(6, 160.0, 400.0)]
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
        dets = [_det(0, 0.0, 0.0), _det(3, 30.0, 30.0)]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        interp_1 = next(d for d in result if d.frame_index == 1)
        interp_2 = next(d for d in result if d.frame_index == 2)
        assert interp_1.interpolated is True
        assert interp_2.interpolated is True

    def test_local_fit_stays_near_anchors(self):
        """Local fitting keeps interpolated points near the gap endpoints, not the global trend."""
        dets = [
            _det(0, 100.0, 200.0),
            _det(5, 300.0, 100.0),
            _det(10, 500.0, 200.0),
            # gap here (frames 11-14)
            _det(15, 400.0, 300.0),
            _det(20, 200.0, 400.0),
        ]
        result = interpolate_gaps(dets, fps=30.0, max_gap_s=0.5)
        interp_12 = next(d for d in result if d.frame_index == 12)
        assert 350.0 < interp_12.x < 550.0
        assert 150.0 < interp_12.y < 350.0


class TestSmoothTrajectory:
    def test_reduces_jitter(self):
        dets = [_det(i, float(i * 10), 200.0 + (10.0 if i % 2 == 0 else -10.0)) for i in range(10)]
        smoothed = smooth_trajectory(dets, window=3)
        assert len(smoothed) == 10
        mid_ys = [d.y for d in smoothed[1:-1]]
        original_ys = [200.0 + (10.0 if i % 2 == 0 else -10.0) for i in range(1, 9)]
        assert np.std(mid_ys) < np.std(original_ys)
        # each interior sample is pulled toward its two opposite-sign neighbours
        assert mid_ys == pytest.approx([200.0 + (-10.0 / 3 if i % 2 == 0 else 10.0 / 3) for i in range(1, 9)])

    def test_preserves_interpolated_flag(self):
        dets = [
            _det(0, 0.0, 0.0, interpolated=False),
            _det(1, 10.0, 10.0, conf=0.5, interpolated=True),
            _det(2, 20.0, 20.0, interpolated=False),
        ]
        smoothed = smooth_trajectory(dets, window=3)
        assert [d.interpolated for d in smoothed] == [False, True, False]

    def test_short_input_unchanged(self):
        dets = [_det(0, 10.0, 20.0)]
        smoothed = smooth_trajectory(dets, window=3)
        assert len(smoothed) == 1
        assert smoothed[0].x == 10.0

    def test_window_of_one_is_identity(self):
        dets = _line(range(5))
        assert smooth_trajectory(dets, window=1) == dets

    def test_preserves_frame_index(self):
        dets = [_det(i, float(i), float(i)) for i in range(5)]
        smoothed = smooth_trajectory(dets, window=3)
        for i, d in enumerate(smoothed):
            assert d.frame_index == i

    def test_confidence_unchanged(self):
        dets = [_det(i, float(i), float(i), conf=0.9 - i * 0.1) for i in range(5)]
        smoothed = smooth_trajectory(dets, window=3)
        for orig, sm in zip(dets, smoothed):
            assert sm.confidence == orig.confidence

    def test_only_consecutive_frames_are_averaged(self):
        """A gap breaks the window: samples on either side never bleed into each other."""
        dets = [_det(0, 0.0, 0.0), _det(1, 0.0, 0.0), _det(2, 0.0, 0.0),
                _det(10, 100.0, 100.0), _det(11, 100.0, 100.0), _det(12, 100.0, 100.0)]
        smoothed = smooth_trajectory(dets, window=3)
        assert [d.y for d in smoothed] == [0.0, 0.0, 0.0, 100.0, 100.0, 100.0]

    def test_edges_average_over_available_neighbours(self):
        dets = [_det(0, 0.0, 0.0), _det(1, 30.0, 0.0), _det(2, 0.0, 0.0)]
        smoothed = smooth_trajectory(dets, window=3)
        assert [d.x for d in smoothed] == pytest.approx([15.0, 10.0, 15.0])


class TestPostprocessTrajectory:
    def test_outlier_removed_gap_filled_and_smoothed(self):
        raw = _line([0, 1, 2, 3, 4, 6, 9, 10, 11, 12])  # frames 5, 7, 8 missing
        raw.insert(5, _det(5, 900.0, 900.0))  # blip off the track at frame 5
        traj = postprocess_trajectory(raw, fps=30.0, max_gap_s=0.5, max_speed_px=150.0, smooth_window=3, min_run_s=0.0)
        assert isinstance(traj, BallTrajectory)
        assert traj.fps == 30.0
        assert [d.frame_index for d in traj.detections] == list(range(13))
        xs = [d.x for d in traj.detections]
        assert xs[1:-1] == pytest.approx([100.0 + 10 * i for i in range(1, 12)])
        assert xs[0] == pytest.approx(105.0) and xs[-1] == pytest.approx(215.0)  # edge windows are one-sided
        assert [d.y for d in traj.detections] == pytest.approx([200.0] * 13)
        assert [d.frame_index for d in traj.detections if d.interpolated] == [5, 7, 8]

    def test_speed_cap_is_scaled_by_fps(self):
        """``max_speed_px`` is per 30fps-frame: at 60fps the per-frame cap halves."""
        raw = _line(range(6), dx=100.0)  # 100 px/frame
        assert len(postprocess_trajectory(raw, fps=30.0, max_speed_px=150.0, min_run_s=0.0).detections) == 6
        assert postprocess_trajectory(raw, fps=60.0, max_speed_px=150.0).detections == []

    def test_empty_input(self):
        traj = postprocess_trajectory([], fps=30.0)
        assert traj.detections == []
        assert traj.fps == 30.0

    def test_smooth_window_one_leaves_positions_untouched(self):
        raw = [_det(i, float(i * 10), 200.0 + (10.0 if i % 2 == 0 else -10.0)) for i in range(8)]
        traj = postprocess_trajectory(raw, fps=30.0, smooth_window=1)
        assert [d.y for d in traj.detections] == [d.y for d in raw]


class TestBuildTrajectory:
    def test_forwards_arguments_to_detection_and_postprocessing(self, tmp_path: Path):
        raw = _line(range(4))
        with patch("court_vision.ball_tracker.detect_ball_sequence", return_value=raw) as detect, \
                patch("court_vision.ball_tracker.postprocess_trajectory", wraps=postprocess_trajectory) as post:
            cb = MagicMock()
            traj = build_trajectory(
                tmp_path, 3, 6, fps=60.0, max_gap_s=0.4, method="tracknet", progress_callback=cb,
                confidence_threshold=0.55, frame_step=2, max_speed_px=120.0, smooth_window=5,
                far_roi=(1, 2, 3, 4),
            )
        detect.assert_called_once_with(
            tmp_path, 3, 6, method="tracknet", confidence_threshold=0.55, frame_step=2,
            progress_callback=cb, far_roi=(1, 2, 3, 4),
        )
        post.assert_called_once_with(raw, 60.0, max_gap_s=0.4, max_speed_px=120.0, smooth_window=5, homography=None, min_run_s=0.12, frame_shape=None)
        assert isinstance(traj, BallTrajectory)
        assert traj.fps == 60.0

    def test_applies_smoothing_to_noisy_detections(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 5)
        noisy = {i: _det(i, 100.0 + 10 * i, 212.0 if i % 2 == 0 else 188.0) for i in range(5)}
        with _patched_detector("tracknet", lambda frames, frame_index, **kw: noisy.get(frame_index)):
            traj = build_trajectory(frames_dir, 0, 4, fps=30.0, method="tracknet", min_run_s=0.0)
        ys = [d.y for d in traj.detections]
        assert len(ys) == 5
        assert np.std(ys) < 10.0  # raw std is ~12

    def test_tracknet_method_uses_tracknet_detector(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 5)
        with _patched_detector("tracknet", lambda frames, frame_index, **kw: None) as mock:
            traj = build_trajectory(frames_dir, 0, 4, fps=30.0, method="tracknet")
        assert mock.call_count == 5
        assert traj.detections == []

    def test_default_method_is_wasb(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, 3)
        with _patched_detector("wasb", lambda frames, frame_index, **kw: None) as mock:
            build_trajectory(frames_dir, 0, 2, fps=30.0)
        assert mock.call_count == 3


class TestRejectStationaryDetections:
    def test_rejects_stationary_detections(self):
        dets = [_det(i, 100.0 + (i % 2), 200.0 + (i % 2), conf=0.8) for i in range(10)]
        result = reject_stationary_detections(dets)
        assert all(d is None for d in result)

    def test_preserves_moving_detections(self):
        dets = [_det(i, 100.0 + i * 20.0, 200.0 + i * 10.0, conf=0.8) for i in range(10)]
        result = reject_stationary_detections(dets)
        non_none = [d for d in result if d is not None]
        assert len(non_none) >= 6

    def test_handles_none_detections(self):
        result = reject_stationary_detections([None, None, None])
        assert all(d is None for d in result)

    def test_handles_sparse_detections(self):
        dets = [
            _det(0, 100.0, 200.0, conf=0.8),
            None,
            _det(2, 101.0, 201.0, conf=0.8),
            None,
            _det(4, 100.5, 200.5, conf=0.8),
        ]
        result = reject_stationary_detections(dets)
        assert result[1] is None
        assert result[3] is None

    def test_short_sequence_unchanged(self):
        dets = [_det(0, 100.0, 200.0, conf=0.8), _det(1, 100.0, 200.0, conf=0.8)]
        result = reject_stationary_detections(dets, window=5)
        assert len(result) == 2
        assert all(d is not None for d in result)

    def test_local_stationary_stretch_inside_a_moving_track(self):
        moving = [_det(i, 100.0 + 30.0 * i, 200.0, conf=0.8) for i in range(6)]
        parked = [_det(6 + i, 280.0, 200.0, conf=0.8) for i in range(6)]
        result = reject_stationary_detections(moving + parked, window=5, std_threshold=5.0)
        assert all(d is not None for d in result[:4])
        assert all(d is None for d in result[8:])

    def test_one_second_window_with_fps(self):
        """With ``fps`` the window is ``window_s`` long, so short tracks are never judged."""
        parked = [_det(i, 100.0 + (i % 2), 200.0, conf=0.8) for i in range(40)]
        assert all(d is None for d in reject_stationary_detections(parked, std_threshold=3.0, fps=30.0))
        short = parked[:20]
        assert reject_stationary_detections(short, std_threshold=3.0, fps=30.0) == short
        assert all(d is None for d in reject_stationary_detections(short, std_threshold=3.0, fps=30.0, window_s=0.5))

    def test_sparse_window_is_not_judged(self):
        """A window must be ``min_fill`` populated before it can be called stationary."""
        moving = [_det(i, 100.0 + 30.0 * i, 200.0, conf=0.8) for i in range(15)]
        parked = [_det(15 + i, 550.0, 200.0, conf=0.8) if i % 2 == 0 else None for i in range(15)]
        dets = moving + parked
        default = reject_stationary_detections(dets, std_threshold=3.0, fps=10.0)  # 10-frame window, 8 needed
        assert all(d is not None for d in default[15::2])
        relaxed = reject_stationary_detections(dets, std_threshold=3.0, fps=10.0, min_fill=0.3)
        assert all(d is None for d in relaxed[21::2])
        assert all(d is None for d in relaxed[16::2])  # the Nones stay None


class TestTrimWeakEdges:
    def test_weak_edges_before_first_and_after_last_strong_are_nulled(self):
        dets = [
            _det(0, 1.0, 1.0, conf=0.5),  # weak, before any strong -> dropped
            None,
            _det(2, 1.0, 1.0, conf=0.8),  # strong
            _det(3, 1.0, 1.0, conf=0.5),  # weak but between strong -> kept
            _det(4, 1.0, 1.0, conf=0.9, interpolated=True),  # interpolated between strong -> kept
            _det(5, 1.0, 1.0, conf=0.8),  # strong
            _det(6, 1.0, 1.0, conf=0.9, interpolated=True),  # interpolated after last strong -> dropped
            _det(7, 1.0, 1.0, conf=0.5),  # weak after last strong -> dropped
            None,
        ]
        result = trim_weak_edges(dets, strong_confidence=0.75)
        assert [None if d is None else d.frame_index for d in result] == [None, None, 2, 3, 4, 5, None, None, None]
        assert result[2] is dets[2] and result[3] is dets[3]

    def test_no_strong_detection_leaves_input_untouched(self):
        dets = [_det(0, 1.0, 1.0, conf=0.5), None, _det(2, 1.0, 1.0, conf=0.6)]
        result = trim_weak_edges(dets, strong_confidence=0.75)
        assert result == dets
        assert result is not dets

    def test_interpolated_detections_are_never_strong(self):
        dets = [_det(0, 1.0, 1.0, conf=0.99, interpolated=True), _det(1, 1.0, 1.0, conf=0.5)]
        assert trim_weak_edges(dets, strong_confidence=0.75) == dets

    def test_threshold_is_configurable(self):
        dets = [_det(0, 1.0, 1.0, conf=0.5), _det(1, 1.0, 1.0, conf=0.8), _det(2, 1.0, 1.0, conf=0.5)]
        assert trim_weak_edges(dets, strong_confidence=0.75) == [None, dets[1], None]
        assert trim_weak_edges(dets, strong_confidence=0.5) == dets

    def test_all_none_and_empty(self):
        assert trim_weak_edges([None, None]) == [None, None]
        assert trim_weak_edges([]) == []


class TestMapBallToCourt:
    def test_maps_detection_to_court_coords(self):
        det = _det(0, 100.0, 200.0)
        court_x, court_y = map_ball_to_court(det, np.eye(3))
        assert abs(court_x - 100.0) < 0.1
        assert abs(court_y - 200.0) < 0.1

    def test_returns_none_for_none_homography(self):
        assert map_ball_to_court(_det(0, 100.0, 200.0), None) is None

    def test_works_with_scaling_homography(self):
        H = np.array([[2.0, 0, 0], [0, 3.0, 0], [0, 0, 1.0]], dtype=np.float64)
        court_x, court_y = map_ball_to_court(_det(0, 10.0, 20.0), H)
        assert abs(court_x - 20.0) < 0.1
        assert abs(court_y - 60.0) < 0.1

    def test_projective_homography_divides_by_w(self):
        H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 2.0]], dtype=np.float64)
        court_x, court_y = map_ball_to_court(_det(0, 10.0, 20.0), H)
        assert (court_x, court_y) == pytest.approx((5.0, 10.0))

    def test_degenerate_projection_returns_origin(self):
        H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0.0]], dtype=np.float64)
        assert map_ball_to_court(_det(0, 10.0, 20.0), H) == (0.0, 0.0)

    def test_linear_court_mapping(self):
        court_x, court_y = map_ball_to_court(_det(0, 640.0, 360.0), H_LINEAR)
        assert (court_x, court_y) == pytest.approx((0.0, 0.0))
        court_x, court_y = map_ball_to_court(_det(0, 690.0, 385.0), H_LINEAR)
        assert (court_x, court_y) == pytest.approx((1.0, -1.0))


class TestTrackletLinking:
    """Short tracklets survive only by linking to a long run (trajectory.reject_velocity_outliers)."""

    def test_short_run_linked_to_long_run_is_kept(self):
        long = _line(range(0, 12))                       # 12 frames at 10 px/frame
        short = _line([14, 15, 16])                       # continues after a 2-frame occlusion (x = 240..260)
        result = reject_velocity_outliers(long + short, fps=30.0, max_speed_px_per_frame=150.0)
        assert [d.frame_index for d in result] == list(range(12)) + [14, 15, 16]

    def test_isolated_short_run_far_away_is_dropped(self):
        long = _line(range(0, 12))
        junk = [_det(f, 900.0, 600.0 + f) for f in (13, 14, 15)]  # a shoe, 800 px away
        result = reject_velocity_outliers(long + junk, fps=30.0, max_speed_px_per_frame=150.0)
        assert [d.frame_index for d in result] == list(range(12))

    def test_short_run_after_a_gap_links_when_speed_matches(self):
        long = _line(range(0, 12))                       # ends at x=210, 10 px/frame
        # 19 frames later (a new tracklet: gap > fps/2), 190 px further on: bridge = 10 px/frame
        short = _line([30, 31, 32], x0=100.0)             # x = 400, 410, 420
        result = reject_velocity_outliers(long + short, fps=30.0, max_speed_px_per_frame=150.0)
        assert [d.frame_index for d in result] == list(range(12)) + [30, 31, 32]

    def test_a_lone_pair_only_links_across_a_short_gap(self):
        long = _line(range(0, 12))
        # two detections are too little evidence to bridge 0.6 s, even at the right speed
        pair = _line([30, 31], x0=100.0)
        assert [d.frame_index for d in reject_velocity_outliers(long + pair, fps=30.0)] == list(range(12))
        # ...but they do link across a quarter second
        near_pair = _line([17, 18], x0=100.0)
        result = reject_velocity_outliers(long + near_pair, fps=30.0)
        assert [d.frame_index for d in result] == list(range(12)) + [17, 18]

    def test_blips_stand_still_and_are_dropped(self):
        track = _line(range(0, 30))                                  # a real flight, 290 px
        head = [_det(f, 900.0, 100.0 + (f % 2)) for f in range(50, 60)]  # a spectator's head
        hopping = [_det(f, 900.0 + 30.0 * (f // 3), 100.0) for f in range(80, 95)]  # detector hopping between heads
        result = reject_velocity_outliers(track + head + hopping, fps=30.0, blip_extent_px=40.0)
        assert [d.frame_index for d in result] == list(range(30))
        # the same runs pass when the check is off
        result = reject_velocity_outliers(track + head + hopping, fps=30.0, blip_extent_px=0.0)
        assert len(result) == 30 + 10 + 15
        # a stationary run of a full second is left to the stationary filter
        held = [_det(f, 900.0, 100.0 + (f % 2)) for f in range(120, 151)]
        assert len(reject_velocity_outliers(track + held, fps=30.0, blip_extent_px=40.0)) == 30 + 31

    def test_a_gap_step_is_held_to_the_local_speed(self):
        """A blip six frames before the ball re-appears far away averages a legal
        speed over the gap, but not a plausible one next to the ball's own speed."""
        from court_vision.trajectory import build_runs

        slow = _line(range(10, 20), x0=700.0, dx=2.0, y=170.0)     # far-court ball, 2 px/frame
        blip = [_det(4, 640.0, 530.0), _det(5, 640.0, 531.0)]      # bottom of the frame, 5 frames earlier
        runs = build_runs(blip + slow, 75.0, max_gap_frames=30)
        assert [len(r) for r in runs] == [2, 10]
        # the same gap is fine when the ball keeps its speed across it
        fast = _line(list(range(0, 5)) + list(range(10, 15)), dx=60.0)
        assert [len(r) for r in build_runs(fast, 75.0, max_gap_frames=30)] == [10]

    def test_short_run_that_bridges_too_fast_is_dropped(self):
        long = _line(range(0, 12))                       # ends at x=210, 10 px/frame
        # 19 frames later but 1200 px away: bridge 63 px/frame, 6x the track speed
        short = [_det(30, 1410.0, 200.0), _det(31, 1420.0, 200.0)]
        result = reject_velocity_outliers(long + short, fps=30.0, max_speed_px_per_frame=150.0)
        assert [d.frame_index for d in result] == list(range(12))

    def test_court_gate_keeps_airborne_far_balls_and_drops_the_stands(self):
        from court_vision.trajectory import court_gate

        # image-like mapping: pixel y = 600 - 20 * court_y (far baseline at y=362, net at 600)
        H = np.array([[0.02, 0.0, -6.4], [0.0, -0.05, 30.0], [0.0, 0.0, 1.0]])
        on_court = _det(0, 400.0, 500.0)      # inside the ground polygon
        far_air = _det(1, 320.0, 250.0)       # above the far baseline: projects ~17 m past it, still a ball
        stands = _det(2, 320.0, 40.0)         # way above the far box
        wide = _det(3, 1200.0, 500.0)         # 17 m outside the sideline
        kept = court_gate([on_court, far_air, stands, wide], H, frame_shape=(720, 1280))
        assert kept == [on_court, far_air]
        assert court_gate([stands], None) == [stands]


class TestShotSpeed:
    def test_bounce_speed_from_feet_to_landing(self):
        from court_vision.ball_speed import bounce_speed

        # 20 m in 0.5 s at 30 fps (15 frames) = 40 m/s = 144 km/h
        est = bounce_speed((0.0, -12.0), (2.0, 7.9, 15), 0, 30.0)
        assert est is not None and est.method == "bounce"
        assert abs(est.kmh - 143.4) < 1.0
        assert bounce_speed((0.0, -12.0), (2.0, 7.9, 2), 0, 30.0) is None  # too short a flight


class TestStaticSpots:
    def test_balls_lying_on_the_court_are_removed_and_the_flight_survives(self):
        from court_vision.trajectory import reject_static_spots

        flight = _line(range(0, 200), x0=100.0, dx=4.0, y=300.0)          # a moving ball
        # a ball on the ground the detector returns every other frame for 5 s
        resting = [_det(f, 900.0 + (f % 3) * 0.5, 500.0) for f in range(0, 300, 2)]
        kept = reject_static_spots(sorted(flight + resting, key=lambda d: d.frame_index), fps=60.0)
        assert all(d.y == 300.0 for d in kept)
        assert len(kept) == 200

    def test_a_contact_zone_crossed_on_every_shot_is_not_a_static_spot(self):
        from court_vision.trajectory import reject_static_spots

        # the ball passes the same spot for 2 frames on each of 20 shots spread over a minute
        crossings = [_det(180 * k + i, 700.0, 200.0) for k in range(20) for i in range(2)]
        assert len(reject_static_spots(crossings, fps=60.0)) == 40

    def test_a_short_pause_is_not_a_static_spot(self):
        from court_vision.trajectory import reject_static_spots

        pause = [_det(f, 500.0, 200.0) for f in range(0, 40)]            # 0.67 s in one place (toss apex, occlusion)
        assert len(reject_static_spots(pause, fps=60.0)) == 40


class TestInterleavedTracks:
    def test_detector_flipping_between_two_balls_keeps_the_moving_one(self):
        """Frame by frame the detector alternates between a flying ball and a
        ball rolling slowly on the ground: both chains must survive as tracks
        and the flight must not shatter into two-frame runs."""
        flight = _line(range(0, 120), x0=100.0, dx=6.0, y=200.0)
        roller = [_det(f, 900.0 + 0.5 * f, 600.0) for f in range(0, 120)]
        alternating = [flight[f] if f % 2 == 0 else roller[f] for f in range(120)]
        kept = reject_velocity_outliers(alternating, fps=60.0, max_speed_px_per_frame=75.0, min_run_s=0.12)
        assert len([d for d in kept if d.y == 200.0]) == 60   # the flight survived intact
        # the roller ran at the same time as the flight: one ball at a time, so it is dropped
        assert all(d.y == 200.0 for d in kept)

    def test_overlapping_tracks_leave_one_ball_per_frame(self):
        from court_vision.trajectory import _resolve_overlaps

        long = _line(range(0, 30), y=100.0)
        short = _line(range(10, 20), y=500.0)       # same frames as part of the long one
        kept = _resolve_overlaps([short, long], min_len=4)
        assert len(kept) == 30 and all(d.y == 100.0 for d in kept)
        # a track inside a hole of the long one is still inside its span: dropped, not interleaved
        holey = _line(list(range(0, 10)) + list(range(20, 30)), y=100.0)
        filler = _line(range(11, 19), y=500.0)
        kept = _resolve_overlaps([filler, holey], min_len=4)
        assert all(d.y == 100.0 for d in kept) and len(kept) == 20
        # but a track after the span is kept
        later = _line(range(40, 50), y=500.0)
        assert len(_resolve_overlaps([later, holey], min_len=4)) == 30


class TestSelectBallPath:
    def _cands(self, frames, moving_conf=0.7, junk_conf=0.9, junk=True):
        out = []
        for f in range(frames):
            c = [_det(f, 100.0 + 8.0 * f, 300.0, moving_conf)]           # the ball in play, 8 px/frame
            if junk:
                c.append(_det(f, 900.0, 500.0, junk_conf))                 # a ball on the ground, stronger blob
            out.append(sorted(c, key=lambda d: -d.confidence))
        return out

    def test_moving_ball_beats_a_stronger_static_blob(self):
        from court_vision.trajectory import select_ball_path

        path = select_ball_path(self._cands(120), fps=60.0, max_speed_px_per_frame=75.0)
        assert len(path) == 120
        assert all(d.y == 300.0 for d in path)

    def test_missing_frames_are_skipped_and_the_path_resumes(self):
        from court_vision.trajectory import select_ball_path

        cands = self._cands(60, junk=False)
        for f in range(20, 26):
            cands[f] = []                                                  # the detector lost the ball for 6 frames
        path = select_ball_path(cands, fps=60.0, max_speed_px_per_frame=75.0)
        assert [d.frame_index for d in path] == [f for f in range(60) if not 20 <= f < 26]

    def test_restart_after_a_long_loss(self):
        from court_vision.trajectory import select_ball_path

        cands = self._cands(120, junk=False)
        for f in range(40, 80):
            cands[f] = []                                                  # lost for 40 frames (> max gap of 30)
        path = select_ball_path(cands, fps=60.0, max_speed_px_per_frame=75.0)
        assert len(path) == 80                                             # both halves kept

    def test_a_drifting_blob_on_the_next_court_is_not_followed(self):
        """Our ball crosses the patch of the picture where a ball on the next
        court creeps along at 1 px/frame with a stronger blob: the path must
        keep our ball's velocity, not hop onto the creeping one."""
        from court_vision.trajectory import select_ball_path

        cands = []
        for f in range(90):
            ours = _det(f, 1100.0 - 12.0 * f, 250.0, 0.65)          # 12 px/frame leftward through the middle
            theirs = _det(f, 620.0 + 1.0 * f, 255.0, 0.9)            # creeping, stronger
            c = [ours, theirs] if not 38 <= f < 44 else [theirs]    # ours missed for 6 frames as it passes
            cands.append(sorted(c, key=lambda d: -d.confidence))
        path = select_ball_path(cands, fps=60.0, max_speed_px_per_frame=75.0)
        assert all(d.confidence == 0.65 for d in path), "hopped onto the creeping ball"
        assert len(path) == 84

    def test_a_jump_over_the_cap_is_not_followed(self):
        from court_vision.trajectory import select_ball_path

        cands = self._cands(40, junk=False)
        cands[20] = [_det(20, 1200.0, 100.0, 0.95)]                       # one wild blob mid-flight
        path = select_ball_path(cands, fps=60.0, max_speed_px_per_frame=75.0)
        assert 20 not in [d.frame_index for d in path]
        assert len(path) == 39
