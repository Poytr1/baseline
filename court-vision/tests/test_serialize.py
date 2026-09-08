"""Round-trip tests for the JSON (de)serialization helpers."""

import json
from pathlib import Path

import numpy as np
import pytest

from court_vision import serialize as ser
from court_vision.ball_tracker import BallDetection
from court_vision.court_detect import CourtDetectionResult
from court_vision.player_detect import FrameTrackingResult, PlayerDetection, PoseKeypoints
from court_vision.scene_filter import GameplaySegment


class TestBall:
    def test_round_trip(self):
        ball = BallDetection(frame_index=5, x=1.5, y=2.5, confidence=0.7, interpolated=True)
        d = json.loads(json.dumps(ser.ball_to_dict(ball)))
        assert ser.ball_from_dict(d) == ball

    def test_none_and_empty(self):
        assert ser.ball_to_dict(None) is None
        assert ser.ball_from_dict(None) is None
        assert ser.ball_from_dict({}) is None

    def test_interpolated_defaults_false(self):
        assert ser.ball_from_dict({"frame_index": 1, "x": 0, "y": 0, "confidence": 0.5}).interpolated is False


class TestTracking:
    def _tracking(self) -> FrameTrackingResult:
        return FrameTrackingResult(
            frame_index=5,
            ball=BallDetection(5, 1.5, 2.5, 0.7, True),
            players=[
                PlayerDetection(5, (1.0, 2.0, 3.0, 4.0), 0.9, court_position=(0.5, -3.0), role="near_player"),
                PlayerDetection(5, (10.0, 20.0, 30.0, 40.0), 0.8),
            ],
            poses=[PoseKeypoints(5, "near_player", {"nose": (1.0, 2.0, 0.9), "left_wrist": (3.0, 4.0, 0.5)})],
        )

    def test_round_trip_through_json(self):
        t = self._tracking()
        raw = json.loads(json.dumps(ser.tracking_to_dict(t)))
        back = ser.tracking_from_dict(raw)
        assert back == t
        assert isinstance(back.players[0].bbox, tuple)
        assert isinstance(back.players[0].court_position, tuple)
        assert back.players[1].court_position is None and back.players[1].role is None
        assert isinstance(back.poses[0].keypoints["nose"], tuple)

    def test_list_round_trip_sorts_by_frame(self):
        t5 = self._tracking()
        t2 = FrameTrackingResult(2, None, [], [])
        raw = json.loads(json.dumps(ser.tracking_list_to_json([t5, t2])))
        back = ser.tracking_list_from_json(raw)
        assert back == [t2, t5]

    def test_missing_optional_keys(self):
        t = ser.tracking_from_dict({"frame_index": 3})
        assert t == FrameTrackingResult(3, None, [], [])


class TestSegment:
    def test_round_trip(self):
        seg = GameplaySegment(3, 9, 0.1, 0.3, 7)
        assert ser.segment_from_dict(json.loads(json.dumps(ser.segment_to_dict(seg)))) == seg

    def test_frame_count_defaults_from_range(self):
        seg = ser.segment_from_dict({"start_frame": 3, "end_frame": 9, "start_time_s": 0.1, "end_time_s": 0.3})
        assert seg.frame_count == 7


class TestCourt:
    def test_round_trip(self):
        c = CourtDetectionResult(True, np.eye(3) * 2, [(1.0, 2.0), (3.0, 4.0)], 7)
        d = json.loads(json.dumps(ser.court_to_dict(c)))
        back = ser.court_from_dict(d)
        assert back.success is True
        assert np.array_equal(back.homography, c.homography)
        assert back.homography.dtype == np.float64
        assert back.pixel_keypoints == [(1.0, 2.0), (3.0, 4.0)]
        assert back.num_lines_detected == 7

    def test_failed_detection(self):
        d = json.loads(json.dumps(ser.court_to_dict(CourtDetectionResult(False))))
        back = ser.court_from_dict(d)
        assert back.success is False
        assert back.homography is None
        assert back.pixel_keypoints is None
        assert back.num_lines_detected == 0

    def test_minimal_dict(self):
        assert ser.court_from_dict({"success": False}).homography is None
