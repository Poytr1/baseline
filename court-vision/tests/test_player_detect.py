"""Tests for the player detection module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from court_vision.player_detect import (
    PlayerDetection,
    PoseKeypoints,
    FrameTrackingResult,
    detect_players_in_frame,
)


class TestPlayerDetection:
    def test_player_detection_fields(self):
        """PlayerDetection has required fields."""
        det = PlayerDetection(
            frame_index=0,
            bbox=(100.0, 200.0, 150.0, 400.0),
            confidence=0.92,
        )
        assert det.frame_index == 0
        assert det.bbox == (100.0, 200.0, 150.0, 400.0)
        assert det.confidence == 0.92
        assert det.court_position is None
        assert det.role is None

    def test_player_detection_with_court_position(self):
        """PlayerDetection supports court_position and role."""
        det = PlayerDetection(
            frame_index=0,
            bbox=(100.0, 200.0, 150.0, 400.0),
            confidence=0.92,
            court_position=(2.0, -8.0),
            role="near_player",
        )
        assert det.court_position == (2.0, -8.0)
        assert det.role == "near_player"


class TestPoseKeypoints:
    def test_pose_keypoints_fields(self):
        """PoseKeypoints has required fields."""
        pk = PoseKeypoints(
            frame_index=0,
            role="far_player",
            keypoints={"left_wrist": (100.0, 200.0, 0.95)},
        )
        assert pk.frame_index == 0
        assert pk.role == "far_player"
        assert pk.keypoints["left_wrist"] == (100.0, 200.0, 0.95)


class TestFrameTrackingResult:
    def test_frame_tracking_fields(self):
        """FrameTrackingResult combines ball, players, and poses."""
        from court_vision.ball_tracker import BallDetection

        result = FrameTrackingResult(
            frame_index=0,
            ball=BallDetection(frame_index=0, x=300.0, y=200.0, confidence=0.8),
            players=[],
            poses=[],
        )
        assert result.frame_index == 0
        assert result.ball is not None
        assert result.players == []
        assert result.poses == []


class TestDetectPlayersInFrame:
    @patch("court_vision.player_detect._get_yolo_model")
    def test_detects_persons(self, mock_get_model: MagicMock):
        """Detects persons in frame using YOLOv8."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_box = MagicMock()
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[100.0, 200.0, 200.0, 500.0]])
        mock_box.conf = MagicMock()
        mock_box.conf.cpu.return_value.numpy.return_value = np.array([0.92])
        mock_box.cls = MagicMock()
        mock_box.cls.cpu.return_value.numpy.return_value = np.array([0])

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_model.return_value = [mock_result]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        players = detect_players_in_frame(frame, frame_index=0)

        assert len(players) == 1
        assert players[0].bbox == (100.0, 200.0, 200.0, 500.0)
        assert players[0].confidence == pytest.approx(0.92, abs=0.01)

    @patch("court_vision.player_detect._get_yolo_model")
    def test_filters_non_person_classes(self, mock_get_model: MagicMock):
        """Only person class (0) detections are returned."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_box = MagicMock()
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([
            [100.0, 200.0, 200.0, 500.0],
            [300.0, 100.0, 400.0, 200.0],
        ])
        mock_box.conf = MagicMock()
        mock_box.conf.cpu.return_value.numpy.return_value = np.array([0.92, 0.85])
        mock_box.cls = MagicMock()
        mock_box.cls.cpu.return_value.numpy.return_value = np.array([0, 32])

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_model.return_value = [mock_result]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        players = detect_players_in_frame(frame, frame_index=0)

        assert len(players) == 1
        assert players[0].bbox[0] == 100.0
