"""Tests for the pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from court_vision.pipeline import PipelineResult, run_pipeline


class TestRunPipeline:
    @patch("court_vision.pipeline.build_match_data")
    @patch("court_vision.pipeline.track_segment")
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.classify_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_local_file_pipeline(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_homographies: MagicMock,
        mock_track: MagicMock,
        mock_build_match: MagicMock,
        tmp_path: Path,
    ):
        """Pipeline orchestrates ingest -> scene filter for a local file."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.ingest import FrameSequence
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        from court_vision.config import PipelineSettings

        mock_load_config.return_value = PipelineConfig(
            pipeline=PipelineSettings(scene_filter_mode="ml")
        )
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        mock_extract.return_value = FrameSequence(
            frames_dir=tmp_path / "frames",
            fps=30.0,
            total_frames=100,
            resolution=(1280, 720),
        )
        mock_classify.return_value = []
        mock_filter.return_value = [
            GameplaySegment(
                start_frame=0, end_frame=50,
                start_time_s=0.0, end_time_s=1.67,
                frame_count=51,
            )
        ]
        mock_homographies.return_value = []
        mock_track.return_value = []
        mock_build_match.return_value = None

        result = run_pipeline(str(video_path), config_path=None)

        assert isinstance(result, PipelineResult)
        assert result.total_frames == 100
        assert len(result.gameplay_segments) == 1
        mock_extract.assert_called_once()
        mock_classify.assert_called_once()

    @patch("court_vision.pipeline.build_match_data")
    @patch("court_vision.pipeline.track_segment")
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.download_video")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.classify_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_youtube_url_triggers_download(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_download: MagicMock,
        mock_homographies: MagicMock,
        mock_track: MagicMock,
        mock_build_match: MagicMock,
        tmp_path: Path,
    ):
        """YouTube URLs trigger yt-dlp download before frame extraction."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.ingest import FrameSequence

        from court_vision.config import PipelineSettings

        mock_load_config.return_value = PipelineConfig(
            pipeline=PipelineSettings(scene_filter_mode="ml")
        )
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        downloaded = tmp_path / "video.mp4"
        downloaded.touch()
        mock_download.return_value = downloaded
        mock_extract.return_value = FrameSequence(
            frames_dir=tmp_path / "frames",
            fps=30.0,
            total_frames=50,
            resolution=(1280, 720),
        )
        mock_classify.return_value = []
        mock_filter.return_value = []
        mock_homographies.return_value = []
        mock_track.return_value = []
        mock_build_match.return_value = None

        result = run_pipeline(
            "https://www.youtube.com/watch?v=abc123",
            config_path=None,
            output_dir=tmp_path,
        )

        mock_download.assert_called_once()
        assert result.total_frames == 50


class TestRunPipelineWithCourtDetection:
    @patch("court_vision.pipeline.build_match_data")
    @patch("court_vision.pipeline.track_segment")
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.classify_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_pipeline_runs_court_detection_after_scene_filter(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_homographies: MagicMock,
        mock_track: MagicMock,
        mock_build_match: MagicMock,
        tmp_path: Path,
    ):
        """Pipeline calls compute_segment_homographies after scene filter."""
        import torch

        from court_vision.config import PipelineConfig, PipelineSettings
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.ingest import FrameSequence
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        mock_load_config.return_value = PipelineConfig(
            pipeline=PipelineSettings(scene_filter_mode="ml")
        )
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        mock_extract.return_value = FrameSequence(
            frames_dir=frames_dir, fps=30.0, total_frames=100, resolution=(1280, 720),
        )
        mock_classify.return_value = []

        segments = [
            GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
        ]
        mock_filter.return_value = segments

        court_result = CourtDetectionResult(success=True, homography=MagicMock(), num_lines_detected=6)
        mock_homographies.return_value = [court_result]
        mock_track.return_value = []
        mock_build_match.return_value = None

        result = run_pipeline(str(video_path), config_path=None)

        mock_homographies.assert_called_once_with(frames_dir, segments)
        assert result.court_detections is not None
        assert len(result.court_detections) == 1
        assert result.court_detections[0].success is True


class TestRunPipelineWithTracking:
    @patch("court_vision.pipeline.build_match_data")
    @patch("court_vision.pipeline.track_segment")
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.classify_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_pipeline_runs_tracking_after_court_detection(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_homographies: MagicMock,
        mock_track: MagicMock,
        mock_build_match: MagicMock,
        tmp_path: Path,
    ):
        """Pipeline calls track_segment after court detection."""
        import torch

        from court_vision.config import PipelineConfig, PipelineSettings
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.ingest import FrameSequence
        from court_vision.player_detect import FrameTrackingResult
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        mock_load_config.return_value = PipelineConfig(
            pipeline=PipelineSettings(scene_filter_mode="ml")
        )
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        mock_extract.return_value = FrameSequence(
            frames_dir=frames_dir, fps=30.0, total_frames=100, resolution=(1280, 720),
        )
        mock_classify.return_value = []

        segments = [
            GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
        ]
        mock_filter.return_value = segments

        court_result = CourtDetectionResult(success=True, num_lines_detected=6)
        mock_homographies.return_value = [court_result]

        mock_track.return_value = [
            FrameTrackingResult(frame_index=0, ball=None, players=[], poses=[]),
        ]
        mock_build_match.return_value = None

        result = run_pipeline(str(video_path), config_path=None)

        mock_track.assert_called_once()
        assert result.tracking_results is not None
        assert len(result.tracking_results) == 1


class TestRunPipelineWithShotClassification:
    @patch("court_vision.pipeline.build_match_data")
    @patch("court_vision.pipeline.track_segment")
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.classify_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_pipeline_runs_shot_classification(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_homographies: MagicMock,
        mock_track: MagicMock,
        mock_build_match: MagicMock,
        tmp_path: Path,
    ):
        """Pipeline calls build_match_data after tracking."""
        import torch

        from court_vision.config import PipelineConfig, PipelineSettings
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.ingest import FrameSequence
        from court_vision.scene_filter import GameplaySegment
        from court_vision.shot_classify import MatchData

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        mock_load_config.return_value = PipelineConfig(
            pipeline=PipelineSettings(scene_filter_mode="ml")
        )
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        mock_extract.return_value = FrameSequence(
            frames_dir=frames_dir, fps=30.0, total_frames=100, resolution=(1280, 720),
        )
        mock_classify.return_value = []
        segments = [
            GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
        ]
        mock_filter.return_value = segments
        mock_homographies.return_value = [CourtDetectionResult(success=True, num_lines_detected=6)]
        mock_track.return_value = []
        mock_build_match.return_value = MatchData(
            match_id="test", source_url="test.mp4",
            metadata={}, points=[],
        )

        result = run_pipeline(str(video_path), config_path=None)

        mock_build_match.assert_called_once()
        assert result.match_data is not None


class TestRunPipelineWithHeuristicFilter:
    @patch("court_vision.pipeline.build_match_data")
    @patch("court_vision.pipeline.track_segment")
    @patch("court_vision.pipeline.compute_segment_homographies")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_heuristic_mode_uses_heuristic_filter(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_extract: MagicMock,
        mock_homographies: MagicMock,
        mock_track: MagicMock,
        mock_build_match: MagicMock,
        tmp_path: Path,
    ):
        """Pipeline uses heuristic filter when scene_filter_mode='heuristic'."""
        import torch

        from court_vision.config import PipelineConfig, PipelineSettings
        from court_vision.ingest import FrameSequence

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        config = PipelineConfig(pipeline=PipelineSettings(scene_filter_mode="heuristic"))
        mock_load_config.return_value = config
        mock_get_device.return_value = torch.device("cpu")
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        mock_extract.return_value = FrameSequence(
            frames_dir=frames_dir, fps=30.0, total_frames=100, resolution=(1280, 720),
        )
        mock_filter.return_value = []
        mock_homographies.return_value = []
        mock_track.return_value = []
        mock_build_match.return_value = None

        with patch("court_vision.heuristic_scene_filter.classify_frames_heuristic", return_value=[]) as mock_heuristic, \
             patch("court_vision.heuristic_scene_filter.smooth_classifications", return_value=[]) as mock_smooth:
            result = run_pipeline(str(video_path), config_path=None)

        # Heuristic filter should be used, ML model should NOT be loaded
        mock_load_model.assert_not_called()
        mock_heuristic.assert_called_once()
        mock_smooth.assert_called_once()


class TestTrackSegmentUsesBuildTrajectory:
    @patch("court_vision.player_detect.estimate_pose")
    @patch("court_vision.player_detect.detect_players_in_frame")
    @patch("court_vision.player_detect.build_trajectory")
    def test_track_segment_calls_build_trajectory(
        self,
        mock_build_traj: MagicMock,
        mock_detect_players: MagicMock,
        mock_estimate_pose: MagicMock,
        tmp_path: Path,
    ):
        """track_segment uses build_trajectory instead of per-frame detect_ball_in_frame."""
        import numpy as np

        from court_vision.ball_tracker import BallDetection, BallTrajectory
        from court_vision.player_detect import track_segment
        from court_vision.scene_filter import GameplaySegment

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(5):
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segment = GameplaySegment(
            start_frame=0, end_frame=4,
            start_time_s=0.0, end_time_s=0.13, frame_count=5,
        )

        mock_build_traj.return_value = BallTrajectory(
            detections=[
                BallDetection(frame_index=0, x=100.0, y=200.0, confidence=0.8),
                BallDetection(frame_index=2, x=150.0, y=250.0, confidence=0.7),
            ],
            fps=30.0,
        )
        mock_detect_players.return_value = []
        mock_estimate_pose.return_value = None

        results = track_segment(frames_dir, segment)

        mock_build_traj.assert_called_once()
        # Frame 0 and 2 should have ball data, frames 1/3/4 should have None
        ball_frames = {r.frame_index: r.ball for r in results}
        assert ball_frames.get(0) is not None
        assert ball_frames.get(2) is not None
        assert ball_frames.get(1) is None
