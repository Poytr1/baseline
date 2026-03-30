"""Tests for the pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from court_vision.pipeline import PipelineResult, run_pipeline


class TestRunPipeline:
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
        tmp_path: Path,
    ):
        """Pipeline orchestrates ingest -> scene filter for a local file."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.ingest import FrameSequence
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        mock_load_config.return_value = PipelineConfig()
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

        result = run_pipeline(str(video_path), config_path=None)

        assert isinstance(result, PipelineResult)
        assert result.total_frames == 100
        assert len(result.gameplay_segments) == 1
        mock_extract.assert_called_once()
        mock_classify.assert_called_once()

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
        tmp_path: Path,
    ):
        """YouTube URLs trigger yt-dlp download before frame extraction."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.ingest import FrameSequence

        mock_load_config.return_value = PipelineConfig()
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

        result = run_pipeline(
            "https://www.youtube.com/watch?v=abc123",
            config_path=None,
            output_dir=tmp_path,
        )

        mock_download.assert_called_once()
        assert result.total_frames == 50


class TestRunPipelineWithCourtDetection:
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
        tmp_path: Path,
    ):
        """Pipeline calls compute_segment_homographies after scene filter."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.ingest import FrameSequence
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        mock_load_config.return_value = PipelineConfig()
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

        result = run_pipeline(str(video_path), config_path=None)

        mock_homographies.assert_called_once_with(frames_dir, segments)
        assert result.court_detections is not None
        assert len(result.court_detections) == 1
        assert result.court_detections[0].success is True


class TestRunPipelineWithTracking:
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
        tmp_path: Path,
    ):
        """Pipeline calls track_segment after court detection."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.ingest import FrameSequence
        from court_vision.player_detect import FrameTrackingResult
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        mock_load_config.return_value = PipelineConfig()
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

        result = run_pipeline(str(video_path), config_path=None)

        mock_track.assert_called_once()
        assert result.tracking_results is not None
        assert len(result.tracking_results) == 1
