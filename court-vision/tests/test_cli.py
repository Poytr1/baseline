"""Tests for the CLI entry point."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from court_vision.cli import app

runner = CliRunner()


def test_version_command():
    """Version command prints the current version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help_shows_commands():
    """Top-level help lists available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "process" in result.output
    assert "version" in result.output


class TestProcessCommandOutput:
    @patch("court_vision.pipeline.run_pipeline")
    def test_process_shows_court_detection_summary(self, mock_pipeline: MagicMock):
        """Process command displays court detection results."""
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.scene_filter import GameplaySegment

        mock_pipeline.return_value = MagicMock(
            total_frames=100,
            fps=30.0,
            gameplay_segments=[
                GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
            ],
            gameplay_frame_count=51,
            court_detections=[
                CourtDetectionResult(success=True, num_lines_detected=7),
            ],
        )

        result = runner.invoke(app, ["process", "test.mp4"])

        assert result.exit_code == 0
        assert "court" in result.output.lower() or "homography" in result.output.lower()


class TestProcessCommandTrackingOutput:
    @patch("court_vision.pipeline.run_pipeline")
    def test_process_shows_tracking_summary(self, mock_pipeline: MagicMock):
        """Process command displays tracking results summary."""
        from court_vision.ball_tracker import BallDetection
        from court_vision.court_detect import CourtDetectionResult
        from court_vision.player_detect import FrameTrackingResult, PlayerDetection
        from court_vision.scene_filter import GameplaySegment

        mock_pipeline.return_value = MagicMock(
            total_frames=100,
            fps=30.0,
            gameplay_segments=[
                GameplaySegment(start_frame=0, end_frame=50, start_time_s=0.0, end_time_s=1.67, frame_count=51),
            ],
            gameplay_frame_count=51,
            court_detections=[
                CourtDetectionResult(success=True, num_lines_detected=7),
            ],
            tracking_results=[
                FrameTrackingResult(
                    frame_index=0,
                    ball=BallDetection(frame_index=0, x=300.0, y=200.0, confidence=0.8),
                    players=[
                        PlayerDetection(frame_index=0, bbox=(100.0, 300.0, 200.0, 600.0), confidence=0.9, role="near_player"),
                    ],
                    poses=[],
                ),
                FrameTrackingResult(frame_index=1, ball=None, players=[], poses=[]),
            ],
        )

        result = runner.invoke(app, ["process", "test.mp4"])

        assert result.exit_code == 0
        assert "tracking" in result.output.lower() or "ball" in result.output.lower()

