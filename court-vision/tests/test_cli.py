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
