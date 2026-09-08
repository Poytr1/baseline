"""Tests for video ingestion — download and frame extraction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from court_vision.ingest import (
    FrameSequence,
    download_video,
    extract_frames,
    is_youtube_url,
)


class TestIsYoutubeUrl:
    def test_standard_url(self):
        assert is_youtube_url("https://www.youtube.com/watch?v=abc123") is True

    def test_short_url(self):
        assert is_youtube_url("https://youtu.be/abc123") is True

    def test_local_file(self):
        assert is_youtube_url("/path/to/match.mp4") is False

    def test_other_url(self):
        assert is_youtube_url("https://example.com/video.mp4") is False


class TestDownloadVideo:
    @patch("court_vision.ingest.subprocess.run")
    def test_download_calls_ytdlp(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(returncode=0)
        output_path = tmp_path / "video.mp4"

        # Use a side_effect to create the file when yt-dlp "runs"
        def create_file(*args, **kwargs):
            output_path.touch()
            return MagicMock(returncode=0)
        mock_run.side_effect = create_file

        result = download_video(
            "https://www.youtube.com/watch?v=abc123", tmp_path
        )

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "yt-dlp" in call_args
        assert "https://www.youtube.com/watch?v=abc123" in call_args
        assert result == output_path

    @patch("court_vision.ingest.subprocess.run")
    def test_download_raises_on_failure(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        with pytest.raises(RuntimeError, match="yt-dlp download failed"):
            download_video("https://www.youtube.com/watch?v=abc123", tmp_path)


class TestExtractFrames:
    def test_extract_frames_returns_frame_sequence(self, tmp_path: Path):
        """Extract frames from a synthetic video file."""
        video_path = _create_test_video(tmp_path / "test.mp4", num_frames=10)

        result = extract_frames(video_path, target_resolution=(1280, 720))

        assert isinstance(result, FrameSequence)
        assert result.fps > 0
        assert result.total_frames == 10
        assert result.resolution == (1280, 720)
        assert (result.frames_dir / "frame_000000.jpg").exists()
        assert (result.frames_dir / "frame_000009.jpg").exists()

    def test_extract_frames_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            extract_frames(Path("/nonexistent/video.mp4"))


def _create_test_video(path: Path, num_frames: int = 10, fps: int = 30) -> Path:
    """Create a minimal synthetic video for testing."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for _ in range(num_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
