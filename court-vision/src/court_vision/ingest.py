"""Video ingestion — YouTube download and frame extraction."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameSequence:
    """Container for extracted video frames with metadata."""

    frames_dir: Path
    fps: float
    total_frames: int
    resolution: tuple[int, int]

    def frame_path(self, index: int) -> Path:
        """Get the path to a specific frame image."""
        return self.frames_dir / f"frame_{index:06d}.jpg"


def is_youtube_url(source: str) -> bool:
    """Check if a source string is a YouTube URL."""
    return "youtube.com/watch" in source or "youtu.be/" in source


def download_video(url: str, output_dir: Path) -> Path:
    """Download a YouTube video using yt-dlp.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded video.

    Returns:
        Path to the downloaded video file.

    Raises:
        RuntimeError: If yt-dlp download fails.
    """
    output_path = output_dir / "video.mp4"
    result = subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", str(output_path),
            url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr}")

    return output_path


def extract_frames(
    video_path: Path,
    target_resolution: tuple[int, int] = (1280, 720),
    output_dir: Path | None = None,
) -> FrameSequence:
    """Extract frames from a video file, resizing to target resolution.

    Args:
        video_path: Path to the input video file.
        target_resolution: (width, height) to resize frames to.
        output_dir: Directory to write frame images. If None, uses
                    a 'frames' subdirectory next to the video.

    Returns:
        FrameSequence with metadata and paths to extracted frames.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = target_resolution

    if output_dir is None:
        output_dir = video_path.parent / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height))
        frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(frame_path), resized)
        frame_count += 1

    cap.release()

    return FrameSequence(
        frames_dir=output_dir,
        fps=fps,
        total_frames=frame_count,
        resolution=target_resolution,
    )
