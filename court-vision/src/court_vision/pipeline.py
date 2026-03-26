"""Pipeline orchestrator — runs all stages in sequence."""

from dataclasses import dataclass
from pathlib import Path

from court_vision.config import PipelineConfig, load_config
from court_vision.device import get_device
from court_vision.ingest import (
    FrameSequence,
    download_video,
    extract_frames,
    is_youtube_url,
)
from court_vision.scene_filter import (
    GameplaySegment,
    classify_frames,
    filter_gameplay_segments,
    load_scene_model,
)


@dataclass
class PipelineResult:
    """Result of a pipeline run (Phase 1: ingest + scene filter)."""

    source: str
    total_frames: int
    fps: float
    gameplay_segments: list[GameplaySegment]
    gameplay_frame_count: int
    frames_dir: Path


def run_pipeline(
    source: str,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    scene_weights_path: Path | None = None,
) -> PipelineResult:
    """Run the Court Vision pipeline on a video source.

    Phase 1 stages: video ingestion -> scene filter.

    Args:
        source: YouTube URL or local video file path.
        config_path: Path to court-vision.yaml config. None for defaults.
        output_dir: Directory for pipeline output. None for config default.
        scene_weights_path: Path to fine-tuned scene filter weights.
                            None uses ImageNet pre-trained base.

    Returns:
        PipelineResult with frame data and gameplay segments.
    """
    config = load_config(config_path)
    device = get_device(override=config.device)

    if output_dir is None:
        output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Video Ingestion
    if is_youtube_url(source):
        video_path = download_video(source, output_dir)
    else:
        video_path = Path(source)

    resolution = tuple(config.pipeline.target_resolution)
    frame_seq = extract_frames(video_path, target_resolution=resolution)

    # Stage 2: Scene Filter
    model = load_scene_model(scene_weights_path, device)
    results = classify_frames(frame_seq.frames_dir, frame_seq.total_frames, model, device)
    segments = filter_gameplay_segments(results, frame_seq.fps)

    gameplay_frames = sum(seg.frame_count for seg in segments)

    return PipelineResult(
        source=source,
        total_frames=frame_seq.total_frames,
        fps=frame_seq.fps,
        gameplay_segments=segments,
        gameplay_frame_count=gameplay_frames,
        frames_dir=frame_seq.frames_dir,
    )
