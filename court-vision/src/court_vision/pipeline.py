"""Pipeline orchestrator — runs all stages in sequence."""

from dataclasses import dataclass
from pathlib import Path

from court_vision.config import PipelineConfig, load_config
from court_vision.court_detect import CourtDetectionResult, compute_segment_homographies
from court_vision.device import get_device
from court_vision.ingest import (
    FrameSequence,
    download_video,
    extract_frames,
    is_youtube_url,
)
from court_vision.player_detect import FrameTrackingResult, track_segment
from court_vision.scene_filter import (
    GameplaySegment,
    classify_frames,
    filter_gameplay_segments,
    load_scene_model,
)
from court_vision.shot_classify import MatchData, build_match_data


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    source: str
    total_frames: int
    fps: float
    gameplay_segments: list[GameplaySegment]
    gameplay_frame_count: int
    frames_dir: Path
    court_detections: list[CourtDetectionResult] | None = None
    tracking_results: list[FrameTrackingResult] | None = None
    match_data: MatchData | None = None


def run_pipeline(
    source: str,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    scene_weights_path: Path | None = None,
) -> PipelineResult:
    """Run the Court Vision pipeline on a video source.

    Stages: video ingestion -> scene filter -> court detection -> tracking -> shot classification.

    Args:
        source: YouTube URL or local video file path.
        config_path: Path to court-vision.yaml config. None for defaults.
        output_dir: Directory for pipeline output. None for config default.
        scene_weights_path: Path to fine-tuned scene filter weights.
                            None uses ImageNet pre-trained base.

    Returns:
        PipelineResult with frame data, gameplay segments, court detections, and match data.
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
    if config.pipeline.scene_filter_mode == "heuristic":
        from court_vision.heuristic_scene_filter import (
            classify_frames_heuristic,
            smooth_classifications,
        )
        results = classify_frames_heuristic(
            frame_seq.frames_dir, frame_seq.total_frames,
            gameplay_threshold=config.pipeline.gameplay_threshold,
        )
        results = smooth_classifications(results)
    else:
        model = load_scene_model(scene_weights_path, device)
        results = classify_frames(frame_seq.frames_dir, frame_seq.total_frames, model, device)
    segments = filter_gameplay_segments(results, frame_seq.fps)

    gameplay_frames = sum(seg.frame_count for seg in segments)

    # Stage 3: Court Detection & Homography
    court_detections = compute_segment_homographies(frame_seq.frames_dir, segments)

    # Stage 4: Ball Tracking + Player Detection + Pose
    all_tracking: list[FrameTrackingResult] = []
    for i, segment in enumerate(segments):
        homography = None
        if court_detections and i < len(court_detections) and court_detections[i].success:
            homography = court_detections[i].homography
        segment_tracking = track_segment(frame_seq.frames_dir, segment, homography, fps=frame_seq.fps)
        all_tracking.extend(segment_tracking)

    # Stage 5: Shot Classification
    court_homography = None
    for cd in (court_detections or []):
        if cd.success and cd.homography is not None:
            court_homography = cd.homography
            break

    match_data = build_match_data(
        source=source,
        segments=segments,
        tracking_results=all_tracking,
        fps=frame_seq.fps,
        court_homography=court_homography,
    )

    return PipelineResult(
        source=source,
        total_frames=frame_seq.total_frames,
        fps=frame_seq.fps,
        gameplay_segments=segments,
        gameplay_frame_count=gameplay_frames,
        frames_dir=frame_seq.frames_dir,
        court_detections=court_detections,
        tracking_results=all_tracking,
        match_data=match_data,
    )
