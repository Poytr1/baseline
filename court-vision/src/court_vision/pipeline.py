"""Pipeline orchestrator — runs all stages in sequence."""

from dataclasses import dataclass
from pathlib import Path

from court_vision.config import PipelineConfig, load_config
from court_vision.court_detect import CourtDetectionResult, compute_segment_homographies
from court_vision.device import get_device
from court_vision.ball_fusion import snap_ball_to_racket_at_contacts, trim_segment_edges
from court_vision.ingest import (
    FrameSequence,
    download_video,
    extract_frames,
    is_youtube_url,
)
from court_vision.player_detect import (
    FrameTrackingResult,
    build_ball_trajectory,
    detect_players_segment,
)
from court_vision.progress import make_callback, pipeline_progress
from court_vision.scene_filter import (
    GameplaySegment,
    classify_frames,
    filter_gameplay_segments,
    load_scene_model,
)
from court_vision.shot_classify import MatchData, build_match_data, detect_contacts


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
    show_progress: bool = True,
) -> PipelineResult:
    """Run the Court Vision pipeline on a video source.

    Stages: video ingestion -> scene filter -> court detection -> tracking -> shot classification.

    Args:
        source: YouTube URL or local video file path.
        config_path: Path to court-vision.yaml config. None for defaults.
        output_dir: Directory for pipeline output. None for config default.
        scene_weights_path: Path to fine-tuned scene filter weights.
                            None uses ImageNet pre-trained base.
        show_progress: Show Rich progress bars for each stage.

    Returns:
        PipelineResult with frame data, gameplay segments, court detections, and match data.
    """
    config = load_config(config_path)
    device = get_device(override=config.device)

    if output_dir is None:
        output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    with pipeline_progress(show_progress) as progress:
        # Stage 1a: Download (YouTube only)
        if is_youtube_url(source):
            t = progress.add_task("Downloading video", total=None) if progress else None
            video_path = download_video(source, output_dir)
            if progress:
                progress.update(t, total=1, completed=1)
        else:
            video_path = Path(source)

        # Stage 1b: Extract frames
        t = progress.add_task("Extracting frames", total=None) if progress else None
        resolution = tuple(config.pipeline.target_resolution)
        frame_seq = extract_frames(
            video_path, target_resolution=resolution,
            progress_callback=make_callback(progress, t),
        )
        if progress:
            progress.update(t, completed=frame_seq.total_frames, total=frame_seq.total_frames)

        # Stage 2: Scene Filter
        t = progress.add_task("Classifying scenes", total=frame_seq.total_frames) if progress else None
        if config.pipeline.scene_filter_mode == "heuristic":
            from court_vision.heuristic_scene_filter import (
                classify_frames_heuristic,
                smooth_classifications,
            )
            results = classify_frames_heuristic(
                frame_seq.frames_dir, frame_seq.total_frames,
                gameplay_threshold=config.pipeline.gameplay_threshold,
                progress_callback=make_callback(progress, t),
                stride=config.pipeline.scene_filter_stride,
            )
            results = smooth_classifications(results)
        else:
            model = load_scene_model(scene_weights_path, device)
            results = classify_frames(
                frame_seq.frames_dir, frame_seq.total_frames, model, device,
                progress_callback=make_callback(progress, t),
                stride=config.pipeline.scene_filter_stride,
            )
        segments = filter_gameplay_segments(results, frame_seq.fps)
        if progress:
            progress.update(t, completed=frame_seq.total_frames)

        gameplay_frames = sum(seg.frame_count for seg in segments)

        # Stage 3: Court Detection & Homography
        t = progress.add_task("Detecting court", total=len(segments)) if progress else None
        court_detections = compute_segment_homographies(frame_seq.frames_dir, segments)
        if progress:
            progress.update(t, completed=len(segments), total=len(segments))

        # Stage 4a: Ball Tracking (all segments)
        total_tracking_frames = sum(s.end_frame - s.start_frame + 1 for s in segments)
        t_ball = progress.add_task("Tracking ball", total=total_tracking_frames) if progress else None
        ball_trajectories = []
        ball_done = 0
        for segment in segments:
            def ball_cb(cur, tot, _done=ball_done):
                if progress:
                    progress.update(t_ball, completed=_done + cur)

            trajectory = build_ball_trajectory(
                frame_seq.frames_dir, segment,
                fps=frame_seq.fps,
                ball_method=config.pipeline.ball_detection_method,
                progress_callback=ball_cb if progress else None,
            )
            ball_trajectories.append(trajectory)
            ball_done += segment.end_frame - segment.start_frame + 1
        if progress:
            progress.update(t_ball, completed=total_tracking_frames)

        # Stage 4b: Player Detection + Pose (all segments)
        t_player = progress.add_task("Detecting players & pose", total=total_tracking_frames) if progress else None
        all_tracking: list[FrameTrackingResult] = []
        player_done = 0
        for i, segment in enumerate(segments):
            homography = None
            if court_detections and i < len(court_detections) and court_detections[i].success:
                homography = court_detections[i].homography

            def player_cb(cur, tot, _done=player_done):
                if progress:
                    progress.update(t_player, completed=_done + cur)

            segment_tracking = detect_players_segment(
                frame_seq.frames_dir, segment, ball_trajectories[i],
                homography=homography,
                progress_callback=player_cb if progress else None,
                player_detect_stride=config.pipeline.player_detect_stride,
            )
            all_tracking.extend(segment_tracking)
            player_done += segment.end_frame - segment.start_frame + 1

        # Stage 4c: Ball-fusion refinement — trim segment edges and
        # snap ball-at-contact to the hitter's wrist.
        segment_ranges = [(seg.start_frame, seg.end_frame) for seg in segments]
        all_tracking = trim_segment_edges(all_tracking, segment_ranges)
        contacts = detect_contacts(all_tracking, frame_seq.fps)
        all_tracking = snap_ball_to_racket_at_contacts(all_tracking, contacts)

        # Stage 5: Shot Classification
        t = progress.add_task("Classifying shots", total=1) if progress else None
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
        if progress:
            progress.update(t, completed=1)

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
