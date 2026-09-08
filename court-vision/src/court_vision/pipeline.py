"""Pipeline orchestrator — runs all stages in sequence.

The pipeline is decomposed into stage functions (``stage_*``) that each take
explicit inputs and return plain dataclasses. ``run_pipeline`` composes them for
the CLI; the research harness composes the same functions with an on-disk cache
so that tuning a late-stage knob never re-runs neural inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from court_vision.ball_tracker import BallDetection
from court_vision.config import PipelineConfig, load_config
from court_vision.court_detect import CourtDetectionResult, compute_segment_homographies
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
from court_vision.heuristic_scene_filter import classify_frames_heuristic, smooth_classifications
from court_vision.scene_filter import GameplaySegment, filter_gameplay_segments
from court_vision.scoreboard import ScoreTimeline, read_scoreboard_timeline
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
    scoreboard: ScoreTimeline | None = None


def auto_stride(value: int | str, fps: float) -> int:
    """Resolve an ``"auto"`` stride to ``round(fps / 30)`` (min 1)."""
    if value == "auto":
        return max(1, int(round(fps / 30.0)))
    return max(1, int(value))


# ── Stage functions ──────────────────────────────────────────────────────────

def stage_ingest(
    source: str,
    config: PipelineConfig,
    output_dir: Path,
    progress_callback=None,
) -> FrameSequence:
    """Stage 1: download (if URL) and extract frames."""
    if is_youtube_url(source):
        video_path = download_video(source, output_dir)
    else:
        video_path = Path(source)
    resolution = tuple(config.pipeline.target_resolution)
    return extract_frames(video_path, target_resolution=resolution, progress_callback=progress_callback)


def stage_scene(
    frame_seq: FrameSequence,
    config: PipelineConfig,
    progress_callback=None,
) -> list[GameplaySegment]:
    """Stage 2: classify frames (court colour + line heuristics) and group
    contiguous gameplay into segments."""
    p = config.pipeline
    results = classify_frames_heuristic(
        frame_seq.frames_dir, frame_seq.total_frames,
        gameplay_threshold=p.gameplay_threshold,
        progress_callback=progress_callback,
        stride=p.scene_filter_stride,
    )
    results = smooth_classifications(results, window_size=p.scene_smooth_window)
    segments = filter_gameplay_segments(results, frame_seq.fps)
    min_frames = int(p.min_segment_s * frame_seq.fps)
    return [s for s in segments if s.frame_count >= min_frames]


def stage_court(
    frame_seq: FrameSequence,
    segments: list[GameplaySegment],
    config: PipelineConfig,
) -> list[CourtDetectionResult]:
    """Stage 3: one homography per gameplay segment."""
    return compute_segment_homographies(frame_seq.frames_dir, segments, method=config.pipeline.court_method)


def ball_far_roi_for(
    court_detections: list[CourtDetectionResult] | None,
    index: int,
    frame_shape: tuple[int, ...],
    enabled: bool,
) -> tuple[int, int, int, int] | None:
    from court_vision.ball_tracker import far_ball_roi

    if not enabled or not court_detections or index >= len(court_detections):
        return None
    cd = court_detections[index]
    return far_ball_roi(cd.homography, frame_shape) if cd.success else None


def stage_ball(
    frame_seq: FrameSequence,
    segments: list[GameplaySegment],
    config: PipelineConfig,
    progress_callback=None,
    court_detections: list[CourtDetectionResult] | None = None,
) -> list[dict[int, BallDetection | None]]:
    """Stage 4a: per-segment ball trajectories keyed by frame index."""
    p = config.pipeline
    total = sum(s.end_frame - s.start_frame + 1 for s in segments)
    done = 0
    out = []
    shape = (frame_seq.resolution[1], frame_seq.resolution[0])
    for i, segment in enumerate(segments):
        def cb(cur, _tot, _done=done):
            if progress_callback:
                progress_callback(_done + cur, total)

        out.append(build_ball_trajectory(
            frame_seq.frames_dir, segment,
            fps=frame_seq.fps,
            ball_method=p.ball_detection_method,
            confidence_threshold=p.ball_confidence_threshold,
            frame_step=auto_stride(p.ball_frame_step, frame_seq.fps),
            far_roi=ball_far_roi_for(court_detections, i, shape, p.ball_far_crop),
            max_speed_px=p.ball_max_speed_px,
            max_gap_s=p.ball_max_gap_s,
            smooth_window=p.ball_smooth_window,
            stationary_std_px=p.ball_stationary_std_px,
            strong_confidence=p.ball_strong_confidence,
            progress_callback=cb if progress_callback else None,
        ))
        done += segment.end_frame - segment.start_frame + 1
    return out


def stage_players(
    frame_seq: FrameSequence,
    segments: list[GameplaySegment],
    court_detections: list[CourtDetectionResult] | None,
    ball_trajectories: list[dict[int, BallDetection | None]],
    config: PipelineConfig,
    progress_callback=None,
) -> list[FrameTrackingResult]:
    """Stage 4b: players + poses per frame, merged with the ball trajectory."""
    p = config.pipeline
    total = sum(s.end_frame - s.start_frame + 1 for s in segments)
    done = 0
    all_tracking: list[FrameTrackingResult] = []
    for i, segment in enumerate(segments):
        homography = None
        if court_detections and i < len(court_detections) and court_detections[i].success:
            homography = court_detections[i].homography

        def cb(cur, _tot, _done=done):
            if progress_callback:
                progress_callback(_done + cur, total)

        ball_by_frame = ball_trajectories[i] if i < len(ball_trajectories) else {}
        all_tracking.extend(detect_players_segment(
            frame_seq.frames_dir, segment, ball_by_frame,
            homography=homography,
            progress_callback=cb if progress_callback else None,
            player_detect_stride=auto_stride(p.player_detect_stride, frame_seq.fps),
            model_name=p.player_model,
            imgsz=p.player_imgsz,
            conf=p.player_conf,
            far_crop=p.player_far_crop,
            max_court_x=p.player_max_court_x,
            max_court_y=p.player_max_court_y,
        ))
        done += segment.end_frame - segment.start_frame + 1
    return all_tracking


def stage_scoreboard(
    frame_seq: FrameSequence,
    config: PipelineConfig,
) -> ScoreTimeline:
    """Stage 4c: OCR the broadcast scoreboard on a sparse frame sample."""
    return read_scoreboard_timeline(
        frame_seq.frames_dir, frame_seq.total_frames, frame_seq.fps,
        sample_s=config.pipeline.scoreboard_sample_s,
    )


def stage_shots(
    source: str,
    frame_seq: FrameSequence,
    segments: list[GameplaySegment],
    court_detections: list[CourtDetectionResult] | None,
    tracking: list[FrameTrackingResult],
    config: PipelineConfig,
    scoreboard: ScoreTimeline | None = None,
) -> MatchData:
    """Stage 5: contacts, strokes, points, outcomes."""
    court_homography = None
    for cd in (court_detections or []):
        if cd.success and cd.homography is not None:
            court_homography = cd.homography
            break
    segment_homographies = [
        (cd.homography if cd.success else None) for cd in (court_detections or [])
    ]
    return build_match_data(
        source=source,
        segments=segments,
        tracking_results=tracking,
        fps=frame_seq.fps,
        court_homography=court_homography,
        settings=config.pipeline,
        scoreboard=scoreboard,
        segment_homographies=segment_homographies,
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_pipeline(
    source: str,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    show_progress: bool = True,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the Court Vision pipeline on a video source.

    Stages: video ingestion -> scene filter -> court detection -> tracking -> shot classification.

    Args:
        source: YouTube URL or local video file path.
        config_path: Path to court-vision.yaml config. None for defaults.
        output_dir: Directory for pipeline output. None for config default.
        show_progress: Show Rich progress bars for each stage.
        config: Pre-built config (takes precedence over ``config_path``).

    Returns:
        PipelineResult with frame data, gameplay segments, court detections, and match data.
    """
    if config is None:
        config = load_config(config_path)

    if output_dir is None:
        output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    with pipeline_progress(show_progress) as progress:
        t = progress.add_task("Extracting frames", total=None) if progress else None
        frame_seq = stage_ingest(source, config, output_dir, progress_callback=make_callback(progress, t))
        if progress:
            progress.update(t, completed=frame_seq.total_frames, total=frame_seq.total_frames)

        t = progress.add_task("Classifying scenes", total=frame_seq.total_frames) if progress else None
        segments = stage_scene(frame_seq, config, progress_callback=make_callback(progress, t))
        if progress:
            progress.update(t, completed=frame_seq.total_frames)
        gameplay_frames = sum(seg.frame_count for seg in segments)

        t = progress.add_task("Detecting court", total=len(segments)) if progress else None
        court_detections = stage_court(frame_seq, segments, config)
        if progress:
            progress.update(t, completed=len(segments), total=len(segments))

        total_tracking_frames = sum(s.end_frame - s.start_frame + 1 for s in segments)
        t = progress.add_task("Tracking ball", total=total_tracking_frames) if progress else None
        ball_trajectories = stage_ball(
            frame_seq, segments, config, progress_callback=make_callback(progress, t),
            court_detections=court_detections,
        )
        if progress:
            progress.update(t, completed=total_tracking_frames)

        t = progress.add_task("Detecting players & pose", total=total_tracking_frames) if progress else None
        all_tracking = stage_players(
            frame_seq, segments, court_detections, ball_trajectories, config,
            progress_callback=make_callback(progress, t),
        )
        if progress:
            progress.update(t, completed=total_tracking_frames)

        t = progress.add_task("Reading scoreboard", total=1) if progress else None
        scoreboard = None
        if config.pipeline.outcome_method in ("auto", "scoreboard"):
            try:
                scoreboard = stage_scoreboard(frame_seq, config)
            except Exception:  # OCR is best-effort; never fail the pipeline on it
                scoreboard = None
        if progress:
            progress.update(t, completed=1)

        t = progress.add_task("Classifying shots", total=1) if progress else None
        match_data = stage_shots(
            source, frame_seq, segments, court_detections, all_tracking, config,
            scoreboard=scoreboard,
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
        scoreboard=scoreboard,
    )
