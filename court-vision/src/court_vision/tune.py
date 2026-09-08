"""Tune pipeline parameters via grid search on cached tracking data."""

import hashlib
import itertools
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from court_vision.ball_tracker import BallDetection
from court_vision.evaluate import EvaluationResult, match_shots
from court_vision.player_detect import (
    FrameTrackingResult,
    PlayerDetection,
    PoseKeypoints,
)
from court_vision.review_data import load_match_json
from court_vision.scene_filter import GameplaySegment
from court_vision.shot_classify import (
    MatchData,
    Point,
    Shot,
    classify_stroke,
    detect_contacts,
    detect_point_boundaries,
)

logger = logging.getLogger(__name__)


@dataclass
class TuneResult:
    """Result from a single parameter combination."""

    params: dict
    evaluation: EvaluationResult


def load_tracking_results(tracking_path: Path) -> list[FrameTrackingResult]:
    """Load per-frame tracking results from JSON.

    Args:
        tracking_path: Path to tracking_data.json.

    Returns:
        List of FrameTrackingResult sorted by frame index.
    """
    with open(tracking_path) as f:
        raw = json.load(f)

    results: list[FrameTrackingResult] = []
    for entry in raw:
        ball = None
        if entry.get("ball"):
            b = entry["ball"]
            ball = BallDetection(
                frame_index=b["frame_index"],
                x=b["x"],
                y=b["y"],
                confidence=b["confidence"],
                interpolated=b.get("interpolated", False),
            )
        players = [
            PlayerDetection(
                frame_index=p["frame_index"],
                bbox=tuple(p["bbox"]),
                confidence=p["confidence"],
                court_position=tuple(p["court_position"]) if p.get("court_position") else None,
                role=p.get("role"),
            )
            for p in entry.get("players", [])
        ]
        poses = [
            PoseKeypoints(
                frame_index=pk["frame_index"],
                role=pk["role"],
                keypoints={k: tuple(v) for k, v in pk["keypoints"].items()},
            )
            for pk in entry.get("poses", [])
        ]
        results.append(FrameTrackingResult(
            frame_index=entry["frame_index"],
            ball=ball,
            players=players,
            poses=poses,
        ))

    return sorted(results, key=lambda r: r.frame_index)


def reclassify_shots(
    tracking_results: list[FrameTrackingResult],
    segments: list[GameplaySegment],
    source: str,
    fps: float,
    proximity_threshold: float = 100.0,
    min_frames_between_contacts: int = 5,
) -> MatchData:
    """Re-run shot classification with given parameters on cached tracking data.

    Only re-runs detect_contacts + build_match_data (Stage 5).
    Does NOT re-run neural inference (Stages 1-4).

    Args:
        tracking_results: Cached per-frame tracking data.
        segments: Gameplay segments (point boundaries).
        source: Source video path (for match ID).
        fps: Video frame rate.
        proximity_threshold: Ball-player contact distance threshold.
        min_frames_between_contacts: Min frames between consecutive contacts.

    Returns:
        Re-classified MatchData.
    """
    contacts = detect_contacts(
        tracking_results,
        fps,
        proximity_threshold=proximity_threshold,
        min_frames_between_contacts=min_frames_between_contacts,
    )

    tracking_by_frame = {t.frame_index: t for t in tracking_results}

    match_id = hashlib.md5(source.encode()).hexdigest()[:12]
    boundaries = detect_point_boundaries(segments)

    points: list[Point] = []
    for point_num, (start, end, start_t, end_t) in enumerate(boundaries, 1):
        point_contacts = [
            (frame, role) for frame, role in contacts
            if start <= frame <= end
        ]

        shots: list[Shot] = []
        for shot_num, (frame, role) in enumerate(point_contacts, 1):
            tracking = tracking_by_frame.get(frame)
            stroke = "forehand"
            stroke_conf = 0.4
            if tracking and tracking.poses:
                player_pose = next(
                    (p for p in tracking.poses if p.role == role), None
                )
                if player_pose:
                    stroke, stroke_conf = classify_stroke(player_pose)

            shots.append(Shot(
                shot_number=shot_num,
                frame=frame,
                time_s=frame / fps,
                player=role,
                stroke=stroke,
                placement=None,
                confidence=stroke_conf,
            ))

        outcome = None
        outcome_player = None
        if shots:
            last_shot = shots[-1]
            outcome = "winner" if last_shot.confidence > 0.6 else "error"
            outcome_player = last_shot.player

        points.append(Point(
            point_number=point_num,
            start_frame=start,
            end_frame=end,
            start_time_s=start_t,
            end_time_s=end_t,
            server=shots[0].player if shots else None,
            shots=shots,
            outcome=outcome,
            outcome_player=outcome_player,
            rally_length=len(shots),
        ))

    return MatchData(
        match_id=match_id,
        source_url=source,
        metadata={"players": ["near_player", "far_player"], "date_processed": str(date.today())},
        points=points,
    )


def grid_search(
    ground_truth_path: Path,
    tracking_path: Path,
    source: str = "data/test_input_video.mp4",
    fps: float = 30.0,
    proximity_values: list[float] | None = None,
    min_frames_values: list[int] | None = None,
) -> list[TuneResult]:
    """Grid search over contact detection parameters.

    Args:
        ground_truth_path: Path to corrected match_data.json.
        tracking_path: Path to cached tracking_data.json.
        source: Source video path (for match ID computation).
        fps: Video frame rate.
        proximity_values: Values for proximity_threshold to try.
        min_frames_values: Values for min_frames_between_contacts to try.

    Returns:
        List of TuneResult sorted by F1 score (best first).
    """
    if proximity_values is None:
        proximity_values = [50.0, 75.0, 100.0, 125.0, 150.0, 200.0]
    if min_frames_values is None:
        min_frames_values = [3, 5, 8, 10, 15, 20, 25, 30]

    ground_truth = load_match_json(ground_truth_path)
    tracking_results = load_tracking_results(tracking_path)

    segments = [
        GameplaySegment(
            start_frame=p.start_frame,
            end_frame=p.end_frame,
            start_time_s=p.start_time_s,
            end_time_s=p.end_time_s,
            frame_count=p.end_frame - p.start_frame + 1,
        )
        for p in ground_truth.points
    ]

    results: list[TuneResult] = []

    for prox, min_frames in itertools.product(proximity_values, min_frames_values):
        predicted = reclassify_shots(
            tracking_results=tracking_results,
            segments=segments,
            source=source,
            fps=fps,
            proximity_threshold=prox,
            min_frames_between_contacts=min_frames,
        )
        evaluation = match_shots(ground_truth, predicted)
        results.append(TuneResult(
            params={"proximity_threshold": prox, "min_frames_between_contacts": min_frames},
            evaluation=evaluation,
        ))

    results.sort(key=lambda r: (-r.evaluation.f1, -r.evaluation.precision))
    return results
