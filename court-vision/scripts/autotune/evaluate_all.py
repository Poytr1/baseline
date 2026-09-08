"""Unified scorecard — the single source of truth for the auto-tuner.

Every experiment (config sweep, code rewrite, model swap) is judged by the JSON
this module emits. It combines:

  - final shot metrics (precision / recall / F1 / stroke acc / player acc) from
    court_vision.evaluate.match_shots, and
  - per-stage metrics (court reproj px + success, ball loc px + detect rate,
    player count acc + IoU) from scripts.autotune.metrics_stages.

Two evaluation modes:

  --mode cached  (fast, default): re-run only Stage 5 (contact detection +
      shot classification) on a cached tracking_data.json, using
      tune.reclassify_shots. Per-stage ball/player metrics are scored against
      the cached tracking. Court is scored against whatever homographies are
      supplied (or skipped). Use this for parameter sweeps over Stage-5 knobs.

  --mode full    (slow): run the whole pipeline end-to-end with a config file
      via court_vision.pipeline.run_pipeline, then score everything from the
      fresh PipelineResult. Use this when a change affects Stages 1-4
      (court/ball/player code or models).

CLI:
    python -m scripts.autotune.evaluate_all \
        --mode cached \
        --ground-truth data/match_data.json \
        --tracking data/tracking_data.json \
        --labels data/autotune_labels.json \
        --proximity 100 --min-frames 5 \
        --out scorecard.json

The combined `score` field is a single scalar the tuner maximizes. It is a
weighted blend that (a) rewards higher F1 / accuracy / detect-rate and
(b) penalizes pixel errors via a smooth 1/(1+err) transform, so a missing
stage (inf error) contributes 0 rather than crashing the comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from court_vision.evaluate import match_shots
from court_vision.review_data import load_match_json
from court_vision.scene_filter import GameplaySegment

from scripts.autotune.labels import LabelSet, empty_labels, load_labels
from scripts.autotune.metrics_stages import (
    BallStageMetrics,
    CourtStageMetrics,
    PlayerStageMetrics,
    score_ball,
    score_court,
    score_players,
)


def _px_score(err: float, scale: float) -> float:
    """Map a pixel error in [0, inf) to (0, 1]; 0px -> 1.0, inf -> 0.0."""
    if not math.isfinite(err):
        return 0.0
    return 1.0 / (1.0 + err / scale)


@dataclass
class Scorecard:
    mode: str
    params: dict
    # final-stage
    precision: float
    recall: float
    f1: float
    stroke_accuracy: float
    player_accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    # per-stage (None when not scored in this run)
    court: dict | None = None
    ball: dict | None = None
    players: dict | None = None
    # gates
    tests_passed: int | None = None
    tests_total: int | None = None
    # blended objective (higher is better)
    score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def combined_score(
    f1: float,
    stroke_acc: float,
    player_acc: float,
    court: CourtStageMetrics | None,
    ball: BallStageMetrics | None,
    players: PlayerStageMetrics | None,
) -> float:
    """Blend final + per-stage signals into one scalar in roughly [0, 1].

    Weights favour final-shot F1 (the product goal) while still rewarding
    healthy upstream stages so the tuner doesn't win F1 by luck on a broken
    court/ball stage. Stages that weren't scored are simply omitted and the
    remaining weights renormalize.
    """
    terms: list[tuple[float, float]] = []  # (weight, value)
    terms.append((0.40, f1))
    terms.append((0.10, stroke_acc))
    terms.append((0.10, player_acc))
    if court is not None and court.frames_scored:
        court_val = 0.5 * court.success_rate + 0.5 * _px_score(court.reproj_error_px, scale=2.0)
        terms.append((0.15, court_val))
    if ball is not None and ball.frames_scored:
        ball_val = (
            0.6 * ball.detect_rate
            + 0.3 * _px_score(ball.loc_error_px, scale=15.0)
            + 0.1 * (1.0 - ball.false_positive_rate)
        )
        terms.append((0.15, ball_val))
    if players is not None and players.frames_scored:
        player_val = 0.5 * players.count_accuracy + 0.5 * players.mean_iou
        terms.append((0.10, player_val))

    wsum = sum(w for w, _ in terms)
    if wsum == 0:
        return 0.0
    return sum(w * v for w, v in terms) / wsum


def _segment_homographies_for_labels(
    labels: LabelSet,
    segments: list[GameplaySegment],
    court_detections,
) -> dict[int, object]:
    """Map each labelled court frame to the homography of the segment covering
    it (or None). court_detections is the list aligned to segments from
    PipelineResult.court_detections."""
    out: dict[int, object] = {}
    for cl in labels.court:
        H = None
        for i, seg in enumerate(segments):
            if seg.start_frame <= cl.frame <= seg.end_frame:
                if court_detections and i < len(court_detections) and court_detections[i].success:
                    H = court_detections[i].homography
                break
        out[cl.frame] = H
    return out


def evaluate_cached(
    ground_truth_path: Path,
    tracking_path: Path,
    labels: LabelSet,
    source: str,
    fps: float,
    proximity_threshold: float,
    min_frames_between_contacts: float,
) -> Scorecard:
    from court_vision.tune import load_tracking_results, reclassify_shots

    gt = load_match_json(ground_truth_path)
    tracking = load_tracking_results(tracking_path)

    segments = [
        GameplaySegment(
            start_frame=p.start_frame,
            end_frame=p.end_frame,
            start_time_s=p.start_time_s,
            end_time_s=p.end_time_s,
            frame_count=p.end_frame - p.start_frame + 1,
        )
        for p in gt.points
    ]

    predicted = reclassify_shots(
        tracking_results=tracking,
        segments=segments,
        source=source,
        fps=fps,
        proximity_threshold=proximity_threshold,
        min_frames_between_contacts=int(min_frames_between_contacts),
    )
    ev = match_shots(gt, predicted)

    ball_m = score_ball(labels, tracking) if labels.ball else None
    player_m = score_players(labels, tracking) if labels.players else None
    # No fresh court homographies in cached mode; skip court unless labels carry
    # their own reference (handled in full mode instead).
    court_m = None

    score = combined_score(ev.f1, ev.stroke_accuracy, ev.player_accuracy, court_m, ball_m, player_m)

    from scripts.autotune.metrics_stages import to_dict

    return Scorecard(
        mode="cached",
        params={
            "proximity_threshold": proximity_threshold,
            "min_frames_between_contacts": int(min_frames_between_contacts),
        },
        precision=ev.precision,
        recall=ev.recall,
        f1=ev.f1,
        stroke_accuracy=ev.stroke_accuracy,
        player_accuracy=ev.player_accuracy,
        true_positives=ev.true_positives,
        false_positives=ev.false_positives,
        false_negatives=ev.false_negatives,
        court=None,
        ball=to_dict(ball_m) if ball_m else None,
        players=to_dict(player_m) if player_m else None,
        score=score,
    )


def evaluate_full(
    ground_truth_path: Path,
    labels: LabelSet,
    source: str,
    config_path: Path | None,
) -> Scorecard:
    from court_vision.export import export_json
    from court_vision.pipeline import run_pipeline
    from scripts.autotune.metrics_stages import to_dict

    gt = load_match_json(ground_truth_path)
    result = run_pipeline(source=source, config_path=config_path, show_progress=False)

    # Final metrics: persist + reload through the same path the CLI uses.
    tmp = Path(".autotune_pred_match.json")
    export_json(result.match_data, tmp)
    predicted = load_match_json(tmp)
    tmp.unlink(missing_ok=True)
    ev = match_shots(gt, predicted)

    tracking = result.tracking_results or []
    ball_m = score_ball(labels, tracking) if labels.ball else None
    player_m = score_players(labels, tracking) if labels.players else None
    court_m = None
    if labels.court:
        homs = _segment_homographies_for_labels(labels, result.gameplay_segments, result.court_detections)
        court_m = score_court(labels, homs)

    score = combined_score(ev.f1, ev.stroke_accuracy, ev.player_accuracy, court_m, ball_m, player_m)

    return Scorecard(
        mode="full",
        params={"config": str(config_path) if config_path else "defaults"},
        precision=ev.precision,
        recall=ev.recall,
        f1=ev.f1,
        stroke_accuracy=ev.stroke_accuracy,
        player_accuracy=ev.player_accuracy,
        true_positives=ev.true_positives,
        false_positives=ev.false_positives,
        false_negatives=ev.false_negatives,
        court=to_dict(court_m) if court_m else None,
        ball=to_dict(ball_m) if ball_m else None,
        players=to_dict(player_m) if player_m else None,
        score=score,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit a unified auto-tune scorecard as JSON.")
    ap.add_argument("--mode", choices=["cached", "full"], default="cached")
    ap.add_argument("--ground-truth", type=Path, default=Path("data/match_data.json"))
    ap.add_argument("--tracking", type=Path, default=Path("data/tracking_data.json"))
    ap.add_argument("--labels", type=Path, default=None, help="Per-stage label fixture JSON.")
    ap.add_argument("--source", type=str, default="data/test_input_video.mp4")
    ap.add_argument("--config", type=Path, default=None, help="Pipeline config YAML (full mode).")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--proximity", type=float, default=100.0)
    ap.add_argument("--min-frames", type=float, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    labels = load_labels(args.labels) if args.labels and args.labels.exists() else empty_labels()

    if args.mode == "cached":
        card = evaluate_cached(
            args.ground_truth, args.tracking, labels,
            source=args.source, fps=args.fps,
            proximity_threshold=args.proximity,
            min_frames_between_contacts=args.min_frames,
        )
    else:
        card = evaluate_full(args.ground_truth, labels, source=args.source, config_path=args.config)

    out_json = card.to_json()
    if args.out:
        args.out.write_text(out_json)
    print(out_json)


if __name__ == "__main__":
    main()
