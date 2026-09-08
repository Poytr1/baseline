"""Per-stage detection metrics for auto-tuning.

Scores the court / ball / player stages independently against the labelled
fixture in labels.py, given a list of cached FrameTrackingResult (per-frame
ball + player detections) and the court detections produced by the pipeline.

All metrics are oriented so the tuner can treat them uniformly:
  - *_error_px  : lower is better (Euclidean pixel distance)
  - *_rate / *_accuracy / *_iou : higher is better, in [0, 1]

Reuses court_detect._compute_reprojection_error and compute_homography so the
court score is the same math the pipeline uses internally.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from court_vision.court_detect import _compute_reprojection_error, compute_homography
from court_vision.player_detect import FrameTrackingResult

from scripts.autotune.labels import LabelSet


@dataclass
class CourtStageMetrics:
    frames_scored: int
    reproj_error_px: float        # mean reprojection error over labelled corners (lower better)
    success_rate: float           # fraction of labelled frames with a usable homography


@dataclass
class BallStageMetrics:
    frames_scored: int
    detect_rate: float            # of visible-ball labels, fraction the pipeline also detected
    loc_error_px: float           # mean localization error on matched frames (lower better)
    false_positive_rate: float    # of not-visible labels, fraction pipeline wrongly detected


@dataclass
class PlayerStageMetrics:
    frames_scored: int
    count_accuracy: float         # fraction of labelled frames with correct player count
    mean_iou: float               # mean best-match IoU over labelled boxes (higher better)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_court(
    labels: LabelSet,
    court_homographies: dict[int, np.ndarray | None],
) -> CourtStageMetrics:
    """Score court detection.

    Args:
        labels: label fixture (uses labels.court).
        court_homographies: frame_index -> homography (pixel->court meters) or None.
            The pipeline produces one homography per segment; the caller is
            responsible for mapping each labelled frame to the homography that
            covers it (or None if that segment failed).
    """
    from court_vision.court_detect import COURT_KEYPOINTS

    court_pts = np.array(
        [
            COURT_KEYPOINTS["baseline_near_left_doubles"][:2],
            COURT_KEYPOINTS["baseline_near_right_doubles"][:2],
            COURT_KEYPOINTS["baseline_far_right_doubles"][:2],
            COURT_KEYPOINTS["baseline_far_left_doubles"][:2],
        ],
        dtype=np.float64,
    )

    errors: list[float] = []
    successes = 0
    scored = 0
    for cl in labels.court:
        ordered = cl.ordered_pixels()
        if ordered is None:
            continue
        scored += 1
        H = court_homographies.get(cl.frame)
        if H is None:
            continue
        successes += 1
        pixel_pts = np.array(ordered, dtype=np.float64)
        errors.append(_compute_reprojection_error(pixel_pts, court_pts, H))

    return CourtStageMetrics(
        frames_scored=scored,
        reproj_error_px=float(np.mean(errors)) if errors else math.inf,
        success_rate=successes / scored if scored else 0.0,
    )


def labelled_homographies(labels: LabelSet) -> dict[int, np.ndarray]:
    """Build the *ideal* homography per labelled court frame from the labels
    themselves. Useful as a sanity check (reproj error should be ~0) and as the
    reference when the pipeline maps players to court coordinates."""
    from court_vision.court_detect import COURT_KEYPOINTS

    court_pts = np.array(
        [
            COURT_KEYPOINTS["baseline_near_left_doubles"][:2],
            COURT_KEYPOINTS["baseline_near_right_doubles"][:2],
            COURT_KEYPOINTS["baseline_far_right_doubles"][:2],
            COURT_KEYPOINTS["baseline_far_left_doubles"][:2],
        ],
        dtype=np.float64,
    )
    out: dict[int, np.ndarray] = {}
    for cl in labels.court:
        ordered = cl.ordered_pixels()
        if ordered is None:
            continue
        H = compute_homography(np.array(ordered, dtype=np.float64), court_pts)
        if H is not None:
            out[cl.frame] = H
    return out


def score_ball(
    labels: LabelSet,
    tracking: list[FrameTrackingResult],
    match_radius_px: float = 40.0,
) -> BallStageMetrics:
    """Score ball detection against labelled ball centers.

    A labelled visible ball counts as detected if the pipeline produced a
    (non-interpolated) ball within match_radius_px.
    """
    by_frame = {t.frame_index: t for t in tracking}

    visible = [b for b in labels.ball if b.visible and b.x is not None]
    not_visible = [b for b in labels.ball if not b.visible]

    matched = 0
    loc_errors: list[float] = []
    for bl in visible:
        t = by_frame.get(bl.frame)
        if t is None or t.ball is None or getattr(t.ball, "interpolated", False):
            continue
        d = math.hypot(t.ball.x - bl.x, t.ball.y - bl.y)
        if d <= match_radius_px:
            matched += 1
            loc_errors.append(d)

    false_pos = 0
    for bl in not_visible:
        t = by_frame.get(bl.frame)
        if t is not None and t.ball is not None and not getattr(t.ball, "interpolated", False):
            false_pos += 1

    return BallStageMetrics(
        frames_scored=len(visible) + len(not_visible),
        detect_rate=matched / len(visible) if visible else 0.0,
        loc_error_px=float(np.mean(loc_errors)) if loc_errors else math.inf,
        false_positive_rate=false_pos / len(not_visible) if not_visible else 0.0,
    )


def score_players(
    labels: LabelSet,
    tracking: list[FrameTrackingResult],
) -> PlayerStageMetrics:
    """Score player detection: count accuracy + mean best-match IoU."""
    by_frame = {t.frame_index: t for t in tracking}

    count_correct = 0
    ious: list[float] = []
    scored = 0
    for pl in labels.players:
        scored += 1
        t = by_frame.get(pl.frame)
        pred_boxes = [p.bbox for p in t.players] if t else []
        if len(pred_boxes) == len(pl.boxes):
            count_correct += 1
        for gt in pl.boxes:
            best = max((_iou(gt.bbox, pb) for pb in pred_boxes), default=0.0)
            ious.append(best)

    return PlayerStageMetrics(
        frames_scored=scored,
        count_accuracy=count_correct / scored if scored else 0.0,
        mean_iou=float(np.mean(ious)) if ious else 0.0,
    )


def to_dict(m) -> dict:
    return asdict(m)
