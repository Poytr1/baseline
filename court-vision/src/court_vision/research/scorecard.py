"""Unified scorecard: everything the tuner (and the user) cares about.

Usability criteria, in order:  points segmented correctly, point winners
correct, ball tracked, forehand/backhand right, serves and slices found.
Each gets its own metric; ``score`` blends them into a single number to
rank experiments. Everything is computed against the clip's human-corrected
``match_data.json`` (points/shots/winners) and, when available, the
per-stage label fixture (court corners / ball centres / player boxes).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from court_vision.evaluate import match_shots
from court_vision.player_detect import FrameTrackingResult
from court_vision.research.labels import LabelSet
from court_vision.scene_filter import GameplaySegment
from court_vision.shot_classify import MatchData, Point, point_winner

_SIDE = {"forehand": "forehand", "backhand": "backhand"}


@dataclass
class Scorecard:
    clip: str
    # points
    gt_points: int = 0
    pred_points: int = 0
    points_matched: int = 0
    point_precision: float = 0.0
    point_recall: float = 0.0
    # outcomes (over matched points with a GT winner)
    outcome_total: int = 0
    outcome_correct: int = 0
    outcome_accuracy: float = 0.0
    outcome_sources: dict = field(default_factory=dict)
    server_correct: int = 0
    server_total: int = 0
    # shots
    shot_tp: int = 0
    shot_fp: int = 0
    shot_fn: int = 0
    shot_precision: float = 0.0
    shot_recall: float = 0.0
    shot_f1: float = 0.0
    player_accuracy: float = 0.0
    stroke_accuracy: float = 0.0
    side_total: int = 0
    side_correct: int = 0
    side_accuracy: float = 0.0  # forehand vs backhand on GT FH/BH shots
    serve_precision: float = 0.0
    serve_recall: float = 0.0
    slice_precision: float = 0.0
    slice_recall: float = 0.0
    stroke_confusion: dict = field(default_factory=dict)  # gt -> pred -> count
    # stages (label-free)
    ball_coverage: float = 0.0  # fraction of gameplay frames with a real (non-interpolated) ball
    ball_coverage_any: float = 0.0
    players_both_rate: float = 0.0
    court_success_rate: float = 0.0
    # stages (vs labels)
    ball_detect_rate: float | None = None
    ball_loc_error_px: float | None = None
    ball_false_positive_rate: float | None = None
    player_count_accuracy: float | None = None
    player_mean_iou: float | None = None
    court_reproj_error_px: float | None = None
    # composite
    score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = [
            f"score={self.score:.3f}",
            f"points {self.points_matched}/{self.gt_points} (pred {self.pred_points})",
            f"winner {self.outcome_correct}/{self.outcome_total}",
            f"shots P={self.shot_precision:.2f} R={self.shot_recall:.2f} F1={self.shot_f1:.2f}",
            f"stroke={self.stroke_accuracy:.2f} side={self.side_correct}/{self.side_total}",
            f"serve P/R={self.serve_precision:.2f}/{self.serve_recall:.2f}",
            f"slice P/R={self.slice_precision:.2f}/{self.slice_recall:.2f}",
            f"ball cov={self.ball_coverage:.2f} players={self.players_both_rate:.2f}",
        ]
        return " | ".join(parts)


def _overlap(a: Point, b: Point) -> float:
    inter = min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame) + 1
    if inter <= 0:
        return 0.0
    union = max(a.end_frame, b.end_frame) - min(a.start_frame, b.start_frame) + 1
    return inter / union


def match_points(gt: MatchData, pred: MatchData, min_iou: float = 0.3) -> list[tuple[Point, Point]]:
    """Greedy one-to-one matching of predicted points to GT points by IoU."""
    pairs = []
    used = set()
    for g in gt.points:
        best, best_iou = None, min_iou
        for p in pred.points:
            if id(p) in used:
                continue
            iou = _overlap(g, p)
            if iou > best_iou:
                best, best_iou = p, iou
        if best is not None:
            used.add(id(best))
            pairs.append((g, best))
    return pairs


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_run(
    clip_name: str,
    gt: MatchData | None,
    pred: MatchData,
    tracking: list[FrameTrackingResult],
    segments: list[GameplaySegment],
    court_success: list[bool],
    fps: float,
    labels: LabelSet | None = None,
    court_homographies: dict[int, np.ndarray | None] | None = None,
) -> Scorecard:
    sc = Scorecard(clip=clip_name)

    # ── label-free stage health ──
    n_frames = len(tracking)
    if n_frames:
        sc.ball_coverage = sum(1 for t in tracking if t.ball is not None and not t.ball.interpolated) / n_frames
        sc.ball_coverage_any = sum(1 for t in tracking if t.ball is not None) / n_frames
        sc.players_both_rate = sum(1 for t in tracking if len(t.players) == 2) / n_frames
    sc.court_success_rate = (sum(court_success) / len(court_success)) if court_success else 0.0

    # ── stage labels ──
    if labels is not None:
        _score_stage_labels(sc, labels, tracking, court_homographies or {})

    if gt is None:
        sc.notes.append("no ground truth: only stage health scored")
        sc.score = _composite(sc, has_gt=False)
        return sc

    # ── points ──
    pairs = match_points(gt, pred)
    sc.gt_points = len(gt.points)
    sc.pred_points = len(pred.points)
    sc.points_matched = len(pairs)
    sc.point_precision = len(pairs) / len(pred.points) if pred.points else 0.0
    sc.point_recall = len(pairs) / len(gt.points) if gt.points else 0.0

    # ── outcomes / server ──
    for g, p in pairs:
        gw = point_winner(g)
        if gw is not None:
            sc.outcome_total += 1
            if p.winner == gw or (p.winner is None and point_winner(p) == gw):
                sc.outcome_correct += 1
            src = p.outcome_source or "none"
            sc.outcome_sources[src] = sc.outcome_sources.get(src, 0) + 1
        g_server = g.shots[0].player if g.shots else g.server
        if g_server:
            sc.server_total += 1
            if p.server == g_server:
                sc.server_correct += 1
    sc.outcome_accuracy = sc.outcome_correct / sc.outcome_total if sc.outcome_total else 0.0

    # ── shots ──
    tol = max(1, int(round(0.5 * fps)))
    ev = match_shots(gt, pred, frame_tolerance=tol)
    sc.shot_tp, sc.shot_fp, sc.shot_fn = ev.true_positives, ev.false_positives, ev.false_negatives
    sc.shot_precision, sc.shot_recall, sc.shot_f1 = ev.precision, ev.recall, ev.f1
    sc.player_accuracy = ev.player_accuracy
    sc.stroke_accuracy = ev.stroke_accuracy

    gt_shots = {s.frame: s for pt in gt.points for s in pt.shots}
    pred_shots = {s.frame: s for pt in pred.points for s in pt.shots}
    conf: dict[str, dict[str, int]] = {}
    for gf, pf in ev.matched_pairs:
        g, p = gt_shots[gf], pred_shots[pf]
        conf.setdefault(g.stroke, {})
        conf[g.stroke][p.stroke] = conf[g.stroke].get(p.stroke, 0) + 1
        if g.stroke in _SIDE:
            sc.side_total += 1
            if p.stroke == g.stroke:
                sc.side_correct += 1
    sc.stroke_confusion = conf
    sc.side_accuracy = sc.side_correct / sc.side_total if sc.side_total else 0.0

    def class_prf(name: str) -> tuple[float, float]:
        gt_n = sum(1 for s in gt_shots.values() if s.stroke == name)
        pred_n = sum(1 for s in pred_shots.values() if s.stroke == name)
        tp = sum(1 for gf, pf in ev.matched_pairs if gt_shots[gf].stroke == name and pred_shots[pf].stroke == name)
        p = tp / pred_n if pred_n else (1.0 if gt_n == 0 else 0.0)
        r = tp / gt_n if gt_n else (1.0 if pred_n == 0 else 0.0)
        return p, r

    sc.serve_precision, sc.serve_recall = class_prf("serve")
    sc.slice_precision, sc.slice_recall = class_prf("slice")

    sc.score = _composite(sc, has_gt=True)
    return sc


def _composite(sc: Scorecard, has_gt: bool) -> float:
    if not has_gt:
        return 0.6 * sc.ball_coverage + 0.3 * sc.players_both_rate + 0.1 * sc.court_success_rate
    point_f1 = _prf(sc.points_matched, sc.pred_points - sc.points_matched, sc.gt_points - sc.points_matched)[2]
    serve_f1 = _prf_from_pr(sc.serve_precision, sc.serve_recall)
    slice_f1 = _prf_from_pr(sc.slice_precision, sc.slice_recall)
    terms = [
        (0.15, point_f1),
        (0.20, sc.outcome_accuracy),
        (0.25, sc.shot_f1),
        (0.15, sc.side_accuracy),
        (0.05, serve_f1),
        (0.05, slice_f1),
        (0.10, sc.ball_coverage),
        (0.05, sc.players_both_rate),
    ]
    return sum(w * v for w, v in terms)


def _prf_from_pr(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _score_stage_labels(
    sc: Scorecard,
    labels: LabelSet,
    tracking: list[FrameTrackingResult],
    court_homographies: dict[int, np.ndarray | None],
    match_radius_px: float = 40.0,
) -> None:
    from court_vision.court_detect import COURT_KEYPOINTS, _compute_reprojection_error

    by_frame = {t.frame_index: t for t in tracking}

    visible = [b for b in labels.ball if b.visible and b.x is not None]
    not_visible = [b for b in labels.ball if not b.visible]
    if visible or not_visible:
        matched, errs = 0, []
        for bl in visible:
            t = by_frame.get(bl.frame)
            if t is None or t.ball is None or t.ball.interpolated:
                continue
            d = math.hypot(t.ball.x - bl.x, t.ball.y - bl.y)
            if d <= match_radius_px:
                matched += 1
                errs.append(d)
        fp = sum(1 for bl in not_visible if (t := by_frame.get(bl.frame)) and t.ball is not None and not t.ball.interpolated)
        sc.ball_detect_rate = matched / len(visible) if visible else None
        sc.ball_loc_error_px = float(np.mean(errs)) if errs else None
        sc.ball_false_positive_rate = fp / len(not_visible) if not_visible else None

    if labels.players:
        count_ok, ious = 0, []
        for pl in labels.players:
            t = by_frame.get(pl.frame)
            pred_boxes = [p.bbox for p in t.players] if t else []
            if len(pred_boxes) == len(pl.boxes):
                count_ok += 1
            for g in pl.boxes:
                ious.append(max((_iou(g.bbox, pb) for pb in pred_boxes), default=0.0))
        sc.player_count_accuracy = count_ok / len(labels.players)
        sc.player_mean_iou = float(np.mean(ious)) if ious else 0.0

    if labels.court:
        court_pts = np.array([
            COURT_KEYPOINTS["baseline_near_left_doubles"], COURT_KEYPOINTS["baseline_near_right_doubles"],
            COURT_KEYPOINTS["baseline_far_right_doubles"], COURT_KEYPOINTS["baseline_far_left_doubles"],
        ], dtype=np.float64)
        errs = []
        for cl in labels.court:
            ordered = cl.ordered_pixels()
            H = court_homographies.get(cl.frame)
            if ordered is None or H is None:
                continue
            errs.append(_compute_reprojection_error(np.array(ordered, dtype=np.float64), court_pts, H))
        sc.court_reproj_error_px = float(np.mean(errs)) if errs else None


def load_scorecard(path: Path) -> Scorecard:
    import json
    return Scorecard(**json.loads(Path(path).read_text()))
