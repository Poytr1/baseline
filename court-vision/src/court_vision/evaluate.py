"""Evaluate pipeline shot detection against human-corrected ground truth."""

from dataclasses import dataclass

from court_vision.shot_classify import MatchData


@dataclass
class EvaluationResult:
    """Metrics from comparing pipeline output to ground truth."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    stroke_correct: int
    stroke_total: int
    stroke_accuracy: float
    player_correct: int
    player_total: int
    player_accuracy: float
    matched_pairs: list[tuple[int, int]]  # (gt_frame, pred_frame) pairs


def match_shots(
    ground_truth: MatchData,
    predicted: MatchData,
    frame_tolerance: int = 15,
) -> EvaluationResult:
    """Match predicted shots to ground truth and compute metrics.

    Uses greedy matching: for each ground truth shot, find the closest
    predicted shot within frame_tolerance. Each predicted shot can match
    at most one ground truth shot.

    Args:
        ground_truth: Human-corrected match data.
        predicted: Pipeline-produced match data.
        frame_tolerance: Max frame distance for a match (default 15 = 0.5s at 30fps).

    Returns:
        EvaluationResult with precision, recall, F1, and accuracy metrics.
    """
    gt_shots = []
    for point in ground_truth.points:
        gt_shots.extend(point.shots)

    pred_shots = []
    for point in predicted.points:
        pred_shots.extend(point.shots)

    used_pred = set()
    matched_pairs = []
    stroke_correct = 0
    player_correct = 0

    for gt_shot in gt_shots:
        best_idx = None
        best_dist = frame_tolerance + 1

        for i, pred_shot in enumerate(pred_shots):
            if i in used_pred:
                continue
            dist = abs(gt_shot.frame - pred_shot.frame)
            if dist <= frame_tolerance and dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            used_pred.add(best_idx)
            matched_pairs.append((gt_shot.frame, pred_shots[best_idx].frame))
            if gt_shot.stroke == pred_shots[best_idx].stroke:
                stroke_correct += 1
            if gt_shot.player == pred_shots[best_idx].player:
                player_correct += 1

    tp = len(matched_pairs)
    fp = len(pred_shots) - tp
    fn = len(gt_shots) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvaluationResult(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        stroke_correct=stroke_correct,
        stroke_total=tp,
        stroke_accuracy=stroke_correct / tp if tp > 0 else 0.0,
        player_correct=player_correct,
        player_total=tp,
        player_accuracy=player_correct / tp if tp > 0 else 0.0,
        matched_pairs=matched_pairs,
    )
