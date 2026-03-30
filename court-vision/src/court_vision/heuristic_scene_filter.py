"""Heuristic-based scene filter using court color and line detection."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from court_vision.court_detect import classify_lines, detect_court_lines
from court_vision.scene_filter import SceneCategory, SceneFilterResult


def _get_court_color_mask(hsv: np.ndarray) -> np.ndarray:
    """Create a binary mask of pixels matching tennis court colors."""
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    blue_mask = cv2.inRange(hsv, (90, 50, 40), (130, 255, 255))
    clay_mask = cv2.inRange(hsv, (10, 100, 100), (25, 255, 255))
    return cv2.bitwise_or(green_mask, cv2.bitwise_or(blue_mask, clay_mask))


def compute_court_color_ratio(frame: np.ndarray) -> float:
    """Fraction of pixels matching known tennis court color ranges."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _get_court_color_mask(hsv)
    return float(np.count_nonzero(mask) / (frame.shape[0] * frame.shape[1]))


def compute_line_score(frame: np.ndarray) -> float:
    """Score based on detected white lines consistent with court geometry.

    Reuses detect_court_lines from court_detect with relaxed thresholds
    for higher recall at the scene filtering stage.
    """
    lines = detect_court_lines(
        frame,
        hough_threshold=60,
        min_line_length=80,
        max_line_gap=40,
    )
    if not lines:
        return 0.0

    horizontal, vertical = classify_lines(lines, angle_threshold=30.0)

    line_density = min(len(lines) / 10.0, 1.0)
    grid_bonus = 0.2 if len(horizontal) >= 1 and len(vertical) >= 1 else 0.0

    return min(line_density + grid_bonus, 1.0)


def compute_court_spatial_score(frame: np.ndarray) -> float:
    """Score based on court color being concentrated in the lower portion."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    court_mask = _get_court_color_mask(hsv)

    h = frame.shape[0]
    upper_third = court_mask[:h // 3, :]
    lower_two_thirds = court_mask[h // 3:, :]

    upper_ratio = np.count_nonzero(upper_third) / upper_third.size if upper_third.size > 0 else 0
    lower_ratio = np.count_nonzero(lower_two_thirds) / lower_two_thirds.size if lower_two_thirds.size > 0 else 0

    if lower_ratio > 0.15 and lower_ratio > upper_ratio * 1.5:
        return 1.0
    elif lower_ratio > 0.10:
        return 0.5
    return 0.0


@dataclass
class HeuristicWeights:
    """Weights for combining heuristic signals."""
    court_color: float = 0.40
    line_detection: float = 0.35
    spatial_distribution: float = 0.25


@dataclass
class HeuristicScores:
    """Individual heuristic scores for a single frame."""
    court_color_ratio: float
    line_score: float
    spatial_score: float
    composite: float


def compute_gameplay_score(
    frame: np.ndarray,
    weights: HeuristicWeights | None = None,
) -> HeuristicScores:
    """Compute composite gameplay likelihood score for a single frame."""
    if weights is None:
        weights = HeuristicWeights()

    color_ratio = compute_court_color_ratio(frame)
    line_score = compute_line_score(frame)
    spatial_score = compute_court_spatial_score(frame)

    color_score = min(color_ratio / 0.40, 1.0)

    composite = (
        weights.court_color * color_score
        + weights.line_detection * line_score
        + weights.spatial_distribution * spatial_score
    )

    return HeuristicScores(
        court_color_ratio=color_ratio,
        line_score=line_score,
        spatial_score=spatial_score,
        composite=composite,
    )


def classify_frame_heuristic(
    frame: np.ndarray,
    frame_index: int = 0,
    gameplay_threshold: float = 0.45,
    weights: HeuristicWeights | None = None,
) -> SceneFilterResult:
    """Classify a single frame using heuristic signals."""
    scores = compute_gameplay_score(frame, weights)

    if scores.composite >= gameplay_threshold:
        category = SceneCategory.GAMEPLAY
        confidence = scores.composite
    else:
        category = SceneCategory.TRANSITION
        confidence = 1.0 - scores.composite

    return SceneFilterResult(
        frame_index=frame_index,
        category=category,
        confidence=confidence,
    )


def classify_frames_heuristic(
    frames_dir: Path,
    total_frames: int,
    gameplay_threshold: float = 0.45,
    weights: HeuristicWeights | None = None,
) -> list[SceneFilterResult]:
    """Classify all frames in a directory using heuristics.

    Drop-in replacement for scene_filter.classify_frames().
    """
    results: list[SceneFilterResult] = []
    for i in range(total_frames):
        frame_path = frames_dir / f"frame_{i:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        result = classify_frame_heuristic(
            frame, frame_index=i,
            gameplay_threshold=gameplay_threshold, weights=weights,
        )
        results.append(result)
    return results
