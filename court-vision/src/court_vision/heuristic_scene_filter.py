"""Heuristic-based scene filter using court color and line detection."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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
