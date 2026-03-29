"""Ball tracking — detection, trajectory building, and court coordinate mapping."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BallDetection:
    """Single-frame ball detection result."""

    frame_index: int
    x: float
    y: float
    confidence: float
    interpolated: bool = False


def detect_ball_in_frame(
    frame: np.ndarray,
    frame_index: int = 0,
    min_radius: int = 3,
    max_radius: int = 20,
) -> BallDetection | None:
    """Detect the tennis ball in a single frame using color + shape filtering.

    Looks for small, bright, circular objects (white or yellow) that match
    typical tennis ball appearance in broadcast video.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        frame_index: Index of this frame in the video sequence.
        min_radius: Minimum ball radius in pixels.
        max_radius: Maximum ball radius in pixels.

    Returns:
        BallDetection with pixel coordinates, or None if no ball found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask 1: bright yellow (tennis ball)
    yellow_mask = cv2.inRange(hsv, (20, 80, 180), (40, 255, 255))

    # Mask 2: bright white (can appear white under broadcast lighting)
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))

    combined = cv2.bitwise_or(yellow_mask, white_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_detection: BallDetection | None = None
    best_score = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < np.pi * min_radius**2 or area > np.pi * max_radius**2:
            continue

        # Check circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue

        # Get center via minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        if radius < min_radius or radius > max_radius:
            continue

        # Score: higher circularity and smaller size (balls are small) score better
        score = circularity * (1.0 / (1.0 + radius / max_radius))

        if score > best_score:
            best_score = score
            best_detection = BallDetection(
                frame_index=frame_index,
                x=float(cx),
                y=float(cy),
                confidence=float(min(circularity, 1.0)),
            )

    return best_detection
