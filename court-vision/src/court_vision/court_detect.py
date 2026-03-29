"""Court detection — line detection, keypoint extraction, homography."""

# Standard tennis court dimensions in meters.
# Origin (0, 0) = center of net.
# X-axis: parallel to net, positive = right when facing far end.
# Y-axis: perpendicular to net, positive = far side.

_SINGLES_WIDTH_HALF = 4.115  # Singles sideline to center
_DOUBLES_WIDTH_HALF = 5.485  # Doubles sideline to center
_BASELINE_DIST = 11.885  # Baseline to net
_SERVICE_LINE_DIST = 6.4  # Service line to net

COURT_KEYPOINTS: dict[str, tuple[float, float]] = {
    # Far baseline (y = +11.885)
    "baseline_far_left_doubles": (-_DOUBLES_WIDTH_HALF, _BASELINE_DIST),
    "baseline_far_left_singles": (-_SINGLES_WIDTH_HALF, _BASELINE_DIST),
    "baseline_far_center": (0.0, _BASELINE_DIST),
    "baseline_far_right_singles": (_SINGLES_WIDTH_HALF, _BASELINE_DIST),
    "baseline_far_right_doubles": (_DOUBLES_WIDTH_HALF, _BASELINE_DIST),
    # Far service line (y = +6.4)
    "service_far_left": (-_SINGLES_WIDTH_HALF, _SERVICE_LINE_DIST),
    "service_far_center": (0.0, _SERVICE_LINE_DIST),
    "service_far_right": (_SINGLES_WIDTH_HALF, _SERVICE_LINE_DIST),
    # Net (y = 0)
    "net_left_singles": (-_SINGLES_WIDTH_HALF, 0.0),
    "net_center": (0.0, 0.0),
    "net_right_singles": (_SINGLES_WIDTH_HALF, 0.0),
    "net_left_doubles": (-_DOUBLES_WIDTH_HALF, 0.0),
    "net_right_doubles": (_DOUBLES_WIDTH_HALF, 0.0),
    # Near service line (y = -6.4)
    "service_near_left": (-_SINGLES_WIDTH_HALF, -_SERVICE_LINE_DIST),
    "service_near_center": (0.0, -_SERVICE_LINE_DIST),
    "service_near_right": (_SINGLES_WIDTH_HALF, -_SERVICE_LINE_DIST),
    # Near baseline (y = -11.885)
    "baseline_near_left_doubles": (-_DOUBLES_WIDTH_HALF, -_BASELINE_DIST),
    "baseline_near_left_singles": (-_SINGLES_WIDTH_HALF, -_BASELINE_DIST),
    "baseline_near_center": (0.0, -_BASELINE_DIST),
    "baseline_near_right_singles": (_SINGLES_WIDTH_HALF, -_BASELINE_DIST),
    "baseline_near_right_doubles": (_DOUBLES_WIDTH_HALF, -_BASELINE_DIST),
}


import cv2
import numpy as np


def detect_court_lines(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    min_line_length: int = 100,
    max_line_gap: int = 30,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Detect court lines in a frame using Hough line transform.

    Applies white-pixel masking (court lines are white), Canny edge
    detection, and probabilistic Hough transform.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        canny_low: Lower Canny edge threshold.
        canny_high: Upper Canny edge threshold.
        hough_threshold: Hough accumulator threshold.
        min_line_length: Minimum line length in pixels.
        max_line_gap: Maximum gap between line segments to merge.

    Returns:
        List of line segments as ((x1, y1), (x2, y2)) tuples.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_or(white_mask, bright_mask)
    edges = cv2.Canny(combined, canny_low, canny_high)
    raw_lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
    )
    if raw_lines is None:
        return []
    lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        lines.append(((int(x1), int(y1)), (int(x2), int(y2))))
    return lines


def classify_lines(
    lines: list[tuple[tuple[int, int], tuple[int, int]]],
    angle_threshold: float = 30.0,
) -> tuple[
    list[tuple[tuple[int, int], tuple[int, int]]],
    list[tuple[tuple[int, int], tuple[int, int]]],
]:
    """Classify detected lines as horizontal or vertical.

    Lines within `angle_threshold` degrees of horizontal (0°) are
    classified as horizontal. Lines within `angle_threshold` degrees
    of vertical (90°) are classified as vertical. Diagonal lines
    (between the two thresholds) are discarded.

    Args:
        lines: List of line segments as ((x1, y1), (x2, y2)).
        angle_threshold: Maximum deviation from axis in degrees.

    Returns:
        Tuple of (horizontal_lines, vertical_lines).
    """
    horizontal = []
    vertical = []

    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))

        # Normalize to 0-90 range
        if angle > 90:
            angle = 180 - angle

        if angle <= angle_threshold:
            horizontal.append(((x1, y1), (x2, y2)))
        elif angle >= (90 - angle_threshold):
            vertical.append(((x1, y1), (x2, y2)))
        # else: diagonal — discard

    return horizontal, vertical
