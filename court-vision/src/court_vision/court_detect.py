"""Court detection — line detection, keypoint extraction, homography."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from court_vision.scene_filter import GameplaySegment

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


def find_line_intersection(
    line1: tuple[tuple[int, int], tuple[int, int]],
    line2: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[float, float] | None:
    """Find the intersection point of two line segments (extended to infinite lines).

    Uses the cross-product method for line-line intersection.

    Args:
        line1: First line as ((x1, y1), (x2, y2)).
        line2: Second line as ((x3, y3), (x4, y4)).

    Returns:
        (x, y) intersection point, or None if lines are parallel.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # parallel or coincident

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)

    return (float(ix), float(iy))


def extract_keypoints(
    horizontal_lines: list[tuple[tuple[int, int], tuple[int, int]]],
    vertical_lines: list[tuple[tuple[int, int], tuple[int, int]]],
    frame_width: int = 1280,
    frame_height: int = 720,
) -> list[tuple[float, float]]:
    """Extract court keypoints as intersections of horizontal and vertical lines.

    Computes all pairwise intersections between horizontal and vertical
    lines and filters to points within the frame bounds.

    Args:
        horizontal_lines: Detected horizontal court lines.
        vertical_lines: Detected vertical court lines.
        frame_width: Image width for bounds checking.
        frame_height: Image height for bounds checking.

    Returns:
        List of (x, y) pixel coordinates for detected keypoints.
    """
    keypoints: list[tuple[float, float]] = []

    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            point = find_line_intersection(h_line, v_line)
            if point is None:
                continue

            x, y = point
            # Keep only points within frame bounds (with small margin)
            margin = 50
            if -margin <= x <= frame_width + margin and -margin <= y <= frame_height + margin:
                keypoints.append(point)

    return keypoints


def compute_homography(
    pixel_points: np.ndarray,
    court_points: np.ndarray,
) -> np.ndarray | None:
    """Compute the homography matrix from pixel to court coordinates.

    Requires at least 4 point correspondences.

    Args:
        pixel_points: Array of shape (N, 2) — pixel (x, y) coordinates.
        court_points: Array of shape (N, 2) — court (x, y) coordinates in meters.

    Returns:
        3x3 homography matrix, or None if computation fails.
    """
    if len(pixel_points) < 4 or len(court_points) < 4:
        return None

    H, mask = cv2.findHomography(
        pixel_points, court_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    if H is None:
        return None

    return H


def _compute_reprojection_error(
    pixel_pts: np.ndarray,
    court_pts: np.ndarray,
    H: np.ndarray,
) -> float:
    """Compute mean reprojection error for a homography.

    Projects pixel_pts through H and compares to expected court_pts.

    Args:
        pixel_pts: Array of shape (N, 2) — pixel coordinates.
        court_pts: Array of shape (N, 2) — expected court coordinates.
        H: 3x3 homography matrix.

    Returns:
        Mean Euclidean distance between projected and expected points.
    """
    n = len(pixel_pts)
    if n == 0:
        return 0.0

    # Convert to homogeneous coordinates
    ones = np.ones((n, 1), dtype=np.float64)
    pixel_h = np.hstack([pixel_pts, ones])  # (N, 3)

    # Project through homography
    projected_h = (H @ pixel_h.T).T  # (N, 3)

    # Convert from homogeneous
    w = projected_h[:, 2:3]
    w = np.where(np.abs(w) < 1e-10, 1.0, w)
    projected = projected_h[:, :2] / w

    # Compute mean Euclidean distance
    errors = np.sqrt(np.sum((projected - court_pts) ** 2, axis=1))
    return float(np.mean(errors))


def pixel_to_court(
    pixel_point: np.ndarray,
    homography: np.ndarray,
) -> tuple[float, float]:
    """Transform a pixel coordinate to court coordinates using a homography.

    Args:
        pixel_point: (x, y) pixel coordinate as numpy array.
        homography: 3x3 homography matrix from compute_homography.

    Returns:
        (x, y) court coordinate in meters.
    """
    # Convert to homogeneous coordinates
    px = np.array([pixel_point[0], pixel_point[1], 1.0], dtype=np.float64)

    # Apply homography
    transformed = homography @ px

    # Convert back from homogeneous
    w = transformed[2]
    if abs(w) < 1e-10:
        return (0.0, 0.0)

    return (float(transformed[0] / w), float(transformed[1] / w))


def match_keypoints_to_court(
    pixel_keypoints: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Match detected pixel keypoints to canonical court coordinates.

    Uses geometric ordering: sorts keypoints by y-coordinate to separate
    "near" (bottom of image, large y) from "far" (top of image, small y),
    then by x-coordinate within each row to get left-to-right ordering.

    Matches the 4 outermost corners to singles court corners.

    Args:
        pixel_keypoints: List of (x, y) pixel coordinates from keypoint extraction.

    Returns:
        Tuple of (pixel_points, court_points) as (N, 2) arrays, or None
        if fewer than 4 keypoints are available.
    """
    if len(pixel_keypoints) < 4:
        return None

    points = np.array(pixel_keypoints, dtype=np.float64)

    # Sort by y-coordinate: top of image (far court) has small y
    sorted_by_y = points[points[:, 1].argsort()]

    # Split into "far" (top half) and "near" (bottom half)
    mid = len(sorted_by_y) // 2
    far_points = sorted_by_y[:mid]
    near_points = sorted_by_y[mid:]

    # Sort each group by x-coordinate (left to right)
    far_sorted = far_points[far_points[:, 0].argsort()]
    near_sorted = near_points[near_points[:, 0].argsort()]

    # Take outermost corners: far-left, far-right, near-left, near-right
    far_left = far_sorted[0]
    far_right = far_sorted[-1]
    near_left = near_sorted[0]
    near_right = near_sorted[-1]

    pixel_pts = np.array([near_left, near_right, far_right, far_left], dtype=np.float64)

    # Map to singles court corners
    court_pts = np.array([
        [COURT_KEYPOINTS["baseline_near_left_singles"][0],
         COURT_KEYPOINTS["baseline_near_left_singles"][1]],
        [COURT_KEYPOINTS["baseline_near_right_singles"][0],
         COURT_KEYPOINTS["baseline_near_right_singles"][1]],
        [COURT_KEYPOINTS["baseline_far_right_singles"][0],
         COURT_KEYPOINTS["baseline_far_right_singles"][1]],
        [COURT_KEYPOINTS["baseline_far_left_singles"][0],
         COURT_KEYPOINTS["baseline_far_left_singles"][1]],
    ], dtype=np.float64)

    return pixel_pts, court_pts


@dataclass
class CourtDetectionResult:
    """Result of court detection on a single frame."""

    success: bool
    homography: np.ndarray | None = None
    pixel_keypoints: list[tuple[float, float]] | None = None
    num_lines_detected: int = 0


def detect_court(frame: np.ndarray) -> CourtDetectionResult:
    """Detect the tennis court in a frame and compute the homography.

    Full pipeline: detect lines -> classify -> extract keypoints ->
    match to court -> compute homography.

    Args:
        frame: BGR image as numpy array (H, W, 3).

    Returns:
        CourtDetectionResult with homography if successful.
    """
    # Step 1: Detect lines
    lines = detect_court_lines(frame)
    if not lines:
        return CourtDetectionResult(success=False, num_lines_detected=0)

    # Step 2: Classify into horizontal and vertical
    horizontal, vertical = classify_lines(lines)
    if len(horizontal) < 2 or len(vertical) < 2:
        return CourtDetectionResult(success=False, num_lines_detected=len(lines))

    # Step 3: Extract keypoints from intersections
    h, w = frame.shape[:2]
    keypoints = extract_keypoints(horizontal, vertical, frame_width=w, frame_height=h)
    if len(keypoints) < 4:
        return CourtDetectionResult(success=False, num_lines_detected=len(lines))

    # Step 4: Match pixel keypoints to court coordinates
    match_result = match_keypoints_to_court(keypoints)
    if match_result is None:
        return CourtDetectionResult(
            success=False,
            pixel_keypoints=keypoints,
            num_lines_detected=len(lines),
        )

    pixel_pts, court_pts = match_result

    # Step 5: Compute homography
    H = compute_homography(pixel_pts, court_pts)
    if H is None:
        return CourtDetectionResult(
            success=False,
            pixel_keypoints=keypoints,
            num_lines_detected=len(lines),
        )

    # Step 6: Validate reprojection error
    reproj_error = _compute_reprojection_error(pixel_pts, court_pts, H)
    if reproj_error > 10.0:
        return CourtDetectionResult(
            success=False,
            pixel_keypoints=keypoints,
            num_lines_detected=len(lines),
        )

    return CourtDetectionResult(
        success=True,
        homography=H,
        pixel_keypoints=keypoints,
        num_lines_detected=len(lines),
    )


def compute_segment_homographies(
    frames_dir: Path,
    segments: list[GameplaySegment],
) -> list[CourtDetectionResult]:
    """Compute a homography for each gameplay segment.

    Samples up to 3 frames per segment (25%, 50%, 75%) and uses the
    first successful detection.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        segments: List of gameplay segments from scene filter.

    Returns:
        List of CourtDetectionResult, one per segment.
    """
    results: list[CourtDetectionResult] = []

    for segment in segments:
        seg_len = segment.end_frame - segment.start_frame
        # Sample at 25%, 50%, 75% of the segment
        sample_offsets = [0.25, 0.50, 0.75]
        sample_frames = [
            segment.start_frame + int(seg_len * offset)
            for offset in sample_offsets
        ]
        # Deduplicate (short segments may repeat)
        sample_frames = list(dict.fromkeys(sample_frames))

        best_result = CourtDetectionResult(success=False, num_lines_detected=0)

        for frame_idx in sample_frames:
            frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            result = detect_court(frame)
            if result.success:
                best_result = result
                break  # Use first successful detection
            elif result.num_lines_detected > best_result.num_lines_detected:
                best_result = result

        results.append(best_result)

    return results
