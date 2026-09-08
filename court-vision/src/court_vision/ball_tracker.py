"""Ball tracking — detection, trajectory building, and court coordinate mapping."""

from collections.abc import Callable
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


def _find_ball_candidates(
    mask: np.ndarray,
    frame_index: int,
    min_radius: int,
    max_radius: int,
) -> list[tuple[BallDetection, float]]:
    """Find ball candidates from a binary mask.

    Returns list of (BallDetection, score) tuples sorted by score descending.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[BallDetection, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < np.pi * min_radius**2 or area > np.pi * max_radius**2:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or radius > max_radius:
            continue

        score = circularity * (1.0 / (1.0 + radius / max_radius))
        det = BallDetection(
            frame_index=frame_index,
            x=float(cx),
            y=float(cy),
            confidence=float(min(circularity, 1.0)),
        )
        candidates.append((det, score))

    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates


def _on_court_surface(hsv: np.ndarray, cx: float, cy: float, radius: float = 10.0) -> bool:
    """Check if a point sits on court-colored pixels (green, blue, or clay)."""
    h, w = hsv.shape[:2]
    r = int(radius)
    x0 = max(0, int(cx) - r)
    y0 = max(0, int(cy) - r)
    x1 = min(w, int(cx) + r)
    y1 = min(h, int(cy) + r)
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return False

    green = cv2.inRange(patch, (35, 40, 40), (85, 255, 255))
    blue = cv2.inRange(patch, (90, 50, 40), (130, 255, 255))
    red_clay = cv2.inRange(patch, (0, 50, 50), (10, 255, 255))
    orange_clay = cv2.inRange(patch, (10, 50, 50), (25, 255, 255))
    court = cv2.bitwise_or(green, cv2.bitwise_or(blue, cv2.bitwise_or(red_clay, orange_clay)))
    ratio = np.count_nonzero(court) / court.size
    return ratio > 0.4


def detect_ball_in_frame(
    frame: np.ndarray,
    frame_index: int = 0,
    min_radius: int = 3,
    max_radius: int = 20,
    min_confidence: float = 0.0,
) -> BallDetection | None:
    """Detect the tennis ball in a single frame using color + shape filtering.

    Uses a two-stage approach: first tries to detect via yellow/green ball
    color (high precision), then falls back to white detection only if no
    yellow candidate is found. White candidates sitting on court-colored
    surfaces (court lines) are rejected.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        frame_index: Index of this frame in the video sequence.
        min_radius: Minimum ball radius in pixels.
        max_radius: Maximum ball radius in pixels.
        min_confidence: Minimum confidence threshold for accepted detections.

    Returns:
        BallDetection with pixel coordinates, or None if no ball found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Stage 1: yellow/green ball color (high precision)
    yellow_mask = cv2.inRange(hsv, (20, 60, 150), (45, 255, 255))
    yellow_candidates = _find_ball_candidates(yellow_mask, frame_index, min_radius, max_radius)

    if yellow_candidates:
        best = yellow_candidates[0][0]
        if best.confidence >= min_confidence:
            return best

    # Stage 2: white fallback — tighter mask, reject candidates on court surface
    white_mask = cv2.inRange(hsv, (0, 0, 220), (180, 40, 255))
    white_candidates = _find_ball_candidates(white_mask, frame_index, min_radius, max_radius)

    for det, _score in white_candidates:
        if _on_court_surface(hsv, det.x, det.y):
            continue
        if det.confidence >= min_confidence:
            return det

    return None


@dataclass
class BallTrajectory:
    """Sequence of ball detections across frames."""

    detections: list[BallDetection]
    fps: float


def detect_ball_tracknet(
    frames: "list[np.ndarray]",
    frame_index: int,
    **kwargs,
) -> "BallDetection | None":
    """Thin wrapper that lazily imports and delegates to tracknet.detect_ball_tracknet.

    This wrapper exists so that tests can patch
    ``court_vision.ball_tracker.detect_ball_tracknet`` without triggering the
    circular import that would result from a top-level import of tracknet.

    Args:
        frames: Exactly 3 BGR frames (any resolution).
        frame_index: Frame index for the detection result.
        **kwargs: Forwarded to the real implementation.

    Returns:
        BallDetection with pixel coordinates, or None.
    """
    from court_vision.tracknet import detect_ball_tracknet as _impl
    return _impl(frames, frame_index, **kwargs)


def detect_ball_wasb(
    frames: "list[np.ndarray]",
    frame_index: int,
    **kwargs,
) -> "BallDetection | None":
    """Thin wrapper that lazily imports and delegates to wasb.detect_ball_wasb.

    Mirrors :func:`detect_ball_tracknet` so tests can patch this symbol.
    """
    from court_vision.wasb import detect_ball_wasb as _impl
    return _impl(frames, frame_index, **kwargs)


def build_trajectory(
    frames_dir: "Path",
    start_frame: int,
    end_frame: int,
    fps: float,
    max_gap_s: float = 0.5,
    method: str = "tracknet",
    progress_callback: "Callable[[int, int], None] | None" = None,
) -> BallTrajectory:
    """Build a ball trajectory by detecting the ball in each frame.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        start_frame: First frame index to process.
        end_frame: Last frame index to process (inclusive).
        fps: Video frame rate.
        max_gap_s: Maximum gap in seconds to interpolate through.
        method: Detection method — "wasb" (HRNet heatmap, recommended),
                "tracknet" (3-frame sliding window), or "hsv" (per-frame
                color+contour).

    Returns:
        BallTrajectory with detections and interpolated positions.
    """
    from pathlib import Path

    frames_dir = Path(frames_dir)
    raw_detections: list[BallDetection] = []

    if method in ("tracknet", "wasb"):
        detect_fn = detect_ball_wasb if method == "wasb" else detect_ball_tracknet
        # Read all frames into memory for sliding window
        frames_cache: dict[int, np.ndarray] = {}
        for i in range(start_frame, end_frame + 1):
            frame_path = frames_dir / f"frame_{i:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                frames_cache[i] = frame

        for i in range(start_frame, end_frame + 1):
            if i not in frames_cache:
                continue

            # Build 3-frame buffer: [i-2, i-1, i]
            current = frames_cache[i]
            h, w = current.shape[:2]
            black = np.zeros((h, w, 3), dtype=np.uint8)

            frame_minus2 = frames_cache.get(i - 2, black) if i - 2 >= start_frame else black
            frame_minus1 = frames_cache.get(i - 1, black) if i - 1 >= start_frame else black

            buffer = [frame_minus2, frame_minus1, current]
            det = detect_fn(buffer, frame_index=i)
            if det is not None:
                raw_detections.append(det)
            if progress_callback:
                progress_callback(i - start_frame + 1, end_frame - start_frame + 1)
    else:
        # HSV method: per-frame detection
        for i in range(start_frame, end_frame + 1):
            frame_path = frames_dir / f"frame_{i:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            det = detect_ball_in_frame(frame, frame_index=i)
            if det is not None:
                raw_detections.append(det)
            if progress_callback:
                progress_callback(i - start_frame + 1, end_frame - start_frame + 1)

    filtered = reject_velocity_outliers(raw_detections, fps)
    interpolated = interpolate_gaps(filtered, fps, max_gap_s)
    smoothed = smooth_trajectory(interpolated, window=3)

    return BallTrajectory(detections=smoothed, fps=fps)


def reject_velocity_outliers(
    detections: list[BallDetection],
    fps: float,
    max_speed_px_per_frame: float = 150.0,
) -> list[BallDetection]:
    """Reject detections that imply physically impossible ball movement.

    Walks through consecutive detections and drops any whose distance
    from both the previous and next accepted detection exceeds
    max_speed_px_per_frame (scaled by the frame gap). This eliminates
    random false positives that jump across the screen.

    Args:
        detections: Sorted raw ball detections (no Nones).
        fps: Video frame rate.
        max_speed_px_per_frame: Maximum allowed displacement per frame in pixels.

    Returns:
        Filtered list of detections with outliers removed.
    """
    if len(detections) < 2:
        return list(detections)

    kept: list[BallDetection] = [detections[0]]

    for det in detections[1:]:
        prev = kept[-1]
        frame_gap = max(det.frame_index - prev.frame_index, 1)
        dist = ((det.x - prev.x) ** 2 + (det.y - prev.y) ** 2) ** 0.5
        if dist <= max_speed_px_per_frame * frame_gap:
            kept.append(det)

    return kept


def interpolate_gaps(
    detections: list[BallDetection],
    fps: float,
    max_gap_s: float = 0.5,
) -> list[BallDetection]:
    """Fill short gaps in ball detections with locally-fit interpolation.

    For each gap, fits a quadratic polynomial to nearby anchor points
    (up to 2 detections on each side of the gap). Falls back to linear
    when fewer than 3 local anchors are available.

    Args:
        detections: Sorted list of ball detections (by frame_index).
        fps: Video frame rate.
        max_gap_s: Maximum gap duration in seconds to interpolate.

    Returns:
        Detections with interpolated positions filling short gaps.
    """
    if len(detections) <= 1:
        return list(detections)

    max_gap_frames = int(max_gap_s * fps)
    result: list[BallDetection] = [detections[0]]

    for i in range(1, len(detections)):
        prev = detections[i - 1]
        curr = detections[i]
        gap = curr.frame_index - prev.frame_index

        if 1 < gap <= max_gap_frames:
            # Gather local anchors: up to 2 before gap, up to 2 after
            before = detections[max(0, i - 2):i]
            after = detections[i:min(len(detections), i + 2)]
            local_anchors = before + after

            anchor_frames = np.array([d.frame_index for d in local_anchors], dtype=np.float64)
            anchor_x = np.array([d.x for d in local_anchors], dtype=np.float64)
            anchor_y = np.array([d.y for d in local_anchors], dtype=np.float64)

            degree = min(2, len(local_anchors) - 1)
            poly_x = np.polyfit(anchor_frames, anchor_x, degree)
            poly_y = np.polyfit(anchor_frames, anchor_y, degree)

            for j in range(1, gap):
                frame_idx = prev.frame_index + j
                interp_x = float(np.polyval(poly_x, frame_idx))
                interp_y = float(np.polyval(poly_y, frame_idx))
                interp_conf = min(prev.confidence, curr.confidence) * 0.5
                result.append(BallDetection(
                    frame_index=frame_idx,
                    x=interp_x,
                    y=interp_y,
                    confidence=interp_conf,
                    interpolated=True,
                ))

        result.append(curr)

    return result


def smooth_trajectory(
    detections: list[BallDetection],
    window: int = 3,
) -> list[BallDetection]:
    """Apply moving-average smoothing to ball positions.

    Smooths x and y coordinates independently using a centered
    moving average. Preserves frame_index, confidence, and
    interpolated flag. Edge detections use smaller windows.

    Args:
        detections: Sorted list of ball detections.
        window: Smoothing window size (must be odd).

    Returns:
        New list of smoothed detections.
    """
    if len(detections) < window:
        return list(detections)

    half = window // 2
    smoothed: list[BallDetection] = []

    for i, det in enumerate(detections):
        start = max(0, i - half)
        end = min(len(detections), i + half + 1)
        neighbors = detections[start:end]

        avg_x = sum(d.x for d in neighbors) / len(neighbors)
        avg_y = sum(d.y for d in neighbors) / len(neighbors)

        smoothed.append(BallDetection(
            frame_index=det.frame_index,
            x=avg_x,
            y=avg_y,
            confidence=det.confidence,
            interpolated=det.interpolated,
        ))

    return smoothed


def reject_stationary_detections(
    detections: list[BallDetection | None],
    window: int = 5,
    std_threshold: float = 5.0,
) -> list[BallDetection | None]:
    """Reject ball detections that are stationary (false positives).

    For each detection, examines the surrounding window of detections.
    If the standard deviation of positions within the window is below
    std_threshold in both x and y, the detection is marked as None.

    Args:
        detections: List of per-frame detections (None = no detection).
        window: Number of surrounding detections to consider.
        std_threshold: Maximum std_dev in pixels to consider stationary.

    Returns:
        Filtered list with stationary detections replaced by None.
    """
    if len(detections) < window:
        return list(detections)

    result: list[BallDetection | None] = list(detections)

    # Collect all non-None positions
    non_none_positions = [(d.x, d.y) for d in detections if d is not None]

    if len(non_none_positions) < window:
        return result

    xs = [p[0] for p in non_none_positions]
    ys = [p[1] for p in non_none_positions]
    global_std_x = float(np.std(xs))
    global_std_y = float(np.std(ys))

    if global_std_x < std_threshold and global_std_y < std_threshold:
        # All detections are stationary — reject all
        return [None for _ in detections]

    # Per-window check for local stationarity
    half = window // 2
    for i, det in enumerate(detections):
        if det is None:
            continue

        # Gather nearby non-None detections
        start = max(0, i - half)
        end = min(len(detections), i + half + 1)
        nearby = [detections[j] for j in range(start, end) if detections[j] is not None]

        if len(nearby) < 3:
            continue

        local_xs = [d.x for d in nearby]
        local_ys = [d.y for d in nearby]
        if float(np.std(local_xs)) < std_threshold and float(np.std(local_ys)) < std_threshold:
            result[i] = None

    return result


def map_ball_to_court(
    detection: BallDetection,
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    """Map a ball detection from pixel coordinates to court coordinates.

    Uses the homography matrix from court detection (Phase 2A).

    Args:
        detection: Ball detection with pixel (x, y).
        homography: 3x3 homography matrix, or None if unavailable.

    Returns:
        (x, y) court coordinates in meters, or None if homography is None.
    """
    if homography is None:
        return None

    pixel = np.array([detection.x, detection.y, 1.0], dtype=np.float64)
    transformed = homography @ pixel
    w = transformed[2]
    if abs(w) < 1e-10:
        return (0.0, 0.0)

    return (float(transformed[0] / w), float(transformed[1] / w))
