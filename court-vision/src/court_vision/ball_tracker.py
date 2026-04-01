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
    min_confidence: float = 0.0,
) -> BallDetection | None:
    """Detect the tennis ball in a single frame using color + shape filtering.

    Looks for small, bright, circular objects (white or yellow) that match
    typical tennis ball appearance in broadcast video.

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

    if best_detection is not None and best_detection.confidence < min_confidence:
        return None
    return best_detection


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


def build_trajectory(
    frames_dir: "Path",
    start_frame: int,
    end_frame: int,
    fps: float,
    max_gap_s: float = 0.5,
    method: str = "tracknet",
) -> BallTrajectory:
    """Build a ball trajectory by detecting the ball in each frame.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        start_frame: First frame index to process.
        end_frame: Last frame index to process (inclusive).
        fps: Video frame rate.
        max_gap_s: Maximum gap in seconds to interpolate through.
        method: Detection method — "tracknet" (3-frame sliding window)
                or "hsv" (per-frame color+contour).

    Returns:
        BallTrajectory with detections and interpolated positions.
    """
    from pathlib import Path

    frames_dir = Path(frames_dir)
    raw_detections: list[BallDetection] = []

    if method == "tracknet":
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
            det = detect_ball_tracknet(buffer, frame_index=i)
            if det is not None:
                raw_detections.append(det)
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

    interpolated = interpolate_gaps(raw_detections, fps, max_gap_s)

    return BallTrajectory(detections=interpolated, fps=fps)


def interpolate_gaps(
    detections: list[BallDetection],
    fps: float,
    max_gap_s: float = 0.5,
) -> list[BallDetection]:
    """Fill short gaps in ball detections with linear interpolation.

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
            for j in range(1, gap):
                t = j / gap
                interp_x = prev.x + t * (curr.x - prev.x)
                interp_y = prev.y + t * (curr.y - prev.y)
                interp_conf = min(prev.confidence, curr.confidence) * 0.5
                result.append(BallDetection(
                    frame_index=prev.frame_index + j,
                    x=interp_x,
                    y=interp_y,
                    confidence=interp_conf,
                    interpolated=True,
                ))

        result.append(curr)

    return result


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
