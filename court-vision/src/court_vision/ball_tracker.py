"""Ball tracking — neural detection, trajectory building, and court coordinate mapping."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class BallTrajectory:
    """Sequence of ball detections across frames."""

    detections: list[BallDetection]
    fps: float


def detect_ball_tracknet(
    frames: list[np.ndarray],
    frame_index: int,
    **kwargs,
) -> BallDetection | None:
    """Thin wrapper that lazily imports and delegates to tracknet.detect_ball_tracknet.

    This wrapper exists so that tests can patch
    ``court_vision.ball_tracker.detect_ball_tracknet`` without triggering the
    circular import that would result from a top-level import of tracknet.
    """
    from court_vision.tracknet import detect_ball_tracknet as _impl
    return _impl(frames, frame_index, **kwargs)


def detect_ball_wasb(
    frames: list[np.ndarray],
    frame_index: int,
    **kwargs,
) -> BallDetection | None:
    """Thin wrapper that lazily imports and delegates to wasb.detect_ball_wasb."""
    from court_vision.wasb import detect_ball_wasb as _impl
    return _impl(frames, frame_index, **kwargs)


_DETECTORS: dict[str, Callable] = {
    "wasb": detect_ball_wasb,
    "tracknet": detect_ball_tracknet,
}


class _FrameWindow:
    """Small LRU of decoded frames so a sliding 3-frame window never holds a
    whole segment in memory."""

    def __init__(self, frames_dir: Path, capacity: int) -> None:
        self.frames_dir = frames_dir
        self.capacity = max(3, capacity)
        self._cache: OrderedDict[int, np.ndarray | None] = OrderedDict()

    def get(self, index: int) -> np.ndarray | None:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        frame = cv2.imread(str(self.frames_dir / f"frame_{index:06d}.jpg"))
        self._cache[index] = frame
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return frame


def detect_ball_sequence(
    frames_dir: Path,
    start_frame: int,
    end_frame: int,
    method: str = "wasb",
    confidence_threshold: float = 0.3,
    frame_step: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
    far_roi: tuple[int, int, int, int] | None = None,
) -> list[BallDetection]:
    """Run the neural ball detector over a frame range (raw detections only).

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        start_frame: First frame index to process.
        end_frame: Last frame index to process (inclusive).
        method: "wasb" (HRNet heatmap, recommended) or "tracknet" (TrackNet v2).
        confidence_threshold: Minimum heatmap peak to accept a detection.
        frame_step: Temporal spacing of the 3-frame window. Both models were
            trained on ~30fps footage, so 60fps video uses step 2 to keep the
            inter-frame motion comparable.
        progress_callback: Optional (current, total) callback.
        far_roi: Optional (x1, y1, x2, y2) pixel region of the far court. The
            detectors see a 512x288 / 640x360 downscale of the whole frame,
            which shrinks a far-court ball to ~1.5 px; running them a second
            time on this crop keeps the ball a few pixels wide. The higher
            confidence detection of the two wins.
    """
    frames_dir = Path(frames_dir)
    try:
        detect_fn = _DETECTORS[method]
    except KeyError as e:
        raise ValueError(f"Unknown ball detection method {method!r}; expected one of {sorted(_DETECTORS)}") from e

    step = max(1, int(frame_step))
    window = _FrameWindow(frames_dir, capacity=2 * step + 2)
    raw_detections: list[BallDetection] = []
    total = end_frame - start_frame + 1

    for i in range(start_frame, end_frame + 1):
        current = window.get(i)
        if current is None:
            continue

        def prev(k: int) -> np.ndarray:
            # Before the window is full, repeat the earliest available frame:
            # a static triplet yields a low-confidence heatmap instead of the
            # spurious peaks a black frame produces.
            j = max(start_frame, i - k * step)
            frame = window.get(j)
            return current if frame is None else frame

        buffer = [prev(2), prev(1), current]
        det = detect_fn(buffer, frame_index=i, confidence_threshold=confidence_threshold)
        if far_roi is not None:
            x1, y1, x2, y2 = far_roi
            crop_buffer = [f[y1:y2, x1:x2] for f in buffer]
            if crop_buffer[-1].size:
                cdet = detect_fn(crop_buffer, frame_index=i, confidence_threshold=confidence_threshold)
                if cdet is not None:
                    cdet = BallDetection(frame_index=i, x=cdet.x + x1, y=cdet.y + y1, confidence=cdet.confidence)
                    if det is None or cdet.confidence > det.confidence:
                        det = cdet
        if det is not None:
            raw_detections.append(det)
        if progress_callback:
            progress_callback(i - start_frame + 1, total)
    return raw_detections


def far_ball_roi(homography: np.ndarray | None, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    """Far-court region for the second ball-detector pass: from well behind
    the far baseline down to just past the net, sized so its width is about
    half the frame (a 2x zoom for the detector)."""
    if homography is None:
        return None
    try:
        H_inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    pts = []
    for cx, cy in ((-5.485, 11.885), (5.485, 11.885), (-5.485, 0.0), (5.485, 0.0)):
        p = H_inv @ np.array([cx, cy, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        pts.append((p[0] / p[2], p[1] / p[2]))
    h, w = frame_shape[:2]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    far_y, net_y = min(ys), max(ys)
    cx = (min(xs) + max(xs)) / 2
    half_w = max((max(xs) - min(xs)) * 0.65, w * 0.3)
    y1 = max(0, int(far_y - 1.2 * (net_y - far_y)))
    y2 = min(h, int(net_y + 0.25 * (net_y - far_y)))
    x1 = max(0, int(cx - half_w))
    x2 = min(w, int(cx + half_w))
    if x2 - x1 < 64 or y2 - y1 < 36:
        return None
    return (x1, y1, x2, y2)


def postprocess_trajectory(
    raw_detections: list[BallDetection],
    fps: float,
    max_gap_s: float = 0.5,
    max_speed_px: float = 150.0,
    smooth_window: int = 3,
) -> BallTrajectory:
    """Outlier rejection -> gap interpolation -> smoothing.

    ``max_speed_px`` is per *30fps* frame and is scaled by the actual fps.
    """
    per_frame_speed = max_speed_px * 30.0 / max(fps, 1.0)
    filtered = reject_velocity_outliers(raw_detections, fps, max_speed_px_per_frame=per_frame_speed)
    interpolated = interpolate_gaps(filtered, fps, max_gap_s, max_speed_px_per_frame=per_frame_speed)
    smoothed = smooth_trajectory(interpolated, window=smooth_window)
    return BallTrajectory(detections=smoothed, fps=fps)


def build_trajectory(
    frames_dir: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    max_gap_s: float = 0.5,
    method: str = "wasb",
    progress_callback: Callable[[int, int], None] | None = None,
    confidence_threshold: float = 0.3,
    frame_step: int = 1,
    max_speed_px: float = 150.0,
    smooth_window: int = 3,
    far_roi: tuple[int, int, int, int] | None = None,
) -> BallTrajectory:
    """Detect the ball in every frame of a range and build a clean trajectory.

    Convenience wrapper: :func:`detect_ball_sequence` followed by
    :func:`postprocess_trajectory`.
    """
    raw = detect_ball_sequence(
        frames_dir, start_frame, end_frame, method=method,
        confidence_threshold=confidence_threshold, frame_step=frame_step,
        progress_callback=progress_callback, far_roi=far_roi,
    )
    return postprocess_trajectory(raw, fps, max_gap_s=max_gap_s, max_speed_px=max_speed_px, smooth_window=smooth_window)


def reject_velocity_outliers(
    detections: list[BallDetection],
    fps: float,
    max_speed_px_per_frame: float = 150.0,
    min_run: int = 3,
) -> list[BallDetection]:
    """Reject detections that imply physically impossible ball movement.

    Consecutive detections are grouped into *runs* where each step is
    within ``max_speed_px_per_frame`` (scaled by the frame gap). Short runs
    (fewer than ``min_run`` detections) are dropped when the detections on
    either side of them are consistent with each other, i.e. the short run
    is a blip off the real track (typically 1-2 frames of a false peak on a
    player's shoe or a line). A short run at the very start/end of the
    track is dropped only if it is inconsistent with its single neighbour.

    Args:
        detections: Sorted raw ball detections (no Nones).
        fps: Video frame rate (unused; kept for API symmetry).
        max_speed_px_per_frame: Maximum allowed displacement per frame in pixels.
        min_run: Runs shorter than this are candidates for removal.

    Returns:
        Filtered list of detections with outliers removed.
    """
    n = len(detections)
    if n < 2:
        return list(detections)

    def ok(a: BallDetection, b: BallDetection) -> bool:
        gap = max(abs(b.frame_index - a.frame_index), 1)
        dist = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
        return dist <= max_speed_px_per_frame * gap

    runs: list[list[BallDetection]] = [[detections[0]]]
    for det in detections[1:]:
        if ok(runs[-1][-1], det):
            runs[-1].append(det)
        else:
            runs.append([det])

    def speed(a: BallDetection, b: BallDetection) -> float:
        gap = max(abs(b.frame_index - a.frame_index), 1)
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 / gap

    def end_speed(run: list[BallDetection], head: bool) -> float:
        """Speed at the start (head) or end of a run (px/frame)."""
        if len(run) < 2:
            return 0.0
        a, b = (run[0], run[1]) if head else (run[-2], run[-1])
        return speed(a, b)

    def consistent(bridge: float, neighbour_speed: float, internal: float | None) -> bool:
        """A short run continues a neighbour only if bridging it and moving
        along it look like the same ball: speeds within 3x of the
        neighbour's (plus a 2 px/frame floor for slow far-court balls)."""
        ref = max(neighbour_speed, 2.0)
        if bridge > max_speed_px_per_frame or bridge > 3.0 * ref:
            return False
        if internal is not None and internal > 3.0 * ref:
            return False
        return True

    keep = [True] * len(runs)
    for r, run in enumerate(runs):
        if len(run) >= min_run:
            continue
        prev_run = runs[r - 1] if r > 0 else None
        next_run = runs[r + 1] if r + 1 < len(runs) else None
        internal = speed(run[0], run[-1]) if len(run) >= 2 else None
        joins_prev = (prev_run is not None and len(prev_run) >= min_run
                      and consistent(speed(prev_run[-1], run[0]), end_speed(prev_run, head=False), internal))
        joins_next = (next_run is not None and len(next_run) >= min_run
                      and consistent(speed(run[-1], next_run[0]), end_speed(next_run, head=True), internal))
        keep[r] = joins_prev or joins_next

    kept: list[BallDetection] = []
    for r, run in enumerate(runs):
        if keep[r]:
            kept.extend(run)
    return kept


def interpolate_gaps(
    detections: list[BallDetection],
    fps: float,
    max_gap_s: float = 0.5,
    max_speed_px_per_frame: float | None = None,
) -> list[BallDetection]:
    """Fill short gaps in ball detections with locally-fit interpolation.

    For each gap, fits a quadratic polynomial to nearby anchor points
    (up to 2 detections on each side of the gap). Falls back to linear
    when fewer than 3 local anchors are available.
    """
    if len(detections) <= 1:
        return list(detections)

    max_gap_frames = int(max_gap_s * fps)
    result: list[BallDetection] = [detections[0]]

    for i in range(1, len(detections)):
        prev = detections[i - 1]
        curr = detections[i]
        gap = curr.frame_index - prev.frame_index

        implied_speed = ((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2) ** 0.5 / max(gap, 1)
        same_track = max_speed_px_per_frame is None or implied_speed <= max_speed_px_per_frame
        if same_track and max_speed_px_per_frame is not None:
            # The ball keeps roughly its speed through an occlusion; a gap that
            # would have to be crossed much faster than the ball moves on
            # either side joins two different objects.
            def local_speed(a: BallDetection, b: BallDetection) -> float:
                g = max(abs(b.frame_index - a.frame_index), 1)
                return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 / g
            before = local_speed(detections[i - 2], prev) if i >= 2 else None
            after = local_speed(curr, detections[i + 1]) if i + 1 < len(detections) else None
            refs = [v for v in (before, after) if v is not None]
            if refs and implied_speed > 2.5 * max(min(refs), 2.0):
                same_track = False
        if 1 < gap <= max_gap_frames and same_track:
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
                interp_conf = min(prev.confidence, curr.confidence) * 0.5
                result.append(BallDetection(
                    frame_index=frame_idx,
                    x=float(np.polyval(poly_x, frame_idx)),
                    y=float(np.polyval(poly_y, frame_idx)),
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

    Smooths x and y coordinates independently using a centered moving
    average over *consecutive* frames only (a gap breaks the window so a
    bounce or hit on one side of a gap never bleeds into the other).
    """
    if window <= 1 or len(detections) < window:
        return list(detections)

    half = window // 2
    smoothed: list[BallDetection] = []

    for i, det in enumerate(detections):
        neighbors = [det]
        for k in range(1, half + 1):
            if i - k >= 0 and det.frame_index - detections[i - k].frame_index == k:
                neighbors.append(detections[i - k])
            if i + k < len(detections) and detections[i + k].frame_index - det.frame_index == k:
                neighbors.append(detections[i + k])
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
    fps: float | None = None,
    window_s: float = 1.0,
    min_fill: float = 0.8,
) -> list[BallDetection | None]:
    """Reject ball detections that sit still (a ball on the ground, a logo).

    A detection is dropped when the detections in a window around it have a
    standard deviation below ``std_threshold`` in both x and y. With ``fps``
    the window is ``window_s`` seconds long and must be at least ``min_fill``
    populated: a ball in play never stays within a few pixels for a whole
    second, but it does move only 1-3 px/frame at the far baseline or at the
    top of its arc, so a short window would delete exactly the frames where
    the far player hits it.
    """
    if fps:
        window = max(window, int(window_s * fps))
    if len(detections) < window:
        return list(detections)

    result: list[BallDetection | None] = list(detections)

    non_none_positions = [(d.x, d.y) for d in detections if d is not None]
    if len(non_none_positions) < window:
        return result

    xs = [p[0] for p in non_none_positions]
    ys = [p[1] for p in non_none_positions]
    if float(np.std(xs)) < std_threshold and float(np.std(ys)) < std_threshold:
        return [None for _ in detections]

    half = window // 2
    min_count = max(3, int(min_fill * window)) if fps else 3
    for i, det in enumerate(detections):
        if det is None:
            continue
        start = max(0, i - half)
        end = min(len(detections), i + half + 1)
        nearby = [detections[j] for j in range(start, end) if detections[j] is not None]
        if len(nearby) < min_count:
            continue
        local_xs = [d.x for d in nearby]
        local_ys = [d.y for d in nearby]
        if float(np.std(local_xs)) < std_threshold and float(np.std(local_ys)) < std_threshold:
            result[i] = None

    return result


def trim_weak_edges(
    detections: list[BallDetection | None],
    strong_confidence: float = 0.75,
) -> list[BallDetection | None]:
    """Null out weak/interpolated detections before the first and after the
    last *strong* detection of a segment.

    Those edge detections are almost always phantoms (pre-serve bounces,
    post-point camera moves). A segment with no strong detection at all is
    left untouched.
    """
    def strong(b: BallDetection | None) -> bool:
        return b is not None and not b.interpolated and b.confidence >= strong_confidence

    strong_idx = [i for i, d in enumerate(detections) if strong(d)]
    if not strong_idx:
        return list(detections)
    first, last = strong_idx[0], strong_idx[-1]
    return [
        d if (d is None or strong(d) or first <= i <= last) else None
        for i, d in enumerate(detections)
    ]


def map_ball_to_court(
    detection: BallDetection,
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    """Map a ball detection from pixel coordinates to court coordinates.

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
