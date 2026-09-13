"""Ball detection — WASB / TrackNet inference over a frame range."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

# Post-processing lives in trajectory.py (own cache key); the names are
# re-exported here because callers and tests historically found them here.
from court_vision.trajectory import (  # noqa: F401
    BallDetection,
    BallTrajectory,
    interpolate_gaps,
    map_ball_to_court,
    postprocess_trajectory,
    reject_static_spots,
    reject_stationary_detections,
    select_ball_path,
    reject_velocity_outliers,
    smooth_trajectory,
    trim_weak_edges,
)

__all__ = [
    "BallDetection", "BallTrajectory", "build_trajectory", "detect_ball_sequence",
    "detect_ball_tracknet", "detect_ball_wasb", "far_ball_roi", "interpolate_gaps",
    "map_ball_to_court", "postprocess_trajectory", "reject_static_spots", "select_ball_path",
    "reject_stationary_detections",
    "reject_velocity_outliers", "smooth_trajectory", "trim_weak_edges",
]


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


# Resolved at call time (not import time) so tests can patch the wrappers
# above on this module.
_DETECTOR_NAMES: dict[str, str] = {
    "wasb": "detect_ball_wasb",
    "tracknet": "detect_ball_tracknet",
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


def detect_ball_candidates_wasb(
    frames: list[np.ndarray],
    frame_index: int,
    **kwargs,
) -> list[BallDetection]:
    """Thin wrapper that lazily imports and delegates to wasb.detect_ball_candidates_wasb."""
    from court_vision.wasb import detect_ball_candidates_wasb as _impl
    return _impl(frames, frame_index, **kwargs)


_CANDIDATE_DETECTOR_NAMES: dict[str, str] = {"wasb": "detect_ball_candidates_wasb"}


def detect_ball_candidates_sequence(
    frames_dir: Path,
    start_frame: int,
    end_frame: int,
    method: str = "wasb",
    confidence_threshold: float = 0.3,
    frame_step: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
    far_roi: tuple[int, int, int, int] | None = None,
    max_candidates: int = 5,
    merge_px: float = 8.0,
    detect_stride: int = 1,
    far_gate: bool = True,
) -> list[list[BallDetection]]:
    """Like :func:`detect_ball_sequence` but keeps up to ``max_candidates``
    heatmap peaks per frame (strongest first), full frame and far crop
    merged (a candidate within ``merge_px`` of another is the same blob;
    the more confident copy stays). One entry per frame, possibly empty.
    The trajectory stage then chooses the motion-consistent path
    (:func:`court_vision.trajectory.select_ball_path`) — the way to track
    the ball in play on a court with other balls lying around.
    """
    frames_dir = Path(frames_dir)
    name = _CANDIDATE_DETECTOR_NAMES.get(method)
    if name is not None:
        detect_fn = globals()[name]
    else:  # a detector without candidate output: its single detection is the only candidate
        single = globals()[_DETECTOR_NAMES[method]]

        def detect_fn(buffer, frame_index, confidence_threshold, max_candidates):
            d = single(buffer, frame_index=frame_index, confidence_threshold=confidence_threshold)
            return [] if d is None else [d]

    step = max(1, int(frame_step))
    stride = max(1, int(detect_stride))
    window = _FrameWindow(frames_dir, capacity=2 * step + 2)
    out: list[list[BallDetection]] = []
    total = end_frame - start_frame + 1
    far_needed = True
    last_best: BallDetection | None = None
    for i in range(start_frame, end_frame + 1):
        if (i - start_frame) % stride:
            out.append([])
            continue
        current = window.get(i)
        if current is None:
            out.append([])
            continue

        def prev(k: int) -> np.ndarray:
            j = max(start_frame, i - k * step)
            frame = window.get(j)
            return current if frame is None else frame

        buffer = [prev(2), prev(1), current]
        cands = list(detect_fn(buffer, frame_index=i, confidence_threshold=confidence_threshold, max_candidates=max_candidates))
        if far_roi is not None and far_gate:
            # the far pass is for a ball the full frame sees badly: skip it
            # while the full frame tracks the ball confidently in the near court
            far_needed = _far_pass_needed(cands[0] if cands else None, far_roi, last_best)
        if far_roi is not None and (far_needed or not far_gate):
            x1, y1, x2, y2 = far_roi
            crop_buffer = [f[y1:y2, x1:x2] for f in buffer]
            if crop_buffer[-1].size:
                for c in detect_fn(crop_buffer, frame_index=i, confidence_threshold=confidence_threshold, max_candidates=max_candidates):
                    c = BallDetection(frame_index=i, x=c.x + x1, y=c.y + y1, confidence=c.confidence)
                    dup = next((k for k, o in enumerate(cands) if abs(o.x - c.x) <= merge_px and abs(o.y - c.y) <= merge_px), None)
                    if dup is None:
                        cands.append(c)
                    elif c.confidence > cands[dup].confidence:
                        cands[dup] = c
        cands.sort(key=lambda d: -d.confidence)
        out.append(cands[:max_candidates])
        if cands:
            last_best = cands[0]
        if progress_callback:
            progress_callback(i - start_frame + 1, total)
    return out


def detect_ball_sequence(
    frames_dir: Path,
    start_frame: int,
    end_frame: int,
    method: str = "wasb",
    confidence_threshold: float = 0.3,
    frame_step: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
    far_roi: tuple[int, int, int, int] | None = None,
    detect_stride: int = 1,
    far_gate: bool = True,
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
        detect_fn = globals()[_DETECTOR_NAMES[method]]
    except KeyError as e:
        raise ValueError(f"Unknown ball detection method {method!r}; expected one of {sorted(_DETECTOR_NAMES)}") from e

    step = max(1, int(frame_step))
    stride = max(1, int(detect_stride))
    window = _FrameWindow(frames_dir, capacity=2 * step + 2)
    raw_detections: list[BallDetection] = []
    total = end_frame - start_frame + 1
    far_needed = True
    last_det: BallDetection | None = None

    for i in range(start_frame, end_frame + 1):
        if (i - start_frame) % stride:
            continue
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
        if far_roi is not None and far_gate:
            far_needed = _far_pass_needed(det, far_roi, last_det)
        if far_roi is not None and (far_needed or not far_gate):
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
            last_det = det
        if progress_callback:
            progress_callback(i - start_frame + 1, total)
    return raw_detections


def _far_pass_needed(
    best: BallDetection | None,
    far_roi: tuple[int, int, int, int],
    last: BallDetection | None = None,
    strong: float = 0.5,
    margin: int = 40,
    max_jump_px_per_frame: float = 75.0,
) -> bool:
    """Whether the upscaled far-court pass is worth running for this frame.

    It exists because a far-court ball is ~1.5 px to the downscaled full
    frame. So: run it when the full frame found nothing, found something
    weak, found the ball in or near the far region, or found something that
    jumped from where the ball was last (a confident blob on a resting ball
    while the real one is far away); skip it while the ball is tracked
    confidently in the near court (about half the frames of a rally), which
    is the single largest saving in the ball stage.
    """
    if best is None or best.confidence < strong:
        return True
    x1, y1, x2, y2 = far_roi
    if (x1 - margin) <= best.x <= (x2 + margin) and (y1 - margin) <= best.y <= (y2 + margin):
        return True
    if last is None:
        return True  # no track to trust yet
    frames = max(1, best.frame_index - last.frame_index)
    if np.hypot(best.x - last.x, best.y - last.y) / frames > max_jump_px_per_frame:
        return True  # the full frame jumped onto something else
    # the ball was last seen in the far region: it may still be there, hidden
    # from the full frame, while the full frame returns a confident blob elsewhere
    if (x1 - margin) <= last.x <= (x2 + margin) and (y1 - margin) <= last.y <= (y2 + margin):
        return True
    return False


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
    homography: np.ndarray | None = None,
    min_run_s: float = 0.12,
    frame_shape: tuple[int, ...] | None = None,
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
    return postprocess_trajectory(raw, fps, max_gap_s=max_gap_s, max_speed_px=max_speed_px,
                                  smooth_window=smooth_window, homography=homography, min_run_s=min_run_s,
                                  frame_shape=frame_shape)


