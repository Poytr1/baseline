"""Ball trajectory post-processing: from raw per-frame detections to a clean
track.

Detection (``ball_tracker.detect_ball_sequence``) is expensive and cached;
everything here is cheap and runs on every experiment, so it lives in its
own module and gets its own cache key.

Pipeline: court gate -> runs -> tracklet linking -> gap interpolation ->
smoothing -> stationary rejection -> edge trimming.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def map_ball_to_court(
    detection: BallDetection,
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    """Map a ball detection from pixel coordinates to court coordinates
    (metres, origin at the net centre). Airborne balls project *beyond*
    their ground position, so treat the result as approximate."""
    if homography is None:
        return None
    pixel = np.array([detection.x, detection.y, 1.0], dtype=np.float64)
    transformed = homography @ pixel
    w = transformed[2]
    if abs(w) < 1e-10:
        return (0.0, 0.0)
    return (float(transformed[0] / w), float(transformed[1] / w))


# ── 1. court gate ────────────────────────────────────────────────────────────

def court_gate(
    detections: list[BallDetection],
    homography: np.ndarray | None,
    frame_shape: tuple[int, ...] | None = None,
    max_x_m: float = 9.5,
    near_margin_m: float = 6.0,
    far_lift: float = 1.2,
    side_margin_m: float = 2.5,
) -> list[BallDetection]:
    """Drop detections nowhere near the court (stands, back wall, ball kids
    at the fence) — judged in *image* space.

    Court-space bounds cannot be used on the far side: a ball a metre or
    two up near the far baseline projects 15-30 m past it through the
    ground homography. So a detection passes if it lies inside the ground
    polygon of the court widened by ``max_x_m`` / ``near_margin_m``, or
    inside the far-court prism: the far half of the court (plus
    ``side_margin_m`` each side and a metre behind the baseline) together
    with its copy lifted ``far_lift`` x the far half's pixel height, which
    is where airborne far-court balls actually appear. The prism follows
    the court's perspective, so it does not reach into the stands beside
    the far baseline the way a plain box would.
    """
    if homography is None:
        return list(detections)
    try:
        H_inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return list(detections)

    def px(cx: float, cy: float):
        p = H_inv @ np.array([cx, cy, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return (float(p[0] / p[2]), float(p[1] / p[2]))

    quad = [px(-max_x_m, -11.885 - near_margin_m), px(max_x_m, -11.885 - near_margin_m),
            px(max_x_m, 11.885 + 2.0), px(-max_x_m, 11.885 + 2.0)]
    half_w = 5.485 + side_margin_m
    far = [px(-half_w, 0.0), px(half_w, 0.0), px(half_w, 11.885 + 1.0), px(-half_w, 11.885 + 1.0)]
    baseline = [px(-5.485, 11.885), px(5.485, 11.885), px(-5.485, 0.0), px(5.485, 0.0)]
    if any(p is None for p in quad + far + baseline):
        return list(detections)
    import cv2

    poly = np.array(quad, dtype=np.float32)
    ys = [p[1] for p in baseline]
    lift = far_lift * (max(ys) - min(ys))
    prism = np.array(far + [(x, y - lift) for x, y in far], dtype=np.float32)
    if frame_shape is not None:
        h, w = frame_shape[:2]
        prism[:, 0] = np.clip(prism[:, 0], 0.0, float(w))
        prism[:, 1] = np.clip(prism[:, 1], 0.0, float(h))
    hull = cv2.convexHull(prism)

    kept = []
    for d in detections:
        pt = (float(d.x), float(d.y))
        if cv2.pointPolygonTest(poly, pt, False) >= 0 or cv2.pointPolygonTest(hull, pt, False) >= 0:
            kept.append(d)
    return kept


# ── 2. runs + tracklet linking ───────────────────────────────────────────────

def _speed(a: BallDetection, b: BallDetection) -> float:
    gap = max(abs(b.frame_index - a.frame_index), 1)
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 / gap


def build_runs(
    detections: list[BallDetection],
    max_speed_px_per_frame: float,
    max_gap_frames: int | None = None,
    gap_speed_ratio: float = 4.0,
    gap_speed_floor: float = 3.0,
) -> list[list[BallDetection]]:
    """Group consecutive detections into physically continuous runs.

    A step faster than the cap, or (optionally) a gap longer than
    ``max_gap_frames``, starts a new run. A step *across* a gap of two or
    more frames is also held to the ball's local speed: the cap is an
    average over the gap, so a blip at the bottom of the frame six frames
    before the ball re-appears at the far baseline "moves" at a legal
    speed — but not at ``gap_speed_ratio`` times the speed the ball has on
    either side of the gap (floored at ``gap_speed_floor`` px/frame).
    """
    runs: list[list[BallDetection]] = []
    n = len(detections)
    for i, det in enumerate(detections):
        if not runs:
            runs.append([det])
            continue
        prev = runs[-1][-1]
        gap = det.frame_index - prev.frame_index
        step = _speed(prev, det)
        ok = step <= max_speed_px_per_frame and (max_gap_frames is None or gap <= max_gap_frames)
        if ok and gap >= 2:
            local = []
            if len(runs[-1]) >= 2:
                local.append(_speed(runs[-1][-2], prev))
            if i + 1 < n and detections[i + 1].frame_index - det.frame_index <= 2:
                local.append(_speed(det, detections[i + 1]))
            if local:  # with no speed to compare against, the cap alone decides
                ok = step <= gap_speed_ratio * max(max(local), gap_speed_floor)
        if ok:
            runs[-1].append(det)
        else:
            runs.append([det])
    return runs


def reject_velocity_outliers(
    detections: list[BallDetection],
    fps: float,
    max_speed_px_per_frame: float = 150.0,
    min_run: int = 3,
    min_run_s: float = 0.12,
    link_s: float = 1.0,
    link_speed_ratio: float = 5.0,
    link_speed_floor: float = 3.0,
    lone_link_s: float = 0.25,
    blip_extent_px: float = 0.0,
    blip_max_s: float = 1.0,
    stutter_px: float = 1.5,
    stutter_frac: float = 0.4,
) -> list[BallDetection]:
    """Keep the ball's track and drop the blips.

    Detections are grouped into runs (see :func:`build_runs`). A run is
    *long* if it has at least ``min_run`` detections and spans at least
    ``min_run_s``. Long runs are kept — unless they are a *blip*: shorter
    than ``blip_max_s`` and either moving less than ``blip_extent_px``
    overall or standing still (steps under ``stutter_px``) for at least
    ``stutter_frac`` of its frames. A ball never hangs still for a quarter
    second; a spectator's head, a ball on the ground, or the detector
    hopping between heads in the crowd does. ``blip_extent_px`` of 0
    disables both checks. A short run is kept only if it links to a
    kept run within ``link_s``: the bridge between them must be slower
    than the speed cap and not more than ``link_speed_ratio`` times the
    speed the ball has on the long run's end (floored at
    ``link_speed_floor`` px/frame, so slow far-court balls can still
    re-appear a few pixels away). A run with fewer than ``min_run``
    detections is too little evidence to bridge more than ``lone_link_s``.
    Everything else — isolated blips and chains of blips — goes.

    Args:
        detections: Sorted raw ball detections (no Nones).
        fps: Video frame rate.
        max_speed_px_per_frame: Maximum allowed displacement per frame in pixels.
        min_run: Minimum detections for a run to stand on its own.
        min_run_s: Minimum duration for a run to stand on its own.
        link_s: Maximum time gap over which a short run may link to a long one.
        lone_link_s: Maximum gap for a run of fewer than ``min_run`` detections.
        blip_extent_px: A run shorter than ``blip_max_s`` that moves less than this is dropped (0 = off).
        blip_max_s: Runs at least this long are judged by the stationary filter instead.
        stutter_px: A step shorter than this counts as standing still.
        stutter_frac: A run shorter than ``blip_max_s`` standing still this often is dropped.
    """
    if len(detections) < 2:
        return list(detections)
    # A gap longer than ~a third of a second also ends a run, so that a ball
    # re-appearing after an occlusion is a separate tracklet that has to
    # *earn* its link (speed-consistent bridge) instead of being glued on.
    runs = build_runs(detections, max_speed_px_per_frame, max_gap_frames=max(2, int(round(fps / 2.0))))
    min_len = max(min_run, int(round(min_run_s * fps)))
    link = max(1, int(round(link_s * fps)))
    lone_link = max(1, int(round(lone_link_s * fps)))
    blip_frames = int(round(blip_max_s * fps))

    def is_blip(run: list[BallDetection]) -> bool:
        if blip_extent_px <= 0 or run[-1].frame_index - run[0].frame_index >= blip_frames:
            return False
        xs = [d.x for d in run]
        ys = [d.y for d in run]
        if float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))) < blip_extent_px:
            return True
        if len(run) < 3:
            return False
        still = sum(1 for a, b in zip(run, run[1:]) if _speed(a, b) < stutter_px)
        return still >= stutter_frac * (len(run) - 1)

    def is_long(run: list[BallDetection]) -> bool:
        return len(run) >= min_len and not is_blip(run)

    def end_speed(run: list[BallDetection], head: bool) -> float:
        if len(run) < 2:
            return 0.0
        a, b = (run[0], run[1]) if head else (run[-2], run[-1])
        return _speed(a, b)

    def links(short: list[BallDetection], anchor: list[BallDetection], anchor_before: bool) -> bool:
        if anchor_before:
            a, b = anchor[-1], short[0]
            ref = end_speed(anchor, head=False)
        else:
            a, b = short[-1], anchor[0]
            ref = end_speed(anchor, head=True)
        if b.frame_index - a.frame_index > (link if len(short) >= min_run else lone_link):
            return False
        bridge = _speed(a, b)
        return bridge <= max_speed_px_per_frame and bridge <= link_speed_ratio * max(ref, link_speed_floor)

    keep = [is_long(r) for r in runs]
    blip = [is_blip(r) for r in runs]
    # Short runs may link to kept neighbours; iterate so a chain that leads
    # to a long run survives while an isolated chain does not.
    changed = True
    while changed:
        changed = False
        for i, run in enumerate(runs):
            if keep[i] or blip[i]:
                continue
            prev_ok = i > 0 and keep[i - 1] and links(run, runs[i - 1], anchor_before=True)
            next_ok = i + 1 < len(runs) and keep[i + 1] and links(run, runs[i + 1], anchor_before=False)
            if prev_ok or next_ok:
                keep[i] = True
                changed = True
    kept: list[BallDetection] = []
    for i, run in enumerate(runs):
        if keep[i]:
            kept.extend(run)
    return kept


# ── 3. gaps, smoothing, stationary, edges ────────────────────────────────────

def interpolate_gaps(
    detections: list[BallDetection],
    fps: float,
    max_gap_s: float = 0.5,
    max_speed_px_per_frame: float | None = None,
) -> list[BallDetection]:
    """Fill short gaps with a locally fitted quadratic (linear with < 3
    anchors). A gap is only bridged when the ball would have to move no
    faster than the cap and no more than 2.5x faster than it moves on
    either side: an occluded ball keeps its speed, two different objects
    do not."""
    if len(detections) <= 1:
        return list(detections)

    max_gap_frames = int(max_gap_s * fps)
    result: list[BallDetection] = [detections[0]]

    for i in range(1, len(detections)):
        prev = detections[i - 1]
        curr = detections[i]
        gap = curr.frame_index - prev.frame_index

        implied = _speed(prev, curr)
        same_track = max_speed_px_per_frame is None or implied <= max_speed_px_per_frame
        if same_track and max_speed_px_per_frame is not None:
            before = _speed(detections[i - 2], prev) if i >= 2 else None
            after = _speed(curr, detections[i + 1]) if i + 1 < len(detections) else None
            refs = [v for v in (before, after) if v is not None]
            if refs and implied > 2.5 * max(min(refs), 2.0):
                same_track = False
        if 1 < gap <= max_gap_frames and same_track:
            before_a = detections[max(0, i - 2):i]
            after_a = detections[i:min(len(detections), i + 2)]
            local_anchors = before_a + after_a
            anchor_frames = np.array([d.frame_index for d in local_anchors], dtype=np.float64)
            anchor_x = np.array([d.x for d in local_anchors], dtype=np.float64)
            anchor_y = np.array([d.y for d in local_anchors], dtype=np.float64)
            degree = min(2, len(local_anchors) - 1)
            poly_x = np.polyfit(anchor_frames, anchor_x, degree)
            poly_y = np.polyfit(anchor_frames, anchor_y, degree)
            for j in range(1, gap):
                frame_idx = prev.frame_index + j
                result.append(BallDetection(
                    frame_index=frame_idx,
                    x=float(np.polyval(poly_x, frame_idx)),
                    y=float(np.polyval(poly_y, frame_idx)),
                    confidence=min(prev.confidence, curr.confidence) * 0.5,
                    interpolated=True,
                ))
        result.append(curr)
    return result


def smooth_trajectory(
    detections: list[BallDetection],
    window: int = 3,
) -> list[BallDetection]:
    """Centred moving average over *consecutive* frames only, so a gap never
    bleeds one side's motion into the other."""
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
        smoothed.append(BallDetection(
            frame_index=det.frame_index,
            x=sum(d.x for d in neighbors) / len(neighbors),
            y=sum(d.y for d in neighbors) / len(neighbors),
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
    """Reject detections that sit still (a ball on the ground, a logo).

    With ``fps`` the window is ``window_s`` long and must be ``min_fill``
    populated: a ball in play never stays within a few pixels for a whole
    second, but it does move only 1-3 px/frame at the far baseline or at
    the top of its arc, so a short window would delete exactly the frames
    where the far player hits it.
    """
    if fps:
        window = max(window, int(window_s * fps))
    if len(detections) < window:
        return list(detections)
    result: list[BallDetection | None] = list(detections)
    positions = [(d.x, d.y) for d in detections if d is not None]
    if len(positions) < window:
        return result
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    if float(np.std(xs)) < std_threshold and float(np.std(ys)) < std_threshold:
        return [None for _ in detections]
    half = window // 2
    min_count = max(3, int(min_fill * window)) if fps else 3
    for i, det in enumerate(detections):
        if det is None:
            continue
        start, end = max(0, i - half), min(len(detections), i + half + 1)
        nearby = [detections[j] for j in range(start, end) if detections[j] is not None]
        if len(nearby) < min_count:
            continue
        if float(np.std([d.x for d in nearby])) < std_threshold and float(np.std([d.y for d in nearby])) < std_threshold:
            result[i] = None
    return result


def trim_weak_edges(
    detections: list[BallDetection | None],
    strong_confidence: float = 0.75,
) -> list[BallDetection | None]:
    """Null out weak/interpolated detections before the first and after the
    last *strong* detection of a segment (pre-serve bounces, post-point
    camera moves). A segment with no strong detection is left untouched."""
    def strong(b: BallDetection | None) -> bool:
        return b is not None and not b.interpolated and b.confidence >= strong_confidence

    strong_idx = [i for i, d in enumerate(detections) if strong(d)]
    if not strong_idx:
        return list(detections)
    first, last = strong_idx[0], strong_idx[-1]
    return [d if (d is None or strong(d) or first <= i <= last) else None for i, d in enumerate(detections)]


# ── drivers ──────────────────────────────────────────────────────────────────

def postprocess_trajectory(
    raw_detections: list[BallDetection],
    fps: float,
    max_gap_s: float = 0.5,
    max_speed_px: float = 150.0,
    smooth_window: int = 3,
    homography: np.ndarray | None = None,
    frame_shape: tuple[int, ...] | None = None,
    min_run_s: float = 0.12,
) -> BallTrajectory:
    """Court gate -> outlier/tracklet rejection -> gap interpolation -> smoothing.

    ``max_speed_px`` is per *30fps* frame and is scaled by the actual fps.
    ``min_run_s`` is the shortest run that can stand on its own (see
    :func:`reject_velocity_outliers`).
    """
    per_frame_speed = max_speed_px * 30.0 / max(fps, 1.0)
    gated = court_gate(raw_detections, homography, frame_shape)
    blip_px = 0.03 * float(np.hypot(frame_shape[1], frame_shape[0])) if frame_shape else 40.0
    filtered = reject_velocity_outliers(
        gated, fps, max_speed_px_per_frame=per_frame_speed, min_run_s=min_run_s, blip_extent_px=blip_px,
    )
    interpolated = interpolate_gaps(filtered, fps, max_gap_s, max_speed_px_per_frame=per_frame_speed)
    smoothed = smooth_trajectory(interpolated, window=smooth_window)
    return BallTrajectory(detections=smoothed, fps=fps)


def finalize_ball_by_frame(
    detections: list[BallDetection],
    segment,
    stationary_std_px: float = 3.0,
    strong_confidence: float = 0.75,
    fps: float = 30.0,
) -> dict[int, BallDetection | None]:
    """Stationary rejection + edge trimming, returning a per-frame lookup."""
    raw_dets: list[BallDetection | None] = [None] * (segment.end_frame - segment.start_frame + 1)
    for det in detections:
        idx = det.frame_index - segment.start_frame
        if 0 <= idx < len(raw_dets):
            raw_dets[idx] = det
    filtered = reject_stationary_detections(raw_dets, std_threshold=stationary_std_px, fps=fps)
    filtered = trim_weak_edges(filtered, strong_confidence=strong_confidence)
    return {segment.start_frame + i: det for i, det in enumerate(filtered)}
