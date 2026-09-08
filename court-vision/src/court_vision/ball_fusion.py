"""Multi-modal fusion for ball tracking.

Post-processing transforms that refine WASB ball detections using
already-available signals in the pipeline:

1. Segment-boundary filter: drop weak/interpolated detections outside
   the core rally (before first strong detection / after last strong
   detection per gameplay segment). Cuts most "phantom ball" artifacts
   that happen during serves and post-point camera settles.

2. Contact snapping: at detected contact frames, snap the ball to the
   hitting player's dominant wrist when pose is available and the
   WASB-reported position is implausibly far from the racket. The ball
   is physically on the racket at the moment of contact.
"""

from dataclasses import replace

from court_vision.ball_tracker import BallDetection
from court_vision.player_detect import FrameTrackingResult, PoseKeypoints


STRONG_CONFIDENCE = 0.75
CONTACT_SNAP_MAX_WRIST_DISTANCE_PX = 250.0


def _is_strong(ball: BallDetection | None) -> bool:
    return (
        ball is not None
        and not ball.interpolated
        and ball.confidence >= STRONG_CONFIDENCE
    )


def trim_segment_edges(
    tracking_results: list[FrameTrackingResult],
    segments: list[tuple[int, int]],
) -> list[FrameTrackingResult]:
    """Null out weak/interpolated ball detections outside each segment's core.

    For each `(start_frame, end_frame)` range, find the first and last
    **strong** ball detection. Any weak/interpolated detections before
    the first or after the last are set to None — they are almost
    always phantoms (pre-serve bounces, post-point camera movement).

    Segments with no strong detections at all are left untouched (rare
    edge case; don't want to delete a whole rally because WASB was
    uncertain throughout).

    Args:
        tracking_results: Full list of per-frame tracking data.
        segments: List of (start_frame, end_frame) ranges defining
            gameplay segments to trim within.

    Returns:
        New list with trimmed ball detections; non-ball fields
        (players, poses) are preserved.
    """
    by_frame: dict[int, FrameTrackingResult] = {t.frame_index: t for t in tracking_results}

    for start, end in segments:
        strong_frames = [
            idx for idx in range(start, end + 1)
            if idx in by_frame and _is_strong(by_frame[idx].ball)
        ]
        if not strong_frames:
            continue

        first_strong = strong_frames[0]
        last_strong = strong_frames[-1]

        for idx in range(start, end + 1):
            t = by_frame.get(idx)
            if t is None or t.ball is None:
                continue
            if _is_strong(t.ball):
                continue
            if idx < first_strong or idx > last_strong:
                by_frame[idx] = replace(t, ball=None)

    return [by_frame[t.frame_index] for t in tracking_results]


def _dominant_wrist(pose: PoseKeypoints) -> tuple[float, float, float] | None:
    """Return dominant-side wrist keypoint (x, y, visibility).

    Matches the convention used by classify_stroke: right-handed
    assumption — right_wrist is dominant. Falls back to left if right
    is not visible.
    """
    kp = pose.keypoints
    right = kp.get("right_wrist")
    if right is not None and right[2] > 0.3:
        return right
    left = kp.get("left_wrist")
    if left is not None and left[2] > 0.3:
        return left
    return None


def snap_ball_to_racket_at_contacts(
    tracking_results: list[FrameTrackingResult],
    contacts: list[tuple[int, str]],
    max_wrist_distance_px: float = CONTACT_SNAP_MAX_WRIST_DISTANCE_PX,
) -> list[FrameTrackingResult]:
    """Override ball position at contact frames with the hitter's wrist.

    At the moment of contact the ball is physically on the racket face,
    within a few tens of pixels of the dominant wrist. When:

    - there is no ball detection at the contact frame, OR
    - WASB's position is more than max_wrist_distance_px from the wrist

    we replace/create a ball detection at the wrist position. This
    directly improves the contact-frame placement computation that
    feeds Shot.placement.

    Args:
        tracking_results: Per-frame tracking data.
        contacts: (frame_index, player_role) tuples from detect_contacts.
        max_wrist_distance_px: Threshold above which WASB is considered
            wrong and gets overridden.

    Returns:
        New list with snapped ball detections at contact frames.
    """
    by_frame: dict[int, FrameTrackingResult] = {t.frame_index: t for t in tracking_results}

    for frame_idx, role in contacts:
        t = by_frame.get(frame_idx)
        if t is None:
            continue

        hitter_pose = next((p for p in t.poses if p.role == role), None)
        if hitter_pose is None:
            continue

        wrist = _dominant_wrist(hitter_pose)
        if wrist is None:
            continue

        wx, wy, _ = wrist
        ball = t.ball
        needs_snap = ball is None or (
            ((ball.x - wx) ** 2 + (ball.y - wy) ** 2) ** 0.5 > max_wrist_distance_px
        )
        if not needs_snap:
            continue

        snapped = BallDetection(
            frame_index=frame_idx,
            x=float(wx),
            y=float(wy),
            confidence=1.0,
            interpolated=False,
        )
        by_frame[frame_idx] = replace(t, ball=snapped)

    return [by_frame[t.frame_index] for t in tracking_results]
