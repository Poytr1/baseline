"""Shot classification — stroke type, placement zones, points and outcomes."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from court_vision.ball_tracker import map_ball_to_court
from court_vision.config import PipelineSettings
from court_vision.hit_detect import Hit, detect_hits
from court_vision.player_detect import FrameTrackingResult, PlayerDetection, PoseKeypoints
from court_vision.scene_filter import GameplaySegment
from court_vision.scoreboard import ScoreTimeline, infer_point_winner_row

_SERVICE_LINE_Y = 6.4  # Service line distance from net in meters
_SINGLES_WIDTH = 4.115  # Singles sideline to center
_BASELINE_Y = 11.885

STROKES = ("forehand", "backhand", "serve", "volley", "overhead", "slice")


@dataclass
class ShotPlacement:
    """Ball landing placement on court."""

    x: float  # court x in meters
    y: float  # court y in meters
    zone: str  # e.g. "crosscourt_deep", "wide", "t"


@dataclass
class Shot:
    """A single shot within a point."""

    shot_number: int
    frame: int
    time_s: float
    player: str  # "near_player" or "far_player"
    stroke: str  # forehand, backhand, serve, volley, overhead, slice
    placement: ShotPlacement | None
    confidence: float


@dataclass
class Point:
    """A single point in the match."""

    point_number: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    server: str | None  # "near_player" or "far_player"
    shots: list[Shot]
    outcome: str | None  # "winner", "error", "unforced_error"
    outcome_player: str | None
    rally_length: int
    review_status: str = "pending"
    winner: str | None = None  # explicit point winner role
    outcome_source: str | None = None  # "scoreboard" | "trajectory" | "last_hitter"


@dataclass
class MatchData:
    """Full match data for export."""

    match_id: str
    source_url: str
    metadata: dict
    points: list[Point]


def point_winner(point: Point) -> str | None:
    """Winner role of a point, from the explicit field or (outcome, outcome_player)."""
    if point.winner:
        return point.winner
    if point.outcome_player is None:
        return None
    if point.outcome == "winner":
        return point.outcome_player
    if point.outcome in ("error", "unforced_error"):
        return other_role(point.outcome_player)
    return None


def other_role(role: str) -> str:
    return "far_player" if role == "near_player" else "near_player"


# ── Placement zones ──────────────────────────────────────────────────────────

def compute_placement_zone(
    x: float,
    y: float,
    is_serve: bool = False,
    server_side: str | None = None,
    hitter: str | None = None,
) -> str:
    """Compute the placement zone from ball landing court coordinates."""
    if is_serve:
        return _compute_serve_zone(x, y, server_side or "deuce")
    return _compute_rally_zone(x, y, hitter or "near_player")


def _compute_serve_zone(x: float, y: float, server_side: str) -> str:
    abs_x = abs(x)
    if abs_x < _SINGLES_WIDTH / 3:
        return "t"
    if abs_x > _SINGLES_WIDTH * 2 / 3:
        return "wide"
    return "body"


def _compute_rally_zone(x: float, y: float, hitter: str) -> str:
    abs_y = abs(y)
    abs_x = abs(x)
    depth = "short" if abs_y < _SERVICE_LINE_Y else "deep"
    if abs_x < _SINGLES_WIDTH / 3:
        direction = "middle"
    elif hitter == "near_player":
        direction = "crosscourt" if x > 0 else "down_the_line"
    else:
        direction = "crosscourt" if x < 0 else "down_the_line"
    return f"{direction}_{depth}"


# ── Stroke classification ────────────────────────────────────────────────────

def _kp(kp: dict, name: str, min_vis: float = 0.3) -> tuple[float, float] | None:
    v = kp.get(name)
    if v is None or v[2] < min_vis:
        return None
    return (v[0], v[1])


def _body_center_x(kp: dict) -> float | None:
    for a, b in (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")):
        pa, pb = _kp(kp, a), _kp(kp, b)
        if pa and pb:
            return (pa[0] + pb[0]) / 2
    return None


def torso_length(kp: dict) -> float | None:
    ls, rs = _kp(kp, "left_shoulder"), _kp(kp, "right_shoulder")
    lh, rh = _kp(kp, "left_hip"), _kp(kp, "right_hip")
    if ls and rs and lh and rh:
        sx, sy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
        hx, hy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
        return math.hypot(sx - hx, sy - hy)
    return None


def dominant_wrist(kp: dict, hand: str = "right") -> tuple[float, float] | None:
    primary = "right_wrist" if hand == "right" else "left_wrist"
    secondary = "left_wrist" if hand == "right" else "right_wrist"
    return _kp(kp, primary) or _kp(kp, secondary)


def classify_stroke(
    pose: PoseKeypoints,
    ball_xy: tuple[float, float] | None = None,
    role: str | None = None,
    hand: str = "right",
    is_first_shot: bool = False,
    court_y: float | None = None,
    wrist_history: list[tuple[int, float, float]] | None = None,
    torso_len: float | None = None,
    slice_drop_ratio: float = 0.3,
    volley_max_court_y: float = 4.5,
    slice_min_torso_px: float = 30.0,
) -> tuple[str, float]:
    """Classify the stroke type at ball contact.

    Rules (in priority order):
    - serve: first shot of a point; confidence rises when the ball/wrist is
      above the head at contact.
    - overhead: dominant wrist above the head mid-rally.
    - volley: hitter standing inside the service boxes (needs court position).
    - forehand / backhand: which side of the body the ball (or the racket
      wrist) is on. The far player faces the camera, so image-left is their
      right — the side test is mirrored for ``far_player`` and for left-handers.
    - slice: the racket wrist moved *down* over the look-back window before
      contact (high-to-low swing), measured in torso lengths.

    Args:
        pose: PoseKeypoints of the hitter at the contact frame.
        ball_xy: Ball pixel position at contact, if known.
        role: "near_player" or "far_player" (defaults to pose.role).
        hand: Hitter's handedness ("right"/"left").
        is_first_shot: True for the first hit of a point.
        court_y: Hitter's court-space y in metres (None if unknown).
        wrist_history: (frame, x, y) of the racket wrist over the look-back
            window ending at contact (chronological).
        torso_len: Shoulder-to-hip distance in pixels (for slice scaling).
        slice_drop_ratio: Wrist drop (in torso lengths) that marks a slice.
        volley_max_court_y: |court_y| below which a shot is a volley.

    Returns:
        (stroke, confidence).
    """
    kp = pose.keypoints
    role = role or pose.role
    nose = _kp(kp, "nose")
    l_shoulder, r_shoulder = _kp(kp, "left_shoulder"), _kp(kp, "right_shoulder")
    shoulder_y = None
    if l_shoulder and r_shoulder:
        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
    r_wrist, l_wrist = _kp(kp, "right_wrist"), _kp(kp, "left_wrist")
    body_x = _body_center_x(kp)

    head_y = nose[1] if nose else shoulder_y
    both_high = (
        r_wrist is not None and l_wrist is not None and head_y is not None
        and r_wrist[1] < head_y and l_wrist[1] < head_y
    )
    ball_over_head = ball_xy is not None and head_y is not None and ball_xy[1] < head_y
    any_high = (r_wrist is not None and head_y is not None and r_wrist[1] < head_y) or \
               (l_wrist is not None and head_y is not None and l_wrist[1] < head_y)

    behind_baseline = court_y is not None and abs(court_y) > _BASELINE_Y - 1.5
    over_body = (
        ball_xy is not None and body_x is not None
        and (torso_len is None or abs(ball_xy[0] - body_x) < 1.2 * torso_len)
    )
    if is_first_shot and ball_over_head and (any_high or both_high):
        return ("serve", 0.9)
    if ball_over_head and any_high and over_body and behind_baseline:
        return ("serve", 0.8)  # overhead contact above the body from the baseline is a serve, first hit or not
    if is_first_shot and both_high:
        return ("serve", 0.7)
    if ball_over_head and any_high and over_body:
        return ("overhead", 0.7)
    if court_y is not None and abs(court_y) < volley_max_court_y:
        return ("volley", 0.7)

    side: float | None = None
    if ball_xy is not None and body_x is not None:
        side = ball_xy[0] - body_x
    elif body_x is not None and (r_wrist is not None or l_wrist is not None):
        # No ball: the racket arm is the one reaching away from the body.
        # (Anatomical left/right labels are unreliable on back-facing
        # players, so pick by extension, not by name.)
        cands = [w for w in (r_wrist, l_wrist) if w is not None]
        ext = max(cands, key=lambda w: abs(w[0] - body_x))
        side = ext[0] - body_x
    if side is None:
        return ("forehand", 0.4)

    mirror = -1.0 if role == "far_player" else 1.0
    if hand == "left":
        mirror *= -1.0
    is_forehand = side * mirror > 0
    conf = 0.7 if ball_xy is not None else 0.6

    if wrist_history and len(wrist_history) >= 3 and torso_len and torso_len >= slice_min_torso_px:
        pre_y = wrist_history[0][2]
        contact_y = wrist_history[-1][2]
        drop = (contact_y - pre_y) / torso_len
        if drop > slice_drop_ratio:
            return ("slice", 0.6)

    return ("forehand" if is_forehand else "backhand", conf)


def infer_handedness(
    by_frame: dict[int, FrameTrackingResult],
    hits: list[Hit],
    role: str,
) -> str:
    """Infer a player's racket hand from pose at contact.

    Two cues, both weighted by how unambiguous they are:

    * **serve**: the racket arm is the one fully extended overhead, so the
      higher wrist is the dominant hand (weight 2; the toss arm is always
      lower at contact).
    * **one-handed contact**: when the wrists are far apart the racket
      wrist is the one clearly closer to the ball (weight 1). Two-handed
      backhands (wrists together) are skipped.

    Ties default to right.
    """
    votes = {"right": 0.0, "left": 0.0}
    for hit in hits:
        if hit.role != role:
            continue
        pose = _pose_for(by_frame.get(hit.frame), role)
        if pose is None:
            continue
        kp = pose.keypoints
        rw, lw = _kp(kp, "right_wrist"), _kp(kp, "left_wrist")
        nose = _kp(kp, "nose")
        torso = torso_length(kp)
        if rw is None or lw is None or not torso:
            continue
        # serve cue: the racket arm is the one reaching overhead. Pose models
        # often swap left/right on back-facing players, so use the *image side*
        # of the high wrist relative to the body (mirrored for the far player).
        body_x = _body_center_x(kp)
        player = _player_for(by_frame.get(hit.frame), role)
        behind = player is not None and player.court_position is not None and abs(player.court_position[1]) > _BASELINE_Y - 1.5
        serve_like = nose is not None and min(rw[1], lw[1]) < nose[1] and (behind or _first_hit_of_point(hits, hit))
        if serve_like and body_x is not None:
            if abs(rw[1] - lw[1]) > 0.5 * torso:
                high = rw if rw[1] < lw[1] else lw
                side = high[0] - body_x
                if abs(side) >= 0.35 * torso:  # otherwise the pose is too small to tell
                    mirror = -1.0 if role == "far_player" else 1.0
                    votes["right" if side * mirror > 0 else "left"] += 2.0
            continue
        if hit.ball_xy == (0.0, 0.0):
            continue
        if math.hypot(rw[0] - lw[0], rw[1] - lw[1]) < 0.6 * torso:
            continue  # two hands on the racket
        d_r = math.hypot(rw[0] - hit.ball_xy[0], rw[1] - hit.ball_xy[1])
        d_l = math.hypot(lw[0] - hit.ball_xy[0], lw[1] - hit.ball_xy[1])
        if max(d_r, d_l) < 1.5 * min(d_r, d_l):
            continue  # not a clear one-handed reach
        votes["right" if d_r < d_l else "left"] += 1.0
    return "left" if votes["left"] > votes["right"] else "right"


def _first_hit_of_point(hits: list[Hit], hit: Hit, gap_s_frames: int = 90) -> bool:
    """A hit is the first of its point if no earlier hit lies within ~3s."""
    return not any(0 < hit.frame - h.frame <= gap_s_frames for h in hits)


# ── Contact detection (legacy proximity method) ──────────────────────────────

def detect_contacts(
    tracking_results: list[FrameTrackingResult],
    fps: float,
    proximity_threshold: float = 100.0,
    min_frames_between_contacts: int = 10,
) -> list[tuple[int, str]]:
    """Detect frames where the ball is inside / near a player's bounding box.

    Superseded by :func:`court_vision.hit_detect.detect_hits`; kept as the
    ``contact_method: proximity`` option.
    """
    contacts: list[tuple[int, str]] = []
    last_contact_frame = -min_frames_between_contacts

    for result in tracking_results:
        if result.ball is None:
            continue
        if result.frame_index - last_contact_frame < min_frames_between_contacts:
            continue
        ball_x, ball_y = result.ball.x, result.ball.y
        for player in result.players:
            if player.role is None:
                continue
            in_bbox_x = player.bbox[0] <= ball_x <= player.bbox[2]
            in_bbox_y = player.bbox[1] <= ball_y <= player.bbox[3]
            if in_bbox_x and in_bbox_y:
                contacts.append((result.frame_index, player.role))
                last_contact_frame = result.frame_index
                break
            px = (player.bbox[0] + player.bbox[2]) / 2
            py = (player.bbox[1] + player.bbox[3]) / 2
            dist = ((ball_x - px) ** 2 + (ball_y - py) ** 2) ** 0.5
            if dist < proximity_threshold:
                contacts.append((result.frame_index, player.role))
                last_contact_frame = result.frame_index
                break

    return contacts


def split_hits_into_points(
    hits: list[Hit],
    segment: GameplaySegment,
    fps: float,
    gap_s: float = 4.0,
    pad_before_s: float = 1.0,
    pad_after_s: float = 2.5,
) -> list[tuple[int, int, list[Hit]]]:
    """Split a segment's hits into points at long gaps between hits.

    Returns (start_frame, end_frame, hits) per point. A segment without hits
    yields one empty point so review can still see it.
    """
    if not hits:
        return [(segment.start_frame, segment.end_frame, [])]
    gap = int(gap_s * fps)
    groups: list[list[Hit]] = [[hits[0]]]
    for h in hits[1:]:
        if h.frame - groups[-1][-1].frame > gap:
            groups.append([h])
        else:
            groups[-1].append(h)

    pad_b, pad_a = int(pad_before_s * fps), int(pad_after_s * fps)
    out: list[tuple[int, int, list[Hit]]] = []
    for gi, g in enumerate(groups):
        start = max(segment.start_frame, g[0].frame - pad_b)
        end = min(segment.end_frame, g[-1].frame + pad_a)
        if gi == 0:
            start = segment.start_frame
        if gi == len(groups) - 1:
            end = segment.end_frame
        if out and start <= out[-1][1]:
            mid = (out[-1][2][-1].frame + g[0].frame) // 2
            out[-1] = (out[-1][0], mid, out[-1][2])
            start = mid + 1
        out.append((start, end, g))
    return out


# ── Bounce / placement ───────────────────────────────────────────────────────

def _drop_stuck(samples, court, fps: float, min_len_s: float = 0.12, jitter_px: float = 1.5, fast_px: float = 3.0):
    """Remove stretches where the detection sits still while the ball is
    fast on both sides — the net band (and logos) catch the detector while
    the real ball passes behind them, and a stuck run would look like a
    bounce (a local maximum of image y)."""
    n = len(samples)
    if n < 5:
        return samples, court
    xy = np.array([(b.x, b.y) for _, b in samples], dtype=np.float64)
    frames = np.array([f for f, _ in samples], dtype=np.float64)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(frames), 1)
    still = step < jitter_px
    min_len = max(3, int(min_len_s * fps))
    drop = np.zeros(n, dtype=bool)
    i = 0
    while i < len(still):
        if not still[i]:
            i += 1
            continue
        j = i
        while j < len(still) and still[j]:
            j += 1
        # samples i..j form a stuck stretch (still[k] links k and k+1)
        if j - i >= min_len:
            before = step[i - 1] if i > 0 else np.inf
            after = step[j] if j < len(step) else np.inf
            if before > fast_px and after > fast_px:
                drop[i:j + 1] = True
        i = j
    keep = [k for k in range(n) if not drop[k]]
    return [samples[k] for k in keep], [court[k] for k in keep]


def estimate_bounce(
    by_frame: dict[int, FrameTrackingResult],
    start: int,
    end: int,
    homography: np.ndarray | None,
    fps: float = 30.0,
    hitter_role: str | None = None,
    net_margin_m: float = 2.0,
    min_prominence_m: float = 1.0,
) -> tuple[float, float] | None:
    """Estimate where the ball lands after the hit at ``start``.

    Projecting an airborne ball through the ground homography overshoots:
    the higher the ball, the deeper it appears. So along a flight the
    projected depth |y| runs ahead, falls back as the ball descends, is
    closest to the truth at the bounce, and runs away again as the ball
    rises off the ground. The landing is therefore the first local
    *minimum* of projected depth (with at least ``min_prominence_m`` of
    fall-back before it) on the opponent's side of the net. Returns
    court-space (x, y) in metres, or None if no bounce is visible before
    ``end`` (ball still in flight when the clip ends).
    """
    if homography is None:
        return None
    skip = int(0.15 * fps)
    samples = [
        (f, by_frame[f].ball) for f in range(start + skip, end + 1)
        if f in by_frame and by_frame[f].ball is not None
    ]
    if len(samples) < 5:
        return None
    court = [map_ball_to_court(b, homography) for _, b in samples]
    opp_sign = None
    if hitter_role is not None:
        opp_sign = 1.0 if hitter_role == "near_player" else -1.0
    keep = [
        i for i, c in enumerate(court)
        if c is not None and abs(c[1]) > net_margin_m and (opp_sign is None or c[1] * opp_sign > 0)
    ]
    if len(keep) < 5:
        return None
    court = [court[i] for i in keep]
    depth = np.array([abs(c[1]) for c in court], dtype=np.float64)
    k = max(1, int(round(fps / 30.0)))
    if len(depth) > 2 * k + 1:
        padded = np.pad(depth, k, mode="edge")
        depth = np.convolve(padded, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    win = max(2, int(round(fps / 10.0)))
    n = len(depth)
    for i in range(win, n - win):
        left = depth[i - win:i]
        right = depth[i + 1:i + 1 + win]
        if depth[i] > left.min() or depth[i] > right.min():
            continue
        if depth[:i].max() - depth[i] < min_prominence_m:
            continue  # no real fall-back before it (apex plateau, jitter)
        if right.max() - depth[i] < 0.3:
            continue  # not rising again yet
        x, y = court[i]
        if abs(y) > _BASELINE_Y + 2.0 or abs(x) > 7.0:
            continue  # a ball cannot land there (behind the back fence): still airborne
        return court[i]
    return None


# ── Match assembly ───────────────────────────────────────────────────────────

def _pose_for(t: FrameTrackingResult | None, role: str) -> PoseKeypoints | None:
    if t is None:
        return None
    return next((p for p in t.poses if p.role == role), None)


def _player_for(t: FrameTrackingResult | None, role: str) -> PlayerDetection | None:
    if t is None:
        return None
    return next((p for p in t.players if p.role == role), None)


def _wrist_history(
    by_frame: dict[int, FrameTrackingResult],
    role: str,
    hand: str,
    contact_frame: int,
    lookback: int,
) -> tuple[list[tuple[int, float, float]], float | None]:
    """Racket-wrist path over the look-back window ending at contact.

    The racket wrist is taken as the wrist farther from the body centre *at
    contact* (pose models often swap left/right on back-facing players, so
    the anatomical label is not trusted); its keypoint name is then followed
    back in time.
    """
    contact_pose = _pose_for(by_frame.get(contact_frame), role)
    name = "right_wrist" if hand == "right" else "left_wrist"
    if contact_pose is not None:
        kp = contact_pose.keypoints
        body_x = _body_center_x(kp)
        rw, lw = _kp(kp, "right_wrist"), _kp(kp, "left_wrist")
        if body_x is not None and rw is not None and lw is not None:
            name = "right_wrist" if abs(rw[0] - body_x) >= abs(lw[0] - body_x) else "left_wrist"
    hist: list[tuple[int, float, float]] = []
    torsos: list[float] = []
    for f in range(contact_frame - lookback, contact_frame + 1):
        pose = _pose_for(by_frame.get(f), role)
        if pose is None:
            continue
        w = _kp(pose.keypoints, name)
        if w is not None:
            hist.append((f, w[0], w[1]))
        tl = torso_length(pose.keypoints)
        if tl:
            torsos.append(tl)
    return hist, (float(np.median(torsos)) if torsos else None)


def _infer_outcome(
    point_hits: list[Hit],
    start: int,
    end: int,
    next_start: int | None,
    by_frame: dict[int, FrameTrackingResult],
    homography: np.ndarray | None,
    fps: float,
    settings: PipelineSettings,
    scoreboard: ScoreTimeline | None,
) -> tuple[str | None, str]:
    """Return (winner_role, source)."""
    if not point_hits:
        return None, "none"
    last = point_hits[-1]
    server = point_hits[0].role
    method = settings.outcome_method

    if scoreboard is not None and method in ("auto", "scoreboard"):
        row = infer_point_winner_row(scoreboard, start, end, next_start, fps)
        if row is not None:
            before = scoreboard.stable_state(start, min(end, start + int(3 * fps))) or scoreboard.sample_at(start)
            serving_rows = [i for i, r in enumerate(before.rows) if r.serving] if before else []
            if len(serving_rows) == 1:
                serving_row = serving_rows[0]
                winner = server if row == serving_row else other_role(server)
                return winner, "scoreboard"

    if method in ("auto", "trajectory") and homography is not None:
        # Where did the ball go after the last hit?
        samples = [
            by_frame[f].ball for f in range(last.frame + 1, end + 1)
            if f in by_frame and by_frame[f].ball is not None and not by_frame[f].ball.interpolated
        ]
        if len(samples) >= 3:
            hitter_sign = -1.0 if last.role == "near_player" else 1.0
            court_pts = [map_ball_to_court(b, homography) for b in samples]
            court_pts = [p for p in court_pts if p is not None]
            crossed = any(p[1] * hitter_sign < -0.5 for p in court_pts)
            if not crossed:
                return other_role(last.role), "trajectory"  # net / never crossed
            landing = estimate_bounce(by_frame, last.frame, end, homography, fps, hitter_role=last.role)
            if landing is not None:
                inside = abs(landing[0]) <= _SINGLES_WIDTH + 0.6 and abs(landing[1]) <= _BASELINE_Y + 0.6
                return (last.role if inside else other_role(last.role)), "trajectory"

    if method == "last_hitter":
        return last.role, "last_hitter"
    return None, "unknown"


def build_match_data(
    source: str,
    segments: list[GameplaySegment],
    tracking_results: list[FrameTrackingResult],
    fps: float,
    court_homography: np.ndarray | None = None,
    settings: PipelineSettings | None = None,
    scoreboard: ScoreTimeline | None = None,
    segment_homographies: list[np.ndarray | None] | None = None,
) -> MatchData:
    """Build complete match data from tracking results and segments.

    Orchestrates hit detection, point splitting, stroke classification,
    placement and outcome inference.
    """
    settings = settings or PipelineSettings()
    match_id = hashlib.md5(source.encode()).hexdigest()[:12]
    by_frame: dict[int, FrameTrackingResult] = {t.frame_index: t for t in tracking_results}
    lookback = max(1, int(settings.slice_lookback_s * fps))

    # 1. hits + point boundaries per segment
    raw_points: list[tuple[int, int, list[Hit], np.ndarray | None]] = []
    for i, seg in enumerate(segments):
        seg_tracking = [t for t in tracking_results if seg.start_frame <= t.frame_index <= seg.end_frame]
        H = court_homography
        if segment_homographies and i < len(segment_homographies) and segment_homographies[i] is not None:
            H = segment_homographies[i]
        if settings.contact_method == "trajectory":
            hits = detect_hits(
                seg_tracking, fps,
                min_gap_s=settings.contact_min_gap_s,
                player_margin=settings.contact_player_margin,
                min_turn_deg=settings.contact_min_turn_deg,
                min_speed_px=settings.contact_min_speed_px,
                homography=H,
                use_swing=settings.contact_use_swing,
                swing_min=settings.contact_swing_min,
                min_speed_norm=settings.contact_min_speed_norm,
                max_speed_px=settings.ball_max_speed_px,
                frame_height=int(settings.target_resolution[1]),
                close_up_ratio=settings.close_up_ratio,
            )
        else:
            hits = [
                Hit(frame=f, role=r, score=1.0,
                    ball_xy=((by_frame[f].ball.x, by_frame[f].ball.y) if f in by_frame and by_frame[f].ball else (0.0, 0.0)),
                    turn_deg=0.0, speed_pre=0.0, speed_post=0.0, kind="proximity")
                for f, r in detect_contacts(
                    seg_tracking, fps,
                    proximity_threshold=settings.proximity_threshold,
                    min_frames_between_contacts=settings.min_frames_between_contacts,
                )
            ]
        for start, end, group in split_hits_into_points(
            hits, seg, fps,
            gap_s=settings.point_split_gap_s,
            pad_before_s=settings.point_pad_before_s,
            pad_after_s=settings.point_pad_after_s,
        ):
            raw_points.append((start, end, group, H))

    # 2. handedness (auto from the racket wrist at contact)
    all_hits = [h for _, _, group, _ in raw_points for h in group]
    hands = {}
    for role, setting in (("near_player", settings.near_player_hand), ("far_player", settings.far_player_hand)):
        hands[role] = infer_handedness(by_frame, all_hits, role) if setting == "auto" else setting

    # 3. per point: strokes, placement, outcome
    points: list[Point] = []
    for idx, (start, end, hits, H) in enumerate(raw_points):
        next_start = raw_points[idx + 1][0] if idx + 1 < len(raw_points) else None
        shots: list[Shot] = []
        for shot_num, hit in enumerate(hits, 1):
            t = by_frame.get(hit.frame)
            role = hit.role
            hand = hands[role]
            pose = _pose_for(t, role)
            player = _player_for(t, role)
            court_y = player.court_position[1] if (player and player.court_position) else None

            stroke, stroke_conf = "forehand", 0.4
            if pose is not None:
                hist, torso = _wrist_history(by_frame, role, hand, hit.frame, lookback)
                stroke, stroke_conf = classify_stroke(
                    pose, ball_xy=hit.ball_xy, role=role, hand=hand,
                    is_first_shot=(shot_num == 1), court_y=court_y,
                    wrist_history=hist, torso_len=torso,
                    slice_drop_ratio=settings.slice_drop_ratio,
                    volley_max_court_y=settings.volley_max_court_y,
                    slice_min_torso_px=settings.slice_min_torso_px,
                )
            elif shot_num == 1:
                stroke, stroke_conf = "serve", 0.5

            placement = None
            if H is not None:
                nxt = hits[shot_num].frame if shot_num < len(hits) else min(end, hit.frame + int(2.5 * fps))
                landing = estimate_bounce(by_frame, hit.frame, nxt, H, fps, hitter_role=role)
                if landing is not None:
                    zone = compute_placement_zone(
                        x=landing[0], y=landing[1], is_serve=(stroke == "serve"), hitter=role,
                    )
                    placement = ShotPlacement(x=landing[0], y=landing[1], zone=zone)

            shots.append(Shot(
                shot_number=shot_num,
                frame=hit.frame,
                time_s=hit.frame / fps,
                player=role,
                stroke=stroke,
                placement=placement,
                confidence=stroke_conf,
            ))

        # A point starts with the serve: whatever the hit detector saw in the
        # seconds before it (ball bounced by the server, the toss) is not a
        # stroke.
        serve_idx = next((i for i, s in enumerate(shots) if s.stroke == "serve"), None)
        if serve_idx and shots[serve_idx].frame - shots[0].frame <= int(6 * fps):
            hits = hits[serve_idx:]
            shots = shots[serve_idx:]
            for k, s in enumerate(shots, 1):
                s.shot_number = k

        winner, source_tag = _infer_outcome(hits, start, end, next_start, by_frame, H, fps, settings, scoreboard)
        outcome = None
        outcome_player = None
        if shots and winner is not None:
            last_role = shots[-1].player
            outcome = "winner" if winner == last_role else "error"
            outcome_player = last_role

        points.append(Point(
            point_number=idx + 1,
            start_frame=start,
            end_frame=end,
            start_time_s=start / fps,
            end_time_s=end / fps,
            server=shots[0].player if shots else None,
            shots=shots,
            outcome=outcome,
            outcome_player=outcome_player,
            rally_length=len(shots),
            winner=winner,
            outcome_source=source_tag,
        ))

    return MatchData(
        match_id=match_id,
        source_url=source,
        metadata={
            "players": ["near_player", "far_player"],
            "date_processed": str(date.today()),
            "fps": fps,
            "handedness": hands,
            "hits": [
                {"frame": h.frame, "role": h.role, "kind": h.kind, "score": round(h.score, 3),
                 "turn_deg": round(h.turn_deg, 1), "speed_pre": round(h.speed_pre, 1),
                 "speed_post": round(h.speed_post, 1)}
                for h in all_hits
            ],
        },
        points=points,
    )
