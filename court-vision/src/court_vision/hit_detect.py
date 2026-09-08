"""Hit (racket contact) detection from ball trajectory + player pose.

Two complementary candidate sources:

* **trajectory**: the ball's direction of travel changes sharply (or its
  speed jumps, as on a serve) while it sits inside a player's expanded
  bounding box. Bounces bend the trajectory too, so every candidate must
  also pass an *away test*: after the candidate the ball moves away from
  the hitter's side of the court (toward the opponent), which a bounce in
  front of the player does not do.
* **swing**: a sharp peak in the hitter's wrist speed. This catches hits
  where the ball is invisible to the detector (far baseline, high lobs).

Candidates are merged with non-maximum suppression in time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from court_vision.player_detect import FrameTrackingResult, PlayerDetection, PoseKeypoints


@dataclass
class Hit:
    frame: int
    role: str
    score: float
    ball_xy: tuple[float, float]
    turn_deg: float
    speed_pre: float
    speed_post: float
    kind: str  # "turn" | "start" | "burst" | "swing" | "proximity"
    track_end: int = -1  # last frame of the continuous ball track after the hit (-1 = n/a)


def _expanded_bbox(p: PlayerDetection, margin: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = p.bbox
    w, h = x2 - x1, y2 - y1
    return (x1 - w * margin, y1 - h * margin, x2 + w * margin, y2 + h * margin)


def _dist_to_bbox(x: float, y: float, bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _nearest_player(
    t: FrameTrackingResult,
    x: float,
    y: float,
    margin: float,
) -> tuple[str | None, float]:
    """Return (role, distance-to-expanded-bbox) of the closest role-assigned player."""
    best_role, best_d = None, math.inf
    for p in t.players:
        if p.role is None:
            continue
        d = _dist_to_bbox(x, y, _expanded_bbox(p, margin))
        if d < best_d:
            best_role, best_d = p.role, d
    return best_role, best_d


# Sign conventions: ``delta * sign(role) > 0`` means the ball moves *toward*
# the hitter along that depth axis.
_IMG_SIGN = {"near_player": 1.0, "far_player": -1.0}     # image y grows toward the near baseline
_COURT_SIGN = {"near_player": -1.0, "far_player": 1.0}   # court y grows toward the far baseline


def _court_depth_fn(homography: np.ndarray):
    def court_y(x: float, y: float) -> float:
        p = homography @ np.array([x, y, 1.0])
        if abs(p[2]) < 1e-9:
            return 0.0
        return float(p[1] / p[2])
    return court_y


def detect_hits(
    tracking: list[FrameTrackingResult],
    fps: float,
    min_gap_s: float = 0.45,
    player_margin: float = 0.6,
    min_turn_deg: float = 40.0,
    min_speed_px: float = 2.0,
    burst_ratio: float = 3.0,
    homography: np.ndarray | None = None,
    use_swing: bool = True,
    swing_min: float = 0.12,
    min_speed_norm: float = 0.03,
    max_speed_px: float = 150.0,
    trace: dict[int, str] | None = None,
) -> list[Hit]:
    """Detect racket contacts from ball motion + player proximity + pose.

    Args:
        tracking: Per-frame tracking results (sorted by frame).
        fps: Video frame rate.
        min_gap_s: Minimum time between two hits.
        player_margin: Expansion ratio of a player's bbox used as the hit gate.
        min_turn_deg: Minimum direction change to count as a hit candidate.
        min_speed_px: Minimum post-hit speed (px per 30fps-frame).
        burst_ratio: Speed jump ratio (post/pre) that counts as a hit even
            without a direction change (far-side serves).
        homography: Court homography; enables the court-space away test.
        use_swing: Also generate candidates from wrist-speed peaks.
        swing_min: Wrist speed threshold (bbox heights per 30fps-frame).

    Returns:
        Hits sorted by frame.
    """
    tracking = sorted(tracking, key=lambda t: t.frame_index)
    by_frame = {t.frame_index: t for t in tracking}
    scale = max(fps, 1.0) / 30.0  # px/frame -> px per 30fps-frame (60fps frames move half as far)
    w = max(2, int(round(fps / 15.0)))  # velocity window (frames)
    max_df = max(2 * w, int(round(fps / 3.0)))  # tolerated sample gap for velocity
    post_w = max(w, int(round(fps / 4.0)))  # away-test window
    min_gap = max(1, int(min_gap_s * fps))
    court_depth = _court_depth_fn(homography) if homography is not None else None

    # Judge hits on real detections only: interpolation draws straight lines
    # through gaps and would hide the very reversals (and track breaks) that
    # mark a contact.
    pts = [(t.frame_index, t.ball.x, t.ball.y) for t in tracking if t.ball is not None and not t.ball.interpolated]
    candidates: list[Hit] = []

    if len(pts) >= 3:
        frames = np.array([p[0] for p in pts], dtype=np.int64)
        xy = np.array([(p[1], p[2]) for p in pts], dtype=np.float64)
        # Two depth axes for the away test. Image y is robust for fast, flat
        # balls (a rising toss also looks "away", the hover test below
        # catches it); court-space depth handles lobs, whose image y first
        # moves the wrong way, but over-shoots for high balls.
        dep_img = xy[:, 1].copy()
        dep_court = (np.array([court_depth(p[1], p[2]) for p in pts], dtype=np.float64)
                     if court_depth is not None else None)
        n = len(pts)
        # Split the samples into physically continuous runs: a gap in time or
        # a jump faster than a tennis ball can move starts a new run, and no
        # velocity or away-test window may straddle runs.
        cap = max_speed_px * 30.0 / max(fps, 1.0)
        run_id = np.zeros(n, dtype=np.int64)
        for i in range(1, n):
            df = frames[i] - frames[i - 1]
            jump = np.linalg.norm(xy[i] - xy[i - 1]) > cap * max(df, 1)
            run_id[i] = run_id[i - 1] + (1 if (df > max_df or jump) else 0)
        run_last: dict[int, int] = {}
        for i in range(n):
            run_last[int(run_id[i])] = int(frames[i])

        def vel(i: int, j: int) -> tuple[np.ndarray, float] | None:
            df = frames[j] - frames[i]
            if df <= 0 or df > max_df or run_id[i] != run_id[j]:
                return None
            v = (xy[j] - xy[i]) / df
            return v, float(np.linalg.norm(v) * scale)

        window_end: dict[int, int] = {}

        def window_delta(k: int, direction: int, strict: bool = True, dep: np.ndarray = dep_img) -> float | None:
            """Depth change over the away-test window after (+1) / before (-1) k.

            With ``strict`` the window must be *continuous* (no sample gap
            larger than max_df) and span at least 80% of post_w: a real hit
            is followed by an unbroken stretch of the ball flying away,
            whereas a toss peak, a bouncing ball or a false peak on a shoe
            leaves only a few frames of motion before the track breaks.
            """
            j = k
            lim = frames[k] + direction * post_w
            while 0 <= j + direction < n and (frames[j + direction] - lim) * direction <= 0:
                if run_id[j + direction] != run_id[k]:
                    break
                j += direction
            if j == k:
                return None
            span = abs(frames[j] - frames[k])
            if strict and span < 0.8 * post_w:
                return None
            if direction > 0:
                window_end[k] = j
            return float(dep[j] - dep[k]) * direction

        def note(f: int, why: str) -> None:
            if trace is not None and f in trace:
                trace[f] = why

        for k in range(n):
            f = int(frames[k])
            t = by_frame.get(f)
            if t is None:
                continue
            role, d = _nearest_player(t, xy[k][0], xy[k][1], player_margin)
            if role is None or d > 0.0:
                note(f, f"gate: ball {d:.0f}px outside every player bbox")
                continue  # gate: ball must be inside an expanded player bbox
            hitter = next((p for p in t.players if p.role == role), None)
            bbox_h = max(hitter.bbox[3] - hitter.bbox[1], 1.0) if hitter else 100.0
            # perspective-invariant speed floor: a real hit sends the ball at
            # least this many body-heights per frame, near or far
            min_sp = max(min_speed_px, min_speed_norm * bbox_h)

            i_pre = k
            while i_pre > 0 and frames[k] - frames[i_pre] < w:
                i_pre -= 1
            i_post = k
            while i_post < n - 1 and frames[i_post] - frames[k] < w:
                i_post += 1
            pre = vel(i_pre, k) if i_pre < k else None
            post = vel(k, i_post) if i_post > k else None
            sp_pre = pre[1] if pre else 0.0
            sp_post = post[1] if post else 0.0
            track_start = k == 0 or run_id[k] != run_id[k - 1]

            kind = None
            turn = 0.0
            burst = False
            if pre and post and sp_pre > 1e-6 and sp_post > 1e-6:
                cos = float(np.dot(pre[0], post[0]) / (np.linalg.norm(pre[0]) * np.linalg.norm(post[0]) + 1e-9))
                turn = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
                burst = sp_post / max(sp_pre, 1e-6) >= burst_ratio
                if turn >= min_turn_deg and max(sp_pre, sp_post) >= min_sp:
                    kind = "turn"
                elif sp_post >= min_sp and burst:
                    kind = "burst"
            if kind is None and track_start and post and sp_post >= min_sp:
                kind = "start"
                turn = 180.0
            if kind is None:
                note(f, f"no kind: turn={turn:.0f} pre={sp_pre:.1f} post={sp_post:.1f} min_sp={min_sp:.1f} start={track_start} burst={burst}")
                continue

            # Away test: after the candidate the ball must move toward the opponent.
            d_post = window_delta(k, +1, strict=True, dep=dep_img)
            if d_post is None:
                note(f, f"{kind}: post window not continuous / too short")
                continue  # no continuous flight away from the hitter (toss peak, shoe, bounce)
            away_img = d_post * _IMG_SIGN[role] <= -0.12 * bbox_h
            away_court = False
            if dep_court is not None:
                d_post_c = window_delta(k, +1, strict=True, dep=dep_court)
                away_court = d_post_c is not None and d_post_c * _COURT_SIGN[role] <= -1.0
            if not (away_img or away_court):
                note(f, f"{kind}: not away (img d={d_post:.0f}px, need {-0.12 * bbox_h * _IMG_SIGN[role]:.0f})")
                continue  # not moving away from the hitter -> bounce / approach
            d_pre = window_delta(k, -1, strict=False, dep=dep_img)
            sgn = _IMG_SIGN[role]
            away_min = 0.12 * bbox_h
            # A ball that is still hovering around the hitter at the end of
            # the window was tossed or bounced, not struck.
            j_end = window_end.get(k, k)
            t_end = by_frame.get(int(frames[j_end]))
            hitter_end = next((p for p in (t_end.players if t_end else []) if p.role == role), hitter)
            if hitter_end is not None and _dist_to_bbox(
                float(xy[j_end][0]), float(xy[j_end][1]), _expanded_bbox(hitter_end, player_margin / 2)
            ) == 0.0:
                note(f, f"{kind}: ball still hovering at hitter after window")
                continue
            toward_pre = d_pre is None or d_pre * sgn > -away_min * 0.5
            if not (toward_pre or burst or track_start):
                note(f, f"{kind}: ball was not approaching before (d_pre={d_pre})")
                continue
            note(f, f"CANDIDATE {kind} turn={turn:.0f} pre={sp_pre:.1f} post={sp_post:.1f}")

            base = {"turn": 0.6, "burst": 0.3, "start": 0.2}[kind]
            score = base + turn / 180.0 + 0.3 * min(sp_post / 10.0, 1.0)
            candidates.append(Hit(
                frame=f, role=role, score=score, ball_xy=(float(xy[k][0]), float(xy[k][1])),
                turn_deg=turn, speed_pre=sp_pre, speed_post=sp_post, kind=kind,
                track_end=run_last[int(run_id[k])],
            ))

    if use_swing:
        def ball_leaves(frame: int, role: str) -> bool | None:
            """Away test at the ball sample nearest ``frame`` (None = no ball)."""
            if len(pts) < 3:
                return None
            k = int(np.argmin(np.abs(frames - frame)))
            if abs(int(frames[k]) - frame) > post_w:
                return None
            t_k = by_frame.get(int(frames[k]))
            hitter = next((p for p in (t_k.players if t_k else []) if p.role == role), None)
            bbox_h = max(hitter.bbox[3] - hitter.bbox[1], 1.0) if hitter else 100.0
            d = window_delta(k, +1, strict=True, dep=dep_img)
            if d is not None and d * _IMG_SIGN[role] <= -0.12 * bbox_h:
                return True
            if dep_court is not None:
                d_c = window_delta(k, +1, strict=True, dep=dep_court)
                if d_c is not None and d_c * _COURT_SIGN[role] <= -1.0:
                    return True
            return False

        candidates.extend(_swing_candidates(tracking, fps, swing_min, player_margin, by_frame, ball_leaves))

    # A "start" candidate is the ball re-appearing *after* a far-side hit; if
    # a turn/swing candidate for the same player sits just before it (or at
    # most a few frames after — the swing onset lags a little), that one
    # carries the real contact time.
    stub = int(round(fps * 0.5))

    def real_flight(o: Hit) -> bool:
        # a turn whose ball track ends within half a second is a toss / blip
        return not (o.kind in ("turn", "burst", "start") and 0 <= o.track_end - o.frame <= stub)

    refined: list[Hit] = []
    for c in candidates:
        if c.kind == "start" and any(
            o.kind in ("turn", "swing", "burst") and o.role == c.role
            and -2 * min_gap <= o.frame - c.frame <= min_gap // 2 and real_flight(o)
            for o in candidates
        ):
            continue
        refined.append(c)

    # Non-maximum suppression in time.
    refined.sort(key=lambda h: h.score, reverse=True)
    kept: list[Hit] = []
    for c in refined:
        if all(abs(c.frame - k.frame) >= min_gap for k in kept):
            kept.append(c)
    kept.sort(key=lambda h: h.frame)
    return enforce_alternation(kept, fps)


def enforce_alternation(hits: list[Hit], fps: float, max_same_role_gap_s: float = 1.5) -> list[Hit]:
    """Players alternate hits: two hits by the same player closer than
    ``max_same_role_gap_s`` with no opponent hit in between are the same
    stroke seen twice — keep the better-scored one."""
    gap = int(max_same_role_gap_s * fps)
    ball_kinds = ("turn", "burst", "start")
    # Direct evidence of contact (a reversal) beats a track merely starting
    # (the ball re-appearing, or a toss leaving the hand); swings are timed
    # from noisy wrists and rank lowest.
    rank = {"turn": 2, "burst": 1, "start": 0, "swing": 0, "proximity": 0}
    out: list[Hit] = []
    for h in sorted(hits, key=lambda h: h.frame):
        if out and out[-1].role == h.role and h.frame - out[-1].frame < gap:
            prev = out[-1]
            # The ball leaves the racket exactly once and then flies
            # continuously. If the earlier candidate's track broke before the
            # later one started (toss peak -> serve, or a re-acquired track),
            # the later candidate is the real contact.
            stub = int(round(fps * 0.5))
            broke = (prev.kind in ball_kinds and h.kind in ball_kinds
                     and 0 <= prev.track_end < h.frame - 1
                     and prev.track_end - prev.frame <= stub      # earlier flight was a stub (toss)
                     and h.track_end - h.frame > stub)             # later one really flies
            if broke or rank[h.kind] > rank[prev.kind] or (rank[h.kind] == rank[prev.kind] and h.score > prev.score):
                out[-1] = h
            continue
        out.append(h)
    return out


def _wrist_speed_series(
    tracking: list[FrameTrackingResult],
    role: str,
    fps: float,
) -> list[tuple[int, float]]:
    """(frame, speed) of the fastest wrist, in bbox-heights per 30fps-frame.

    Poses on non-detection frames are copies of the previous detection, so
    identical consecutive keypoints are skipped rather than counted as zero
    motion.
    """
    scale = max(fps, 1.0) / 30.0
    series: list[tuple[int, float]] = []
    prev: tuple[int, dict, float] | None = None
    for t in tracking:
        pose: PoseKeypoints | None = next((p for p in t.poses if p.role == role), None)
        player = next((p for p in t.players if p.role == role), None)
        if pose is None or player is None:
            continue
        h = max(player.bbox[3] - player.bbox[1], 1.0)
        kp = pose.keypoints
        if prev is not None and prev[1] is kp:
            continue
        if prev is not None and prev[1].get("right_wrist") == kp.get("right_wrist") \
                and prev[1].get("left_wrist") == kp.get("left_wrist"):
            continue
        if prev is not None:
            df = t.frame_index - prev[0]
            best = 0.0
            for name in ("right_wrist", "left_wrist"):
                a, b = prev[1].get(name), kp.get(name)
                if a is None or b is None or a[2] < 0.3 or b[2] < 0.3:
                    continue
                sp = math.hypot(b[0] - a[0], b[1] - a[1]) / df / h * scale
                best = max(best, sp)
            series.append((t.frame_index, best))
        prev = (t.frame_index, kp, h)
    return series


def _swing_candidates(
    tracking: list[FrameTrackingResult],
    fps: float,
    swing_min: float,
    player_margin: float,
    by_frame: dict[int, FrameTrackingResult],
    ball_leaves=None,
) -> list[Hit]:
    out: list[Hit] = []
    support = max(2, int(round(fps / 5.0)))  # frames around the peak to look for the ball
    for role in ("near_player", "far_player"):
        series = _wrist_speed_series(tracking, role, fps)
        if len(series) < 3:
            continue
        speeds = np.array([s for _, s in series])
        sm = np.convolve(speeds, np.ones(3) / 3.0, mode="same")
        frames_s = [f for f, _ in series]
        half_gap = max(2, int(round(fps * 0.2)))
        for i in range(1, len(series) - 1):
            if sm[i] < swing_min:
                continue
            f = frames_s[i]
            # must be the maximum within +-0.2s
            lo = i
            while lo > 0 and f - frames_s[lo - 1] <= half_gap:
                lo -= 1
            hi = i
            while hi < len(series) - 1 and frames_s[hi + 1] - f <= half_gap:
                hi += 1
            if sm[i] < sm[lo:hi + 1].max():
                continue
            # The wrist is fastest on the follow-through; contact is where the
            # speed starts rising. Walk back to the onset (half the peak).
            onset = i
            while onset > lo and sm[onset - 1] >= 0.5 * sm[i] and sm[onset - 1] <= sm[onset]:
                onset -= 1
            f = frames_s[onset]
            # ball support: any ball sample inside the expanded bbox nearby
            ball_xy = None
            for df in sorted(range(-support, support + 1), key=abs):
                t = by_frame.get(f + df)
                if t is None or t.ball is None:
                    continue
                r, d = _nearest_player(t, t.ball.x, t.ball.y, player_margin)
                if r == role and d == 0.0:
                    ball_xy = (t.ball.x, t.ball.y)
                    break
            # if the ball is tracked here but clearly elsewhere, it's not a hit
            near_ball = [by_frame[f + df].ball for df in range(-support, support + 1)
                         if (f + df) in by_frame and by_frame[f + df].ball is not None]
            if ball_xy is None and near_ball:
                continue
            if ball_xy is None and sm[i] < 2.5 * swing_min:
                continue  # wrist noise alone is not enough without a ball
            if ball_xy is not None and ball_leaves is not None and ball_leaves(f, role) is False:
                continue  # ball is here but does not fly away (pre-serve bounces)
            score = 0.3 + 0.3 * min(sm[i] / (2 * swing_min), 1.0) + (0.3 if ball_xy else 0.0)
            out.append(Hit(
                frame=f, role=role, score=score,
                ball_xy=ball_xy if ball_xy else (0.0, 0.0),
                turn_deg=0.0, speed_pre=0.0, speed_post=float(sm[i]), kind="swing",
            ))
    return out
