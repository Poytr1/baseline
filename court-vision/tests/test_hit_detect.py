"""Tests for racket-contact (hit) detection from ball trajectory + pose."""

import numpy as np
import pytest

from court_vision.ball_tracker import BallDetection
from court_vision.hit_detect import (
    Hit,
    _dist_to_bbox,
    _expanded_bbox,
    _nearest_player,
    detect_hits,
    enforce_alternation,
)
from court_vision.player_detect import FrameTrackingResult, PlayerDetection, PoseKeypoints

FPS = 30.0
NEAR_BBOX = (500.0, 400.0, 600.0, 700.0)  # near player, bottom of frame
FAR_BBOX = (600.0, 100.0, 660.0, 220.0)  # far player, top of frame

# Linear pixel->court homography: court_x = (px - 640) / 50, court_y = -(py - 360) / 25
# (larger image y == closer to the camera == negative court y).
H_LINEAR = np.array([[1 / 50, 0.0, -12.8], [0.0, -1 / 25, 14.4], [0.0, 0.0, 1.0]])


def _player(i: int, role: str, bbox=None) -> PlayerDetection:
    bbox = bbox or (NEAR_BBOX if role == "near_player" else FAR_BBOX)
    return PlayerDetection(frame_index=i, bbox=bbox, confidence=0.9, role=role)


def _frame(i: int, ball_xy, players, poses=()) -> FrameTrackingResult:
    ball = None if ball_xy is None else BallDetection(i, float(ball_xy[0]), float(ball_xy[1]), 0.9)
    return FrameTrackingResult(frame_index=i, ball=ball, players=list(players), poses=list(poses))


def _both(i: int) -> list[PlayerDetection]:
    return [_player(i, "near_player"), _player(i, "far_player")]


def _hit(frame: int, role: str, score: float = 1.0, kind: str = "turn") -> Hit:
    return Hit(frame=frame, role=role, score=score, ball_xy=(0.0, 0.0), turn_deg=0.0,
               speed_pre=0.0, speed_post=0.0, kind=kind)


def _near_hit_ys() -> list[int]:
    """Ball drops toward the near player and reverses inside its bbox at frame 10.

    40 px/frame both ways: fast enough to have left the near player's bbox
    (half-expanded, top at y=310) by the end of the 8-frame away window, and
    symmetric so the reversal is a pure turn (no speed burst) at frame 10.
    """
    return [150 + 40 * i for i in range(11)] + [550 - 40 * (i - 10) for i in range(11, 21)]


class TestDetectHitsTrajectory:
    def test_reversal_inside_near_player_bbox_is_a_hit(self):
        ys = _near_hit_ys()
        tracking = [_frame(i, (550.0, ys[i]), _both(i)) for i in range(21)]
        hits = detect_hits(tracking, FPS)
        assert len(hits) == 1
        hit = hits[0]
        assert hit.role == "near_player"
        assert hit.frame == 10
        assert hit.kind == "turn"
        assert hit.ball_xy == (550.0, 550.0)
        assert hit.turn_deg == pytest.approx(180.0)
        assert hit.score > 0.0
        assert hit.speed_pre > 0.0 and hit.speed_post > 0.0

    def test_reversal_far_from_any_player_is_not_a_hit(self):
        ys = _near_hit_ys()
        lonely = [_player(0, "far_player", bbox=(50.0, 50.0, 100.0, 150.0))]
        tracking = [_frame(i, (550.0, ys[i]), lonely) for i in range(21)]
        assert detect_hits(tracking, FPS) == []

    def test_bounce_in_front_of_player_is_not_a_hit(self):
        """A bounce bends the track too, but the ball keeps coming toward the player."""
        ys = [300 + 20 * i for i in range(11)] + [480, 470, 465, 480, 500, 520, 540, 560, 580, 600, 620]
        tracking = [_frame(i, (550.0, ys[i]), _both(i)) for i in range(len(ys))]
        assert detect_hits(tracking, FPS) == []

    def test_far_player_hit(self):
        ys = [400 - 20 * i for i in range(11)] + [200 + 20 * (i - 10) for i in range(11, 21)]
        tracking = [_frame(i, (630.0, ys[i]), _both(i)) for i in range(21)]
        hits = detect_hits(tracking, FPS)
        assert [(h.frame, h.role, h.kind) for h in hits] == [(10, "far_player", "turn")]

    def test_rally_yields_alternating_hits_sorted_by_frame(self):
        ys = [300 + 25 * i for i in range(11)]  # 300 -> 550 (near player hits at frame 10)
        ys += [550 - 31 * (i - 10) for i in range(11, 25)]  # 519 -> 116 (far player hits at frame 24)
        ys += [116 + 31 * (i - 24) for i in range(25, 39)]  # back down to 550
        tracking = [_frame(i, (630.0, ys[i]), _both(i)) for i in range(len(ys))]
        hits = detect_hits(tracking, FPS)
        assert [(h.frame, h.role) for h in hits] == [(10, "near_player"), (24, "far_player")]
        assert [h.frame for h in hits] == sorted(h.frame for h in hits)

    def test_court_space_away_test_with_homography(self):
        ys = _near_hit_ys()
        tracking = [_frame(i, (550.0, ys[i]), _both(i)) for i in range(21)]
        hits = detect_hits(tracking, FPS, homography=H_LINEAR)
        assert [(h.frame, h.role) for h in hits] == [(10, "near_player")]

    def test_players_without_role_do_not_gate_hits(self):
        ys = _near_hit_ys()
        unassigned = [PlayerDetection(0, NEAR_BBOX, 0.9, role=None)]
        tracking = [_frame(i, (550.0, ys[i]), unassigned) for i in range(21)]
        assert detect_hits(tracking, FPS) == []

    def test_turn_threshold_is_respected(self):
        ys = _near_hit_ys()
        tracking = [_frame(i, (550.0, ys[i]), _both(i)) for i in range(21)]
        assert detect_hits(tracking, FPS, min_turn_deg=181.0, use_swing=False) == []

    def test_empty_and_ball_less_tracking(self):
        assert detect_hits([], FPS) == []
        tracking = [_frame(i, None, _both(i)) for i in range(21)]
        assert detect_hits(tracking, FPS) == []

    def test_unsorted_input_is_tolerated(self):
        ys = _near_hit_ys()
        tracking = [_frame(i, (550.0, ys[i]), _both(i)) for i in range(21)]
        hits = detect_hits(list(reversed(tracking)), FPS)
        assert [(h.frame, h.role) for h in hits] == [(10, "near_player")]


class TestDetectHitsSwing:
    def _pose(self, i: int, wrist_x: float) -> PoseKeypoints:
        return PoseKeypoints(frame_index=i, role="near_player", keypoints={
            "right_wrist": (wrist_x, 500.0, 0.9),
            "left_wrist": (520.0, 500.0, 0.9),
        })

    def _tracking(self) -> list[FrameTrackingResult]:
        # Sharp swing between frames 9 and 10: one bbox-height (300 px) in a single
        # frame. After the 3-frame smoothing that is ~0.34 bbox-heights/frame, which
        # clears the 2.5 x swing_min floor a wrist peak needs when no ball supports it.
        wrist_x = [550.0 + 3 * i for i in range(21)]
        for i in range(10, 21):
            wrist_x[i] += 300.0
        return [_frame(i, None, [_player(i, "near_player")], [self._pose(i, wrist_x[i])]) for i in range(21)]

    def test_wrist_speed_peak_without_ball_is_a_swing_hit(self):
        hits = detect_hits(self._tracking(), FPS)
        assert len(hits) == 1
        hit = hits[0]
        assert hit.role == "near_player"
        assert hit.kind == "swing"
        assert abs(hit.frame - 10) <= 2
        assert hit.ball_xy == (0.0, 0.0)
        assert 0.0 < hit.score <= 1.0

    def test_swing_candidates_can_be_disabled(self):
        assert detect_hits(self._tracking(), FPS, use_swing=False) == []

    def test_swing_threshold(self):
        assert detect_hits(self._tracking(), FPS, swing_min=5.0) == []

    def test_ball_tracked_elsewhere_vetoes_the_swing(self):
        tracking = self._tracking()
        far_away = [_frame(t.frame_index, (50.0, 50.0), t.players, t.poses) for t in tracking]
        assert detect_hits(far_away, FPS) == []


class TestEnforceAlternation:
    def test_same_role_within_gap_keeps_better_score(self):
        hits = [_hit(0, "near_player", 0.5), _hit(10, "near_player", 0.9), _hit(30, "far_player", 0.5)]
        out = enforce_alternation(hits, FPS)
        assert [(h.frame, h.role) for h in out] == [(10, "near_player"), (30, "far_player")]

    def test_same_role_within_gap_keeps_first_when_better(self):
        hits = [_hit(0, "near_player", 0.9), _hit(10, "near_player", 0.5)]
        assert [h.frame for h in enforce_alternation(hits, FPS)] == [0]

    def test_same_role_beyond_gap_is_kept(self):
        hits = [_hit(0, "near_player"), _hit(60, "near_player")]
        assert [h.frame for h in enforce_alternation(hits, FPS, max_same_role_gap_s=1.5)] == [0, 60]

    def test_alternating_roles_are_kept(self):
        hits = [_hit(0, "near_player"), _hit(10, "far_player"), _hit(20, "near_player")]
        assert [h.frame for h in enforce_alternation(hits, FPS)] == [0, 10, 20]

    def test_output_is_sorted_by_frame(self):
        hits = [_hit(40, "far_player"), _hit(0, "near_player")]
        assert [h.frame for h in enforce_alternation(hits, FPS)] == [0, 40]

    def test_empty(self):
        assert enforce_alternation([], FPS) == []


class TestGeometryHelpers:
    def test_expanded_bbox(self):
        p = PlayerDetection(0, (100.0, 100.0, 200.0, 300.0), 0.9, role="near_player")
        assert _expanded_bbox(p, 0.5) == (50.0, 0.0, 250.0, 400.0)
        assert _expanded_bbox(p, 0.0) == p.bbox

    def test_dist_to_bbox(self):
        bbox = (0.0, 0.0, 10.0, 10.0)
        assert _dist_to_bbox(5.0, 5.0, bbox) == 0.0
        assert _dist_to_bbox(13.0, 14.0, bbox) == pytest.approx(5.0)
        assert _dist_to_bbox(-3.0, 5.0, bbox) == pytest.approx(3.0)

    def test_nearest_player_ignores_unassigned(self):
        t = _frame(0, None, [
            PlayerDetection(0, (0.0, 0.0, 10.0, 10.0), 0.9, role=None),
            _player(0, "far_player", bbox=(100.0, 100.0, 120.0, 140.0)),
        ])
        role, d = _nearest_player(t, 5.0, 5.0, margin=0.0)
        assert role == "far_player"
        assert d > 0.0
        assert _nearest_player(_frame(0, None, []), 5.0, 5.0, margin=0.0) == (None, float("inf"))
