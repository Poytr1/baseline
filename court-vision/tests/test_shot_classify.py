"""Tests for the shot classification module."""

import numpy as np
import pytest

from court_vision.ball_tracker import BallDetection
from court_vision.config import PipelineSettings
from court_vision.hit_detect import Hit
from court_vision.player_detect import FrameTrackingResult, PlayerDetection, PoseKeypoints
from court_vision.scene_filter import GameplaySegment
from court_vision.shot_classify import (
    STROKES,
    MatchData,
    Point,
    Shot,
    ShotPlacement,
    build_match_data,
    classify_stroke,
    compute_placement_zone,
    detect_contacts,
    estimate_bounce,
    infer_handedness,
    other_role,
    point_winner,
    split_hits_into_points,
)


def _make_pose(role: str = "near_player", frame_index: int = 0, **overrides) -> PoseKeypoints:
    """Upright pose facing the camera: nose at y=100, shoulders 200, hips 400, body centre x=400."""
    defaults = {
        "nose": (400.0, 100.0, 0.9),
        "left_shoulder": (380.0, 200.0, 0.9),
        "right_shoulder": (420.0, 200.0, 0.9),
        "left_elbow": (360.0, 280.0, 0.9),
        "right_elbow": (440.0, 280.0, 0.9),
        "left_wrist": (350.0, 350.0, 0.9),
        "right_wrist": (450.0, 350.0, 0.9),
        "left_hip": (390.0, 400.0, 0.9),
        "right_hip": (410.0, 400.0, 0.9),
    }
    defaults.update(overrides)
    return PoseKeypoints(frame_index=frame_index, role=role, keypoints=defaults)


def _hit(frame: int, role: str = "near_player", ball_xy=(0.0, 0.0), score: float = 1.0, kind: str = "turn") -> Hit:
    return Hit(frame=frame, role=role, score=score, ball_xy=ball_xy, turn_deg=0.0,
               speed_pre=0.0, speed_post=0.0, kind=kind)


def _segment(start: int, end: int, fps: float = 30.0) -> GameplaySegment:
    return GameplaySegment(start_frame=start, end_frame=end, start_time_s=start / fps,
                           end_time_s=end / fps, frame_count=end - start + 1)


class TestShotPlacement:
    def test_placement_fields(self):
        p = ShotPlacement(x=3.0, y=8.0, zone="crosscourt_deep")
        assert p.x == 3.0
        assert p.y == 8.0
        assert p.zone == "crosscourt_deep"


class TestShot:
    def test_shot_fields(self):
        s = Shot(
            shot_number=1, frame=100, time_s=3.33,
            player="near_player", stroke="forehand",
            placement=ShotPlacement(x=2.0, y=9.0, zone="crosscourt_deep"),
            confidence=0.85,
        )
        assert s.stroke == "forehand"
        assert s.placement.zone == "crosscourt_deep"


class TestPoint:
    def test_point_fields(self):
        p = Point(
            point_number=1,
            start_frame=100, end_frame=200,
            start_time_s=3.33, end_time_s=6.67,
            server="near_player",
            shots=[],
            outcome="winner",
            outcome_player="far_player",
            rally_length=3,
        )
        assert p.review_status == "pending"
        assert p.outcome == "winner"

    def test_new_outcome_fields_default_to_none(self):
        p = Point(1, 0, 10, 0.0, 0.33, None, [], None, None, 0)
        assert p.winner is None
        assert p.outcome_source is None

    def test_new_outcome_fields_settable(self):
        p = Point(1, 0, 10, 0.0, 0.33, None, [], "error", "near_player", 0,
                  winner="far_player", outcome_source="scoreboard")
        assert p.winner == "far_player"
        assert p.outcome_source == "scoreboard"


class TestMatchData:
    def test_match_data_fields(self):
        m = MatchData(
            match_id="test_123",
            source_url="test.mp4",
            metadata={"players": ["near_player", "far_player"]},
            points=[],
        )
        assert m.match_id == "test_123"


class TestPointWinner:
    def test_explicit_winner_wins(self):
        p = Point(1, 0, 10, 0.0, 0.3, None, [], "error", "near_player", 0, winner="near_player")
        assert point_winner(p) == "near_player"

    def test_winner_outcome_means_hitter_won(self):
        p = Point(1, 0, 10, 0.0, 0.3, None, [], "winner", "near_player", 0)
        assert point_winner(p) == "near_player"

    @pytest.mark.parametrize("outcome", ["error", "unforced_error"])
    def test_error_outcome_means_opponent_won(self, outcome: str):
        p = Point(1, 0, 10, 0.0, 0.3, None, [], outcome, "near_player", 0)
        assert point_winner(p) == "far_player"

    def test_unknown_when_no_outcome_player(self):
        p = Point(1, 0, 10, 0.0, 0.3, None, [], None, None, 0)
        assert point_winner(p) is None

    def test_unknown_outcome_label(self):
        p = Point(1, 0, 10, 0.0, 0.3, None, [], "let", "near_player", 0)
        assert point_winner(p) is None

    def test_other_role(self):
        assert other_role("near_player") == "far_player"
        assert other_role("far_player") == "near_player"


class TestComputePlacementZone:
    def test_serve_wide_deuce(self):
        assert compute_placement_zone(x=3.5, y=4.0, is_serve=True, server_side="deuce") == "wide"

    def test_serve_t_deuce(self):
        assert compute_placement_zone(x=0.5, y=4.0, is_serve=True, server_side="deuce") == "t"

    def test_serve_body_deuce(self):
        assert compute_placement_zone(x=2.0, y=4.0, is_serve=True, server_side="deuce") == "body"

    def test_serve_wide_ad(self):
        assert compute_placement_zone(x=-3.5, y=4.0, is_serve=True, server_side="ad") == "wide"

    def test_rally_crosscourt_deep(self):
        assert compute_placement_zone(x=3.0, y=9.0, is_serve=False, hitter="near_player") == "crosscourt_deep"

    def test_rally_down_the_line_deep(self):
        assert compute_placement_zone(x=-3.0, y=9.0, is_serve=False, hitter="near_player") == "down_the_line_deep"

    def test_rally_middle_short(self):
        assert compute_placement_zone(x=0.5, y=3.0, is_serve=False, hitter="near_player") == "middle_short"

    def test_rally_crosscourt_short(self):
        assert compute_placement_zone(x=3.0, y=3.0, is_serve=False, hitter="near_player") == "crosscourt_short"

    def test_rally_direction_mirrored_for_far_player(self):
        assert compute_placement_zone(x=3.0, y=-9.0, is_serve=False, hitter="far_player") == "down_the_line_deep"
        assert compute_placement_zone(x=-3.0, y=-9.0, is_serve=False, hitter="far_player") == "crosscourt_deep"


class TestClassifyStroke:
    """Rules: serve (first shot, arm/ball overhead) > overhead > volley > FH/BH by side, slice by swing path."""

    @pytest.mark.parametrize("kwargs", [
        {},
        {"ball_xy": (500.0, 300.0)},
        {"is_first_shot": True},
        {"court_y": -2.0},
    ])
    def test_returns_known_stroke_and_confidence(self, kwargs):
        stroke, conf = classify_stroke(_make_pose(), **kwargs)
        assert stroke in STROKES
        assert 0.0 < conf <= 1.0

    # ── serve ──

    def test_serve_first_shot_with_ball_over_head(self):
        pose = _make_pose(right_wrist=(430.0, 50.0, 0.9))
        stroke, conf = classify_stroke(pose, ball_xy=(430.0, 40.0), is_first_shot=True)
        assert stroke == "serve"
        assert conf >= 0.7

    def test_serve_first_shot_both_wrists_high_without_ball(self):
        pose = _make_pose(right_wrist=(430.0, 50.0, 0.9), left_wrist=(380.0, 70.0, 0.9))
        stroke, conf = classify_stroke(pose, is_first_shot=True)
        assert stroke == "serve"
        assert conf > 0.0

    def test_ball_evidence_raises_serve_confidence(self):
        pose = _make_pose(right_wrist=(430.0, 50.0, 0.9), left_wrist=(380.0, 70.0, 0.9))
        _, without_ball = classify_stroke(pose, is_first_shot=True)
        _, with_ball = classify_stroke(pose, ball_xy=(430.0, 40.0), is_first_shot=True)
        assert with_ball >= without_ball

    def test_arms_up_mid_rally_is_not_a_serve(self):
        pose = _make_pose(right_wrist=(430.0, 50.0, 0.9), left_wrist=(380.0, 70.0, 0.9))
        stroke, _ = classify_stroke(pose, is_first_shot=False)
        assert stroke != "serve"

    def test_first_shot_with_low_contact_is_not_a_serve(self):
        stroke, _ = classify_stroke(_make_pose(), ball_xy=(500.0, 300.0), is_first_shot=True)
        assert stroke in ("forehand", "backhand")

    # ── overhead ──

    def test_overhead_ball_over_head_with_wrist_up(self):
        pose = _make_pose(right_wrist=(420.0, 50.0, 0.9))
        stroke, conf = classify_stroke(pose, ball_xy=(420.0, 60.0))
        assert stroke == "overhead"
        assert conf > 0.0

    def test_wrist_up_without_ball_is_not_overhead(self):
        pose = _make_pose(right_wrist=(420.0, 50.0, 0.9))
        stroke, _ = classify_stroke(pose)
        assert stroke in ("forehand", "backhand")

    def test_ball_over_head_without_raised_wrist_is_not_overhead(self):
        """A lob flying over a player who has not reached up is not an overhead."""
        stroke, _ = classify_stroke(_make_pose(), ball_xy=(420.0, 60.0))
        assert stroke != "overhead"

    # ── volley ──

    def test_volley_inside_service_boxes(self):
        stroke, conf = classify_stroke(_make_pose(), ball_xy=(500.0, 300.0), court_y=-3.0)
        assert stroke == "volley"
        assert conf > 0.0

    @pytest.mark.parametrize("court_y", [-10.0, 10.0])
    def test_no_volley_from_the_baseline(self, court_y: float):
        stroke, _ = classify_stroke(_make_pose(), ball_xy=(500.0, 300.0), court_y=court_y)
        assert stroke != "volley"

    def test_volley_threshold_is_configurable(self):
        pose = _make_pose()
        assert classify_stroke(pose, ball_xy=(500.0, 300.0), court_y=-5.0, volley_max_court_y=6.0)[0] == "volley"
        assert classify_stroke(pose, ball_xy=(500.0, 300.0), court_y=-5.0, volley_max_court_y=4.5)[0] != "volley"

    def test_unknown_court_position_never_volleys(self):
        stroke, _ = classify_stroke(_make_pose(), ball_xy=(500.0, 300.0), court_y=None)
        assert stroke != "volley"

    def test_serve_takes_priority_over_volley(self):
        pose = _make_pose(right_wrist=(430.0, 50.0, 0.9), left_wrist=(380.0, 70.0, 0.9))
        stroke, _ = classify_stroke(pose, is_first_shot=True, court_y=1.0)
        assert stroke == "serve"

    # ── forehand / backhand ──

    def test_near_right_hander_ball_on_right_is_forehand(self):
        stroke, conf = classify_stroke(_make_pose(), ball_xy=(500.0, 300.0))
        assert stroke == "forehand"
        assert conf > 0.0

    def test_near_right_hander_ball_on_left_is_backhand(self):
        stroke, _ = classify_stroke(_make_pose(), ball_xy=(300.0, 300.0))
        assert stroke == "backhand"

    def test_far_player_side_is_mirrored(self):
        pose = _make_pose(role="far_player")
        assert classify_stroke(pose, ball_xy=(500.0, 300.0))[0] == "backhand"
        assert classify_stroke(pose, ball_xy=(300.0, 300.0))[0] == "forehand"

    def test_role_argument_overrides_pose_role(self):
        pose = _make_pose(role="near_player")
        assert classify_stroke(pose, ball_xy=(500.0, 300.0), role="far_player")[0] == "backhand"

    def test_left_hander_is_mirrored(self):
        assert classify_stroke(_make_pose(), ball_xy=(500.0, 300.0), hand="left")[0] == "backhand"
        assert classify_stroke(_make_pose(), ball_xy=(300.0, 300.0), hand="left")[0] == "forehand"

    def test_left_handed_far_player_double_mirror(self):
        pose = _make_pose(role="far_player")
        assert classify_stroke(pose, ball_xy=(500.0, 300.0), hand="left")[0] == "forehand"

    def test_without_ball_uses_extended_wrist(self):
        assert classify_stroke(_make_pose(right_wrist=(550.0, 250.0, 0.9)))[0] == "forehand"
        assert classify_stroke(_make_pose(right_wrist=(250.0, 250.0, 0.9)))[0] == "backhand"

    def test_ball_evidence_is_more_confident_than_wrist_only(self):
        pose = _make_pose(right_wrist=(550.0, 250.0, 0.9))
        _, wrist_conf = classify_stroke(pose)
        _, ball_conf = classify_stroke(pose, ball_xy=(550.0, 250.0))
        assert ball_conf > wrist_conf

    def test_no_body_keypoints_gives_low_confidence_forehand(self):
        pose = PoseKeypoints(frame_index=0, role="near_player", keypoints={"right_wrist": (500.0, 300.0, 0.9)})
        stroke, conf = classify_stroke(pose, ball_xy=(500.0, 300.0))
        assert stroke == "forehand"
        assert conf <= 0.5

    def test_low_visibility_keypoints_are_ignored(self):
        """Invisible shoulders/hips mean no body centre -> default low-confidence forehand."""
        pose = _make_pose(
            left_shoulder=(380.0, 200.0, 0.1), right_shoulder=(420.0, 200.0, 0.1),
            left_hip=(390.0, 400.0, 0.1), right_hip=(410.0, 400.0, 0.1),
        )
        stroke, conf = classify_stroke(pose, ball_xy=(300.0, 300.0))
        assert stroke == "forehand"
        assert conf <= 0.5

    # ── slice ──

    def test_slice_when_wrist_drops_before_contact(self):
        history = [(0, 450.0, 250.0), (5, 450.0, 300.0), (9, 450.0, 350.0)]
        stroke, conf = classify_stroke(_make_pose(), ball_xy=(500.0, 300.0),
                                       wrist_history=history, torso_len=200.0)
        assert stroke == "slice"
        assert conf > 0.0

    def test_flat_swing_is_not_a_slice(self):
        history = [(0, 450.0, 345.0), (5, 450.0, 348.0), (9, 450.0, 350.0)]
        stroke, _ = classify_stroke(_make_pose(), ball_xy=(500.0, 300.0),
                                    wrist_history=history, torso_len=200.0)
        assert stroke == "forehand"

    def test_rising_swing_is_not_a_slice(self):
        history = [(0, 450.0, 400.0), (5, 450.0, 350.0), (9, 450.0, 300.0)]
        stroke, _ = classify_stroke(_make_pose(), ball_xy=(500.0, 300.0),
                                    wrist_history=history, torso_len=200.0)
        assert stroke == "forehand"

    def test_slice_needs_enough_history(self):
        history = [(0, 450.0, 250.0), (9, 450.0, 350.0)]
        stroke, _ = classify_stroke(_make_pose(), ball_xy=(500.0, 300.0),
                                    wrist_history=history, torso_len=200.0)
        assert stroke == "forehand"

    def test_slice_needs_torso_length(self):
        history = [(0, 450.0, 250.0), (5, 450.0, 300.0), (9, 450.0, 350.0)]
        assert classify_stroke(_make_pose(), ball_xy=(500.0, 300.0),
                               wrist_history=history, torso_len=None)[0] == "forehand"

    def test_slice_skipped_for_tiny_poses(self):
        """A far player only a few pixels tall cannot be judged for swing path."""
        history = [(0, 450.0, 250.0), (5, 450.0, 300.0), (9, 450.0, 350.0)]
        stroke, _ = classify_stroke(_make_pose(), ball_xy=(500.0, 300.0),
                                    wrist_history=history, torso_len=20.0, slice_min_torso_px=30.0)
        assert stroke == "forehand"

    def test_slice_drop_ratio_is_configurable(self):
        history = [(0, 450.0, 250.0), (5, 450.0, 300.0), (9, 450.0, 350.0)]  # drop of 0.5 torso lengths
        pose = _make_pose()
        assert classify_stroke(pose, ball_xy=(500.0, 300.0), wrist_history=history,
                               torso_len=200.0, slice_drop_ratio=0.25)[0] == "slice"
        assert classify_stroke(pose, ball_xy=(500.0, 300.0), wrist_history=history,
                               torso_len=200.0, slice_drop_ratio=0.6)[0] == "forehand"

    def test_slice_does_not_preempt_serve(self):
        pose = _make_pose(right_wrist=(430.0, 50.0, 0.9), left_wrist=(380.0, 70.0, 0.9))
        history = [(0, 430.0, 0.0), (5, 430.0, 30.0), (9, 430.0, 50.0)]
        stroke, _ = classify_stroke(pose, is_first_shot=True, wrist_history=history, torso_len=50.0)
        assert stroke == "serve"


class TestInferHandedness:
    def _by_frame(self, pose: PoseKeypoints) -> dict[int, FrameTrackingResult]:
        return {pose.frame_index: FrameTrackingResult(pose.frame_index, None, [], [pose])}

    def test_right_wrist_reaching_for_ball_votes_right(self):
        pose = _make_pose(frame_index=10, right_wrist=(500.0, 300.0, 0.9), left_wrist=(200.0, 300.0, 0.9))
        hit = _hit(10, ball_xy=(520.0, 300.0))
        assert infer_handedness(self._by_frame(pose), [hit], "near_player") == "right"

    def test_left_wrist_reaching_for_ball_votes_left(self):
        pose = _make_pose(frame_index=10, right_wrist=(500.0, 300.0, 0.9), left_wrist=(200.0, 300.0, 0.9))
        hit = _hit(10, ball_xy=(180.0, 300.0))
        assert infer_handedness(self._by_frame(pose), [hit], "near_player") == "left"

    def test_defaults_to_right_without_evidence(self):
        assert infer_handedness({}, [], "near_player") == "right"
        pose = _make_pose(frame_index=10)
        assert infer_handedness(self._by_frame(pose), [_hit(10, role="far_player")], "near_player") == "right"

    def test_two_handed_contact_is_ignored(self):
        """Wrists together = two hands on the racket: no vote, default right."""
        pose = _make_pose(frame_index=10, right_wrist=(250.0, 300.0, 0.9), left_wrist=(240.0, 300.0, 0.9))
        hit = _hit(10, ball_xy=(200.0, 300.0))
        assert infer_handedness(self._by_frame(pose), [hit], "near_player") == "right"

    def test_serve_cue_uses_the_raised_arm(self):
        left_up = _make_pose(frame_index=10, left_wrist=(300.0, 20.0, 0.9), right_wrist=(320.0, 380.0, 0.9))
        assert infer_handedness(self._by_frame(left_up), [_hit(10)], "near_player") == "left"
        right_up = _make_pose(frame_index=10, right_wrist=(500.0, 20.0, 0.9), left_wrist=(380.0, 380.0, 0.9))
        assert infer_handedness(self._by_frame(right_up), [_hit(10)], "near_player") == "right"


class TestDetectContacts:
    def test_detects_contact_when_ball_near_player(self):
        tracking = [
            FrameTrackingResult(
                frame_index=0,
                ball=BallDetection(frame_index=0, x=300.0, y=200.0, confidence=0.9),
                players=[PlayerDetection(frame_index=0, bbox=(250.0, 100.0, 350.0, 500.0),
                                         confidence=0.9, role="near_player")],
                poses=[],
            ),
            FrameTrackingResult(
                frame_index=1,
                ball=BallDetection(frame_index=1, x=310.0, y=250.0, confidence=0.9),
                players=[PlayerDetection(frame_index=1, bbox=(250.0, 100.0, 350.0, 500.0),
                                         confidence=0.9, role="near_player")],
                poses=[],
            ),
        ]
        contacts = detect_contacts(tracking, fps=30.0)
        assert len(contacts) >= 1
        assert contacts[0][1] in ("near_player", "far_player")

    def test_no_contact_when_ball_far_from_player(self):
        tracking = [
            FrameTrackingResult(
                frame_index=0,
                ball=BallDetection(frame_index=0, x=100.0, y=100.0, confidence=0.9),
                players=[PlayerDetection(frame_index=0, bbox=(500.0, 300.0, 600.0, 600.0),
                                         confidence=0.9, role="near_player")],
                poses=[],
            ),
        ]
        assert detect_contacts(tracking, fps=30.0) == []

    def test_no_contact_when_no_ball(self):
        tracking = [
            FrameTrackingResult(
                frame_index=0, ball=None,
                players=[PlayerDetection(frame_index=0, bbox=(250.0, 100.0, 350.0, 500.0),
                                         confidence=0.9, role="near_player")],
                poses=[],
            ),
        ]
        assert detect_contacts(tracking, fps=30.0) == []

    def test_min_frames_between_contacts_debounces(self):
        player = PlayerDetection(frame_index=0, bbox=(250.0, 100.0, 350.0, 500.0), confidence=0.9, role="near_player")
        tracking = [
            FrameTrackingResult(i, BallDetection(i, 300.0, 200.0, 0.9), [player], []) for i in range(5)
        ]
        contacts = detect_contacts(tracking, fps=30.0, min_frames_between_contacts=10)
        assert [f for f, _ in contacts] == [0]

    def test_players_without_role_are_ignored(self):
        player = PlayerDetection(frame_index=0, bbox=(250.0, 100.0, 350.0, 500.0), confidence=0.9, role=None)
        tracking = [FrameTrackingResult(0, BallDetection(0, 300.0, 200.0, 0.9), [player], [])]
        assert detect_contacts(tracking, fps=30.0) == []


class TestSplitHitsIntoPoints:
    def test_long_gap_splits_points(self):
        seg = _segment(0, 400)
        hits = [_hit(10), _hit(30, "far_player"), _hit(50), _hit(200, "far_player"), _hit(220)]
        points = split_hits_into_points(hits, seg, fps=30.0, gap_s=4.0, pad_before_s=1.0, pad_after_s=2.5)
        assert [[h.frame for h in g] for _, _, g in points] == [[10, 30, 50], [200, 220]]
        (s1, e1, _), (s2, e2, _) = points
        assert s1 == seg.start_frame  # first point starts with the segment
        assert e1 == 50 + 75  # last hit + 2.5 s padding
        assert s2 == 200 - 30  # first hit - 1 s padding
        assert e2 == seg.end_frame  # last point ends with the segment
        assert s2 > e1

    def test_no_hits_yields_one_empty_point(self):
        seg = _segment(0, 400)
        assert split_hits_into_points([], seg, fps=30.0) == [(0, 400, [])]

    def test_hits_within_gap_stay_in_one_point(self):
        seg = _segment(0, 400)
        hits = [_hit(10), _hit(100, "far_player"), _hit(190)]
        points = split_hits_into_points(hits, seg, fps=30.0, gap_s=4.0)
        assert len(points) == 1
        assert points[0] == (0, 400, hits)

    def test_overlapping_padding_is_split_at_the_midpoint(self):
        seg = _segment(0, 400)
        points = split_hits_into_points([_hit(50), _hit(90)], seg, fps=30.0, gap_s=1.0)
        assert [(s, e) for s, e, _ in points] == [(0, 70), (71, 400)]

    def test_points_never_leave_the_segment(self):
        seg = _segment(100, 300)
        points = split_hits_into_points([_hit(105), _hit(295)], seg, fps=30.0, gap_s=1.0)
        for s, e, _ in points:
            assert seg.start_frame <= s <= e <= seg.end_frame


# Linear pixel->court homography: court_x = (px - 640) / 50, court_y = -(py - 360) / 25
# (larger image y == closer to the camera == negative court y).
H_LINEAR = np.array([[1 / 50, 0.0, -12.8], [0.0, -1 / 25, 14.4], [0.0, 0.0, 1.0]])


class TestEstimateBounce:
    """``estimate_bounce`` ignores the first 0.15 s after ``start`` (the hit
    itself), needs at least five samples, ignores the 2 m net band and reports
    the landing on the opponent's side of the net: the point where the ball
    stops descending (in image space the ball is lowest on screen there; the
    court projection of an airborne ball overshoots, so the projected depth is
    at a local minimum there).

    All trajectories below use ``H_LINEAR`` (court_y = -(py - 360) / 25) with
    the ball on the far side of the court (py < 310) unless stated otherwise.
    """

    def _by_frame(self, ys, x0: float = 640.0, first_frame: int = 1, dx: float = 0.0) -> dict[int, FrameTrackingResult]:
        return {
            first_frame + i: FrameTrackingResult(
                first_frame + i, BallDetection(first_frame + i, x0 + dx * i, float(y), 0.9), [], [],
            )
            for i, y in enumerate(ys)
        }

    # frames 21..40: the ball flies away up-screen (py 300 -> 200), comes down
    # again to its bounce at frame 34 (py 280, 3.2 m past the net) and rises
    FLIGHT_AND_BOUNCE = [300, 280, 260, 240, 220, 200, 210, 220, 230, 240, 250, 260, 270, 280,
                         270, 260, 250, 240, 230, 220]

    def test_landing_is_where_the_ball_stops_descending(self):
        by_frame = self._by_frame(self.FLIGHT_AND_BOUNCE, first_frame=21)
        assert estimate_bounce(by_frame, 16, 40, H_LINEAR, fps=30.0) == pytest.approx((0.0, 3.2))
        assert estimate_bounce(by_frame, 16, 40, H_LINEAR, fps=30.0, hitter_role="near_player") == pytest.approx((0.0, 3.2))

    def test_none_without_homography(self):
        by_frame = self._by_frame(self.FLIGHT_AND_BOUNCE, first_frame=21)
        assert estimate_bounce(by_frame, 16, 40, None, fps=30.0) is None

    def test_none_with_too_few_samples(self):
        assert estimate_bounce(self._by_frame([300, 280, 260]), 0, 9, H_LINEAR, fps=30.0) is None
        # seven frames, but only four are left after the 0.15 s skip
        by_frame = self._by_frame([300, 280, 260, 240, 260, 280, 300])
        assert estimate_bounce(by_frame, 0, 7, H_LINEAR, fps=30.0) is None

    def test_none_when_the_window_ends_before_the_bounce(self):
        by_frame = self._by_frame(self.FLIGHT_AND_BOUNCE, first_frame=21)
        assert estimate_bounce(by_frame, 16, 31, H_LINEAR, fps=30.0) is None  # still descending at frame 31

    def test_ball_still_in_flight_has_no_bounce(self):
        away = self._by_frame([300 - 20 * i for i in range(14)])  # up-screen at constant speed
        toward = self._by_frame([40 + 20 * i for i in range(14)])
        assert estimate_bounce(away, 0, 14, H_LINEAR, fps=30.0) is None
        assert estimate_bounce(toward, 0, 14, H_LINEAR, fps=30.0) is None

    # frames 5..40: the ball descends onto the hitter (frame 12, py 280), is
    # sent away up-screen, comes down again and bounces at frame 27 (py 270)
    HIT_THEN_BOUNCE = (
        [210, 220, 230, 240, 250, 260, 270, 280]  # frames 5..12: toward the hit
        + [270, 260, 250, 240, 230, 220, 210, 200]  # frames 13..20: flying away
        + [210, 220, 230, 240, 250, 260, 270]  # frames 21..27: descending to the bounce
        + [260, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140]  # frames 28..40
    )

    def test_hit_itself_is_skipped(self):
        by_frame = self._by_frame(self.HIT_THEN_BOUNCE, first_frame=5)
        # start=10: frames 10..13 are skipped, so the contact at frame 12 is not taken for a bounce
        assert estimate_bounce(by_frame, 10, 40, H_LINEAR, fps=30.0) == pytest.approx((0.0, 3.6))
        # a window opening well before the hit does report the contact (it looks like a landing)
        assert estimate_bounce(by_frame, 2, 40, H_LINEAR, fps=30.0) == pytest.approx((0.0, 3.2))

    def test_skip_scales_with_fps(self):
        by_frame = self._by_frame(self.HIT_THEN_BOUNCE, first_frame=5)
        assert estimate_bounce(by_frame, 6, 40, H_LINEAR, fps=30.0) == pytest.approx((0.0, 3.6))
        # at 6 fps the 0.15 s skip is zero frames, so the contact at frame 12 is visible again
        assert estimate_bounce(by_frame, 6, 40, H_LINEAR, fps=6.0) == pytest.approx((0.0, 3.2))

    def test_only_samples_after_start_are_used(self):
        """Two bounces: a window opening after the first one must land on the second."""
        ys = (
            [200, 210, 220, 230, 240, 250, 260, 270, 280]  # frames 21..29: bounce A at frame 29
            + [270, 260, 250, 240, 230, 220]  # frames 30..35
            + [230, 240, 250, 260, 270, 280]  # frames 36..41: bounce B at frame 41
            + [270, 260, 250, 240, 230, 220, 210, 200, 190]  # frames 42..50
        )
        by_frame = self._by_frame(ys, first_frame=21, dx=1.0)  # court_x tells the bounces apart
        assert estimate_bounce(by_frame, 20, 50, H_LINEAR, fps=30.0) == pytest.approx((0.16, 3.2))
        assert estimate_bounce(by_frame, 30, 50, H_LINEAR, fps=30.0) == pytest.approx((0.4, 3.2))

    def test_hitter_role_restricts_to_the_opponent_side(self):
        far = self._by_frame(self.FLIGHT_AND_BOUNCE, first_frame=21)
        assert estimate_bounce(far, 16, 40, H_LINEAR, fps=30.0, hitter_role="far_player") is None
        # the mirror image: same flight on the near side of the net (py > 410)
        near = self._by_frame([720 - y for y in self.FLIGHT_AND_BOUNCE], first_frame=21)
        assert estimate_bounce(near, 16, 40, H_LINEAR, fps=30.0, hitter_role="far_player") == pytest.approx((0.0, -3.2))
        assert estimate_bounce(near, 16, 40, H_LINEAR, fps=30.0, hitter_role="near_player") is None
        assert estimate_bounce(near, 16, 40, H_LINEAR, fps=30.0) == pytest.approx((0.0, -3.2))

    def test_net_band_is_ignored(self):
        ys = [332, 342, 352, 362, 372, 382, 392, 402, 392, 382, 372, 362, 352, 342, 332, 322]  # all within 2 m of the net
        by_frame = self._by_frame(ys, first_frame=21)
        assert estimate_bounce(by_frame, 16, 36, H_LINEAR, fps=30.0) is None
        assert estimate_bounce(by_frame, 16, 36, H_LINEAR, fps=30.0, net_margin_m=0.0) is not None


class TestBuildMatchData:
    def _tracking_with_contact(self, frame: int = 10) -> list[FrameTrackingResult]:
        return [
            FrameTrackingResult(
                frame_index=frame,
                ball=BallDetection(frame_index=frame, x=340.0, y=300.0, confidence=0.9),
                players=[PlayerDetection(frame_index=frame, bbox=(250.0, 100.0, 350.0, 500.0),
                                         confidence=0.9, role="near_player")],
                poses=[PoseKeypoints(
                    frame_index=frame, role="near_player",
                    keypoints={
                        "nose": (300.0, 100.0, 0.9),
                        "left_shoulder": (280.0, 200.0, 0.9),
                        "right_shoulder": (320.0, 200.0, 0.9),
                        "left_elbow": (260.0, 280.0, 0.9),
                        "right_elbow": (340.0, 280.0, 0.9),
                        "left_wrist": (250.0, 350.0, 0.9),
                        "right_wrist": (450.0, 250.0, 0.9),
                        "left_hip": (290.0, 400.0, 0.9),
                        "right_hip": (310.0, 400.0, 0.9),
                    },
                )],
            ),
        ]

    def test_proximity_contacts_become_shots(self):
        segments = [_segment(0, 50)]
        match = build_match_data(
            source="test.mp4", segments=segments, tracking_results=self._tracking_with_contact(), fps=30.0,
            settings=PipelineSettings(contact_method="proximity", outcome_method="last_hitter"),
        )
        assert isinstance(match, MatchData)
        assert match.source_url == "test.mp4"
        assert len(match.points) == 1
        point = match.points[0]
        assert point.point_number == 1
        assert point.rally_length == 1
        shot = point.shots[0]
        assert shot.shot_number == 1
        assert shot.frame == 10
        assert shot.time_s == pytest.approx(10 / 30.0)
        assert shot.player == "near_player"
        assert shot.stroke == "forehand"  # ball right of the body centre, low contact
        assert shot.placement is None  # no homography
        assert point.server == "near_player"

    def test_last_hitter_outcome(self):
        match = build_match_data(
            source="test.mp4", segments=[_segment(0, 50)], tracking_results=self._tracking_with_contact(), fps=30.0,
            settings=PipelineSettings(contact_method="proximity", outcome_method="last_hitter"),
        )
        point = match.points[0]
        assert point.winner == "near_player"
        assert point.outcome_source == "last_hitter"
        assert point.outcome == "winner"
        assert point.outcome_player == "near_player"

    def test_default_trajectory_method_needs_ball_motion(self):
        """A single ball sample is not a trajectory: no hits, no shots, unknown outcome."""
        match = build_match_data(
            source="test.mp4", segments=[_segment(0, 50)], tracking_results=self._tracking_with_contact(), fps=30.0,
        )
        assert len(match.points) == 1
        assert match.points[0].shots == []
        assert match.points[0].winner is None
        assert match.points[0].outcome_source == "none"

    def test_empty_tracking_produces_empty_points(self):
        match = build_match_data(source="test.mp4", segments=[_segment(0, 50)], tracking_results=[], fps=30.0)
        assert len(match.points) == 1
        assert match.points[0].rally_length == 0
        assert match.points[0].shots == []
        assert match.points[0].server is None

    def test_one_point_per_segment_without_hits(self):
        match = build_match_data(
            source="test.mp4", segments=[_segment(0, 50), _segment(100, 150)], tracking_results=[], fps=30.0,
        )
        assert [p.point_number for p in match.points] == [1, 2]
        assert [(p.start_frame, p.end_frame) for p in match.points] == [(0, 50), (100, 150)]
        assert match.points[1].start_time_s == pytest.approx(100 / 30.0)

    def test_metadata_contents(self):
        match = build_match_data(
            source="test.mp4", segments=[_segment(0, 50)], tracking_results=self._tracking_with_contact(), fps=30.0,
            settings=PipelineSettings(contact_method="proximity", outcome_method="last_hitter"),
        )
        assert match.metadata["players"] == ["near_player", "far_player"]
        assert match.metadata["fps"] == 30.0
        assert set(match.metadata["handedness"]) == {"near_player", "far_player"}
        assert len(match.metadata["hits"]) == 1
        assert match.metadata["hits"][0]["frame"] == 10
        assert match.metadata["hits"][0]["kind"] == "proximity"

    def test_explicit_handedness_setting_is_used(self):
        match = build_match_data(
            source="test.mp4", segments=[_segment(0, 50)], tracking_results=[], fps=30.0,
            settings=PipelineSettings(near_player_hand="left", far_player_hand="right"),
        )
        assert match.metadata["handedness"] == {"near_player": "left", "far_player": "right"}

    def test_match_id_is_deterministic(self):
        a = build_match_data("clip.mp4", [], [], 30.0)
        b = build_match_data("clip.mp4", [], [], 30.0)
        c = build_match_data("other.mp4", [], [], 30.0)
        assert a.match_id == b.match_id
        assert a.match_id != c.match_id
        assert len(a.match_id) == 12
