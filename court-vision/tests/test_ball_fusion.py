"""Tests for ball_fusion — segment edge trim + contact wrist snap."""

from court_vision.ball_fusion import (
    CONTACT_SNAP_MAX_WRIST_DISTANCE_PX,
    snap_ball_to_racket_at_contacts,
    trim_segment_edges,
)
from court_vision.ball_tracker import BallDetection
from court_vision.player_detect import FrameTrackingResult, PoseKeypoints


def _ball(frame: int, x: float = 100, y: float = 100, conf: float = 0.9,
          interp: bool = False) -> BallDetection:
    return BallDetection(frame_index=frame, x=x, y=y, confidence=conf, interpolated=interp)


def _tr(frame: int, ball: BallDetection | None = None,
        poses: list[PoseKeypoints] | None = None) -> FrameTrackingResult:
    return FrameTrackingResult(
        frame_index=frame, ball=ball, players=[], poses=poses or [],
    )


class TestTrimSegmentEdges:
    def test_trims_weak_before_first_strong(self):
        # Frames 0-3 weak, 4-6 strong, 7 weak — weak at 0-3 should be dropped
        trs = [_tr(i, _ball(i, conf=0.5, interp=True)) for i in range(4)]
        trs += [_tr(i, _ball(i, conf=0.9)) for i in range(4, 7)]
        trs += [_tr(7, _ball(7, conf=0.5))]
        out = trim_segment_edges(trs, [(0, 7)])
        # weak before first strong (4) are nulled
        for i in range(4):
            assert out[i].ball is None, f"frame {i} should be trimmed"
        # strong preserved
        for i in range(4, 7):
            assert out[i].ball is not None
        # weak after last strong (6) is trimmed
        assert out[7].ball is None

    def test_preserves_weak_between_strong(self):
        # Strong, weak, strong — middle weak should survive
        trs = [
            _tr(0, _ball(0, conf=0.9)),
            _tr(1, _ball(1, conf=0.4, interp=True)),
            _tr(2, _ball(2, conf=0.9)),
        ]
        out = trim_segment_edges(trs, [(0, 2)])
        assert all(t.ball is not None for t in out)

    def test_noop_if_no_strong(self):
        trs = [_tr(i, _ball(i, conf=0.5, interp=True)) for i in range(3)]
        out = trim_segment_edges(trs, [(0, 2)])
        assert all(t.ball is not None for t in out)

    def test_leaves_nonball_fields_alone(self):
        pose = PoseKeypoints(frame_index=0, role="near_player",
                              keypoints={"right_wrist": (10.0, 20.0, 0.9)})
        trs = [_tr(0, _ball(0, conf=0.4, interp=True), poses=[pose])]
        out = trim_segment_edges(trs, [(0, 0)])
        assert out[0].poses == [pose]

    def test_multiple_segments_independent(self):
        # Two segments; each handled on its own
        trs = []
        # Segment A: frames 0-3, weak + strong at 2 + weak at 3
        trs += [_tr(0, _ball(0, conf=0.4, interp=True))]
        trs += [_tr(1, _ball(1, conf=0.4, interp=True))]
        trs += [_tr(2, _ball(2, conf=0.9))]
        trs += [_tr(3, _ball(3, conf=0.4, interp=True))]
        # Segment B: frames 10-12, weak throughout (no strong)
        trs += [_tr(10, _ball(10, conf=0.4, interp=True))]
        trs += [_tr(11, _ball(11, conf=0.4, interp=True))]
        trs += [_tr(12, _ball(12, conf=0.4, interp=True))]
        out = trim_segment_edges(trs, [(0, 3), (10, 12)])
        # Segment A: 0,1,3 trimmed; 2 kept
        assert out[0].ball is None and out[1].ball is None
        assert out[2].ball is not None
        assert out[3].ball is None
        # Segment B: no strong → no-op
        assert all(t.ball is not None for t in out[4:])


class TestSnapBallToRacket:
    def test_snaps_when_ball_far_from_wrist(self):
        pose = PoseKeypoints(
            frame_index=10, role="near_player",
            keypoints={"right_wrist": (500.0, 400.0, 0.9)},
        )
        # Ball detection is 600 px away — beyond threshold
        trs = [_tr(10, _ball(10, x=100, y=100, conf=0.9), poses=[pose])]
        out = snap_ball_to_racket_at_contacts(trs, [(10, "near_player")])
        snapped = out[0].ball
        assert snapped is not None
        assert snapped.x == 500.0 and snapped.y == 400.0
        assert snapped.confidence == 1.0

    def test_no_snap_when_ball_close_to_wrist(self):
        pose = PoseKeypoints(
            frame_index=10, role="near_player",
            keypoints={"right_wrist": (500.0, 400.0, 0.9)},
        )
        # Ball already 50 px from wrist — leave alone
        trs = [_tr(10, _ball(10, x=510, y=410, conf=0.9), poses=[pose])]
        out = snap_ball_to_racket_at_contacts(trs, [(10, "near_player")])
        assert out[0].ball.x == 510.0 and out[0].ball.y == 410.0

    def test_creates_ball_when_missing(self):
        pose = PoseKeypoints(
            frame_index=10, role="far_player",
            keypoints={"right_wrist": (500.0, 400.0, 0.9)},
        )
        trs = [_tr(10, ball=None, poses=[pose])]
        out = snap_ball_to_racket_at_contacts(trs, [(10, "far_player")])
        assert out[0].ball is not None
        assert out[0].ball.x == 500.0

    def test_skips_when_pose_missing_for_role(self):
        pose = PoseKeypoints(
            frame_index=10, role="near_player",
            keypoints={"right_wrist": (500.0, 400.0, 0.9)},
        )
        trs = [_tr(10, _ball(10, x=100, y=100, conf=0.9), poses=[pose])]
        # Contact is for far_player but only near_player pose exists
        out = snap_ball_to_racket_at_contacts(trs, [(10, "far_player")])
        assert out[0].ball.x == 100.0  # unchanged

    def test_falls_back_to_left_wrist_if_right_invisible(self):
        pose = PoseKeypoints(
            frame_index=10, role="near_player",
            keypoints={
                "right_wrist": (500.0, 400.0, 0.1),  # low visibility
                "left_wrist": (300.0, 350.0, 0.9),
            },
        )
        trs = [_tr(10, _ball(10, x=50, y=50, conf=0.9), poses=[pose])]
        out = snap_ball_to_racket_at_contacts(trs, [(10, "near_player")])
        assert out[0].ball.x == 300.0 and out[0].ball.y == 350.0

    def test_skips_when_no_wrist_visible(self):
        pose = PoseKeypoints(
            frame_index=10, role="near_player",
            keypoints={"right_wrist": (500.0, 400.0, 0.1),
                       "left_wrist": (300.0, 350.0, 0.1)},
        )
        trs = [_tr(10, _ball(10, x=50, y=50, conf=0.9), poses=[pose])]
        out = snap_ball_to_racket_at_contacts(trs, [(10, "near_player")])
        assert out[0].ball.x == 50.0

    def test_contact_frame_not_in_tracking_is_skipped(self):
        trs = [_tr(10, _ball(10, conf=0.9))]
        out = snap_ball_to_racket_at_contacts(trs, [(99, "near_player")])
        assert len(out) == 1 and out[0].ball.x == 100.0
