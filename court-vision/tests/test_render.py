"""The top-down minimap drawn into the bottom-right corner of rendered frames."""

import numpy as np

from court_vision.player_detect import FrameTrackingResult, PlayerDetection
from court_vision.research.render import draw_minimap
from court_vision.shot_classify import Point, Shot, ShotPlacement


def _point(bounce_frame=40):
    shot = Shot(shot_number=1, frame=10, time_s=0.33, player="near_player", stroke="forehand",
                placement=ShotPlacement(x=1.0, y=8.0, zone="crosscourt_deep"), confidence=0.9,
                speed_kmh=112.0, contact=(-0.5, -12.3), bounce_frame=bounce_frame)
    return Point(point_number=1, start_frame=0, end_frame=100, start_time_s=0.0, end_time_s=3.3,
                 server="near_player", rally_length=1, winner=None, outcome=None, outcome_player=None, shots=[shot])


def _tracking(frame):
    near = PlayerDetection(frame_index=frame, bbox=(600.0, 400.0, 680.0, 560.0), confidence=0.9,
                           role="near_player", court_position=(-0.5, -12.3))
    return {frame: FrameTrackingResult(frame_index=frame, ball=None, players=[near], poses=[])}


class TestDrawMinimap:
    def test_draws_only_in_the_bottom_right_corner(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = draw_minimap(img, _point(), 50, _tracking(50))
        assert out is img
        assert img[:360, :640].max() == 0            # top-left untouched
        assert img[400:, 1000:].max() > 0            # panel drawn bottom-right

    def test_landing_appears_only_after_the_bounce(self):
        before = np.zeros((720, 1280, 3), dtype=np.uint8)
        after = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_minimap(before, _point(bounce_frame=40), 20, _tracking(20))
        draw_minimap(after, _point(bounce_frame=40), 45, _tracking(45))
        # the after-frame carries the landing cross and speed label: more green pixels
        green = lambda a: int(((a[:, :, 1] > 200) & (a[:, :, 0] < 50)).sum())
        assert green(after) > green(before)

    def test_too_small_a_frame_is_left_alone(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_minimap(img, _point(), 50, _tracking(50))
        assert img.max() == 0
