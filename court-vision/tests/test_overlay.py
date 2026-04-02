"""Tests for frame overlay rendering."""

import numpy as np

from court_vision.ball_tracker import BallDetection
from court_vision.player_detect import PlayerDetection, PoseKeypoints


class TestDrawBall:
    def test_draws_circle_on_frame(self):
        """draw_ball renders a circle at the ball position."""
        from court_vision.overlay import draw_ball

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ball = BallDetection(frame_index=0, x=320.0, y=240.0, confidence=0.9)

        result = draw_ball(frame, ball)

        # Check that pixels near the ball center are non-zero (circle drawn)
        assert result[240, 320].sum() > 0

    def test_returns_copy(self):
        """draw_ball does not modify the original frame."""
        from court_vision.overlay import draw_ball

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ball = BallDetection(frame_index=0, x=320.0, y=240.0, confidence=0.9)

        result = draw_ball(frame, ball)

        assert frame[240, 320].sum() == 0  # Original unchanged
        assert result is not frame


class TestDrawPlayers:
    def test_draws_bounding_box(self):
        """draw_players renders bounding boxes."""
        from court_vision.overlay import draw_players

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        players = [
            PlayerDetection(
                frame_index=0,
                bbox=(100.0, 200.0, 200.0, 400.0),
                confidence=0.9,
                role="near_player",
            ),
        ]

        result = draw_players(frame, players)

        # Check that pixels along the top edge of bbox are non-zero
        assert result[200, 150].sum() > 0


class TestDrawPoses:
    def test_draws_keypoints(self):
        """draw_poses renders keypoint dots."""
        from court_vision.overlay import draw_poses

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = [
            PoseKeypoints(
                frame_index=0,
                role="near_player",
                keypoints={
                    "nose": (320.0, 100.0, 0.9),
                    "left_shoulder": (300.0, 150.0, 0.8),
                    "right_shoulder": (340.0, 150.0, 0.8),
                },
            ),
        ]

        result = draw_poses(frame, poses)

        # Check that pixels at nose position are non-zero
        assert result[100, 320].sum() > 0


class TestRenderOverlay:
    def test_combines_all_overlays(self):
        """render_overlay applies ball, player, and pose overlays."""
        from court_vision.overlay import render_overlay
        from court_vision.player_detect import FrameTrackingResult

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracking = FrameTrackingResult(
            frame_index=0,
            ball=BallDetection(frame_index=0, x=320.0, y=240.0, confidence=0.9),
            players=[
                PlayerDetection(
                    frame_index=0, bbox=(100.0, 200.0, 200.0, 400.0),
                    confidence=0.9, role="near_player",
                ),
            ],
            poses=[
                PoseKeypoints(
                    frame_index=0, role="near_player",
                    keypoints={"nose": (320.0, 100.0, 0.9)},
                ),
            ],
        )

        result = render_overlay(frame, tracking)

        assert result.shape == frame.shape
        assert result.sum() > 0  # Something was drawn

    def test_handles_no_ball(self):
        """render_overlay handles tracking with no ball detection."""
        from court_vision.overlay import render_overlay
        from court_vision.player_detect import FrameTrackingResult

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracking = FrameTrackingResult(
            frame_index=0, ball=None, players=[], poses=[],
        )

        result = render_overlay(frame, tracking)

        assert result.shape == frame.shape


class TestDrawCourt:
    def test_draws_court_lines(self):
        """draw_court renders court lines when given a valid homography."""
        from court_vision.overlay import draw_court

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Homography: pixel -> court coordinates.
        # Court X range: approx -5.5 to +5.5 -> map to 0..640
        # Court Y range: approx -12 to +12 -> map to 0..480
        sx = 640.0 / 11.0  # ~58 px per meter
        sy = 480.0 / 24.0  # ~20 px per meter
        # court_x = (px - 320) / sx, court_y = (py - 240) / sy
        H = np.array([
            [1.0 / sx, 0.0, -320.0 / sx],
            [0.0, 1.0 / sy, -240.0 / sy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        result = draw_court(frame, H)

        # Court lines should have been drawn — frame should have non-zero pixels
        assert result.sum() > 0

    def test_returns_copy(self):
        """draw_court does not modify the original frame."""
        from court_vision.overlay import draw_court

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)

        result = draw_court(frame, H)

        assert result is not frame
        assert frame.sum() == 0  # Original unchanged

    def test_no_homography_returns_copy(self):
        """draw_court with None homography returns unchanged copy."""
        from court_vision.overlay import draw_court

        frame = np.full((480, 640, 3), 42, dtype=np.uint8)

        result = draw_court(frame, None)

        assert result is not frame
        np.testing.assert_array_equal(result, frame)
