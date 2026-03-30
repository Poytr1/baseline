"""Tests for the shot classification module."""

import pytest

from court_vision.shot_classify import (
    Shot,
    ShotPlacement,
    Point,
    MatchData,
    compute_placement_zone,
)


class TestShotPlacement:
    def test_placement_fields(self):
        """ShotPlacement has x, y, zone."""
        p = ShotPlacement(x=3.0, y=8.0, zone="crosscourt_deep")
        assert p.x == 3.0
        assert p.y == 8.0
        assert p.zone == "crosscourt_deep"


class TestShot:
    def test_shot_fields(self):
        """Shot has all required fields."""
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
        """Point has all required fields."""
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


class TestMatchData:
    def test_match_data_fields(self):
        """MatchData has all required fields."""
        m = MatchData(
            match_id="test_123",
            source_url="test.mp4",
            metadata={"players": ["near_player", "far_player"]},
            points=[],
        )
        assert m.match_id == "test_123"


class TestComputePlacementZone:
    def test_serve_wide_deuce(self):
        """Ball landing far right in deuce service box = wide."""
        zone = compute_placement_zone(x=3.5, y=4.0, is_serve=True, server_side="deuce")
        assert zone == "wide"

    def test_serve_t_deuce(self):
        """Ball landing near center in deuce service box = t."""
        zone = compute_placement_zone(x=0.5, y=4.0, is_serve=True, server_side="deuce")
        assert zone == "t"

    def test_serve_body_deuce(self):
        """Ball landing in middle zone of deuce service box = body."""
        zone = compute_placement_zone(x=2.0, y=4.0, is_serve=True, server_side="deuce")
        assert zone == "body"

    def test_serve_wide_ad(self):
        """Ball landing far left in ad service box = wide."""
        zone = compute_placement_zone(x=-3.5, y=4.0, is_serve=True, server_side="ad")
        assert zone == "wide"

    def test_rally_crosscourt_deep(self):
        """Ball landing crosscourt and deep = crosscourt_deep."""
        zone = compute_placement_zone(x=3.0, y=9.0, is_serve=False, hitter="near_player")
        assert zone == "crosscourt_deep"

    def test_rally_down_the_line_deep(self):
        """Ball landing down the line and deep = down_the_line_deep."""
        zone = compute_placement_zone(x=-3.0, y=9.0, is_serve=False, hitter="near_player")
        assert zone == "down_the_line_deep"

    def test_rally_middle_short(self):
        """Ball landing in middle and short = middle_short."""
        zone = compute_placement_zone(x=0.5, y=3.0, is_serve=False, hitter="near_player")
        assert zone == "middle_short"

    def test_rally_crosscourt_short(self):
        """Ball landing crosscourt and short = crosscourt_short."""
        zone = compute_placement_zone(x=3.0, y=3.0, is_serve=False, hitter="near_player")
        assert zone == "crosscourt_short"
