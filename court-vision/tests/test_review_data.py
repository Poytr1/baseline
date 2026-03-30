"""Tests for review data loading and saving."""

import json
from pathlib import Path

from court_vision.shot_classify import MatchData, Point, Shot, ShotPlacement


class TestLoadMatchJson:
    def test_load_empty_match(self, tmp_path: Path):
        """Load a match JSON with no points."""
        from court_vision.review_data import load_match_json

        match_file = tmp_path / "match.json"
        match_file.write_text(json.dumps({
            "match_id": "abc123",
            "source_url": "test.mp4",
            "metadata": {"players": ["near_player", "far_player"]},
            "points": [],
        }))

        match = load_match_json(match_file)

        assert isinstance(match, MatchData)
        assert match.match_id == "abc123"
        assert match.source_url == "test.mp4"
        assert match.points == []

    def test_load_match_with_shots(self, tmp_path: Path):
        """Load a match JSON with points and shots."""
        from court_vision.review_data import load_match_json

        match_file = tmp_path / "match.json"
        match_file.write_text(json.dumps({
            "match_id": "abc123",
            "source_url": "test.mp4",
            "metadata": {},
            "points": [
                {
                    "point_number": 1,
                    "start_frame": 0,
                    "end_frame": 100,
                    "start_time_s": 0.0,
                    "end_time_s": 3.33,
                    "server": "near_player",
                    "shots": [
                        {
                            "shot_number": 1,
                            "frame": 10,
                            "time_s": 0.33,
                            "player": "near_player",
                            "stroke": "serve",
                            "placement": {
                                "x": 1.5,
                                "y": 4.0,
                                "zone": "t",
                            },
                            "confidence": 0.8,
                        },
                    ],
                    "outcome": "winner",
                    "outcome_player": "near_player",
                    "rally_length": 1,
                    "review_status": "pending",
                },
            ],
        }))

        match = load_match_json(match_file)

        assert len(match.points) == 1
        point = match.points[0]
        assert isinstance(point, Point)
        assert point.point_number == 1
        assert point.review_status == "pending"
        assert len(point.shots) == 1
        shot = point.shots[0]
        assert isinstance(shot, Shot)
        assert shot.stroke == "serve"
        assert isinstance(shot.placement, ShotPlacement)
        assert shot.placement.zone == "t"

    def test_load_match_with_null_placement(self, tmp_path: Path):
        """Load a shot with null placement."""
        from court_vision.review_data import load_match_json

        match_file = tmp_path / "match.json"
        match_file.write_text(json.dumps({
            "match_id": "abc123",
            "source_url": "test.mp4",
            "metadata": {},
            "points": [
                {
                    "point_number": 1,
                    "start_frame": 0, "end_frame": 50,
                    "start_time_s": 0.0, "end_time_s": 1.67,
                    "server": None,
                    "shots": [
                        {
                            "shot_number": 1, "frame": 10, "time_s": 0.33,
                            "player": "near_player", "stroke": "forehand",
                            "placement": None, "confidence": 0.4,
                        },
                    ],
                    "outcome": None, "outcome_player": None,
                    "rally_length": 1, "review_status": "pending",
                },
            ],
        }))

        match = load_match_json(match_file)
        assert match.points[0].shots[0].placement is None


class TestSaveMatchJson:
    def test_save_and_reload(self, tmp_path: Path):
        """Save match data and reload it — round-trip test."""
        from court_vision.review_data import load_match_json, save_match_json

        match = MatchData(
            match_id="test123",
            source_url="test.mp4",
            metadata={"players": ["near_player", "far_player"]},
            points=[
                Point(
                    point_number=1,
                    start_frame=0, end_frame=100,
                    start_time_s=0.0, end_time_s=3.33,
                    server="near_player",
                    shots=[
                        Shot(
                            shot_number=1, frame=10, time_s=0.33,
                            player="near_player", stroke="serve",
                            placement=ShotPlacement(x=1.5, y=4.0, zone="t"),
                            confidence=0.8,
                        ),
                    ],
                    outcome="winner", outcome_player="near_player",
                    rally_length=1, review_status="approved",
                ),
            ],
        )

        output_file = tmp_path / "output.json"
        save_match_json(match, output_file)
        reloaded = load_match_json(output_file)

        assert reloaded.match_id == "test123"
        assert reloaded.points[0].review_status == "approved"
        assert reloaded.points[0].shots[0].stroke == "serve"
        assert reloaded.points[0].shots[0].placement.zone == "t"

    def test_save_preserves_review_status(self, tmp_path: Path):
        """Save preserves updated review_status values."""
        from court_vision.review_data import load_match_json, save_match_json

        match = MatchData(
            match_id="test456",
            source_url="test.mp4",
            metadata={},
            points=[
                Point(
                    point_number=1,
                    start_frame=0, end_frame=50,
                    start_time_s=0.0, end_time_s=1.67,
                    server=None,
                    shots=[],
                    outcome=None, outcome_player=None,
                    rally_length=0, review_status="corrected",
                ),
            ],
        )

        output_file = tmp_path / "output.json"
        save_match_json(match, output_file)

        with open(output_file) as f:
            raw = json.load(f)

        assert raw["points"][0]["review_status"] == "corrected"
