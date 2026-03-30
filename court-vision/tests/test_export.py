"""Tests for the export module."""

import json
from pathlib import Path

import pytest

from court_vision.export import export_json
from court_vision.shot_classify import MatchData, Point, Shot, ShotPlacement


class TestExportJson:
    def _sample_match(self) -> MatchData:
        """Create a minimal MatchData for testing."""
        return MatchData(
            match_id="test_abc123",
            source_url="test.mp4",
            metadata={"players": ["near_player", "far_player"], "date_processed": "2026-03-30"},
            points=[
                Point(
                    point_number=1,
                    start_frame=100, end_frame=200,
                    start_time_s=3.33, end_time_s=6.67,
                    server="near_player",
                    shots=[
                        Shot(
                            shot_number=1, frame=110, time_s=3.67,
                            player="near_player", stroke="serve",
                            placement=ShotPlacement(x=4.0, y=1.0, zone="wide"),
                            confidence=0.9,
                        ),
                        Shot(
                            shot_number=2, frame=140, time_s=4.67,
                            player="far_player", stroke="forehand",
                            placement=ShotPlacement(x=-2.0, y=8.5, zone="crosscourt_deep"),
                            confidence=0.85,
                        ),
                    ],
                    outcome="winner",
                    outcome_player="far_player",
                    rally_length=2,
                ),
            ],
        )

    def test_exports_valid_json(self, tmp_path: Path):
        """Exports match data as valid JSON."""
        match = self._sample_match()
        output = tmp_path / "match.json"
        export_json(match, output)

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["match_id"] == "test_abc123"
        assert len(data["points"]) == 1

    def test_json_structure_matches_spec(self, tmp_path: Path):
        """JSON structure matches the spec format."""
        match = self._sample_match()
        output = tmp_path / "match.json"
        export_json(match, output)

        data = json.loads(output.read_text())
        point = data["points"][0]
        assert "point_number" in point
        assert "shots" in point
        assert "outcome" in point
        assert "review_status" in point

        shot = point["shots"][0]
        assert "shot_number" in shot
        assert "frame" in shot
        assert "player" in shot
        assert "stroke" in shot
        assert "placement" in shot
        assert "confidence" in shot

        placement = shot["placement"]
        assert "x" in placement
        assert "y" in placement
        assert "zone" in placement

    def test_json_has_metadata(self, tmp_path: Path):
        """JSON includes source_url and metadata."""
        match = self._sample_match()
        output = tmp_path / "match.json"
        export_json(match, output)

        data = json.loads(output.read_text())
        assert data["source_url"] == "test.mp4"
        assert "metadata" in data
        assert data["metadata"]["players"] == ["near_player", "far_player"]
