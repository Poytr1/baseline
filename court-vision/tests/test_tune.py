"""Tests for pipeline tuning module."""

import json

from court_vision.scene_filter import GameplaySegment
from court_vision.shot_classify import MatchData, Point, Shot
from court_vision.tune import grid_search, load_tracking_results, reclassify_shots


class TestLoadTrackingResults:
    """Tests for load_tracking_results deserialization."""

    def test_loads_empty_list(self, tmp_path):
        """Empty JSON array returns empty list."""
        p = tmp_path / "tracking.json"
        p.write_text("[]")
        results = load_tracking_results(p)
        assert results == []

    def test_loads_single_frame(self, tmp_path):
        """Single frame with ball and player loads correctly."""
        data = [
            {
                "frame_index": 5,
                "ball": {
                    "frame_index": 5,
                    "x": 100.0,
                    "y": 200.0,
                    "confidence": 0.9,
                    "interpolated": False,
                },
                "players": [
                    {
                        "frame_index": 5,
                        "bbox": [10, 20, 110, 420],
                        "confidence": 0.95,
                        "court_position": None,
                        "role": "near_player",
                    }
                ],
                "poses": [],
            }
        ]
        p = tmp_path / "tracking.json"
        p.write_text(json.dumps(data))
        results = load_tracking_results(p)
        assert len(results) == 1
        assert results[0].frame_index == 5
        assert results[0].ball is not None
        assert results[0].ball.x == 100.0
        assert len(results[0].players) == 1


class TestReclassifyShots:
    """Tests for reclassify_shots with different parameters."""

    def test_returns_match_data(self, tmp_path):
        """Reclassification produces valid MatchData."""
        data = [
            {
                "frame_index": i,
                "ball": {
                    "frame_index": i,
                    "x": 100.0 + i,
                    "y": 200.0,
                    "confidence": 0.9,
                    "interpolated": False,
                },
                "players": [
                    {
                        "frame_index": i,
                        "bbox": [80, 150, 180, 450],
                        "confidence": 0.95,
                        "court_position": None,
                        "role": "near_player",
                    },
                    {
                        "frame_index": i,
                        "bbox": [500, 50, 600, 250],
                        "confidence": 0.90,
                        "court_position": None,
                        "role": "far_player",
                    },
                ],
                "poses": [],
            }
            for i in range(30)
        ]
        p = tmp_path / "tracking.json"
        p.write_text(json.dumps(data))
        results = load_tracking_results(p)
        segments = [
            GameplaySegment(
                start_frame=0, end_frame=29,
                start_time_s=0.0, end_time_s=1.0, frame_count=30,
            )
        ]
        match = reclassify_shots(results, segments, "test.mp4", fps=30.0)
        assert match.match_id is not None
        assert len(match.points) == 1

    def test_different_params_give_different_shots(self, tmp_path):
        """Tighter min_frames_between_contacts produces more or equal shots than loose."""
        data = []
        for i in range(30):
            if i < 10:
                ball_x, ball_y = 100.0, 300.0
            elif i >= 15:
                ball_x, ball_y = 550.0, 150.0
            else:
                ball_x, ball_y = 300.0, 300.0

            data.append({
                "frame_index": i,
                "ball": {
                    "frame_index": i,
                    "x": ball_x,
                    "y": ball_y,
                    "confidence": 0.9,
                    "interpolated": False,
                },
                "players": [
                    {
                        "frame_index": i,
                        "bbox": [80, 150, 180, 450],
                        "confidence": 0.95,
                        "court_position": None,
                        "role": "near_player",
                    },
                    {
                        "frame_index": i,
                        "bbox": [500, 50, 600, 250],
                        "confidence": 0.90,
                        "court_position": None,
                        "role": "far_player",
                    },
                ],
                "poses": [],
            })

        p = tmp_path / "tracking.json"
        p.write_text(json.dumps(data))
        results = load_tracking_results(p)
        segments = [
            GameplaySegment(
                start_frame=0, end_frame=29,
                start_time_s=0.0, end_time_s=1.0, frame_count=30,
            )
        ]

        match_tight = reclassify_shots(
            results, segments, "test.mp4", fps=30.0, min_frames_between_contacts=3,
        )
        match_loose = reclassify_shots(
            results, segments, "test.mp4", fps=30.0, min_frames_between_contacts=20,
        )

        tight_shots = sum(len(pt.shots) for pt in match_tight.points)
        loose_shots = sum(len(pt.shots) for pt in match_loose.points)
        assert tight_shots >= loose_shots


class TestGridSearch:
    """Tests for grid_search over parameter combinations."""

    def test_returns_sorted_results(self, tmp_path):
        """Grid search returns results sorted by F1 descending."""
        from court_vision.review_data import save_match_json

        gt = MatchData(
            match_id="test",
            source_url="test.mp4",
            metadata={},
            points=[
                Point(
                    point_number=1,
                    start_frame=0,
                    end_frame=29,
                    start_time_s=0.0,
                    end_time_s=1.0,
                    server=None,
                    shots=[
                        Shot(
                            shot_number=1,
                            frame=5,
                            time_s=0.17,
                            player="near_player",
                            stroke="forehand",
                            placement=None,
                            confidence=1.0,
                        ),
                    ],
                    outcome=None,
                    outcome_player=None,
                    rally_length=1,
                )
            ],
        )

        gt_path = tmp_path / "gt.json"
        save_match_json(gt, gt_path)

        data = []
        for i in range(30):
            data.append({
                "frame_index": i,
                "ball": {
                    "frame_index": i,
                    "x": 100.0,
                    "y": 300.0,
                    "confidence": 0.9,
                    "interpolated": False,
                },
                "players": [
                    {
                        "frame_index": i,
                        "bbox": [80, 150, 180, 450],
                        "confidence": 0.95,
                        "court_position": None,
                        "role": "near_player",
                    },
                ],
                "poses": [],
            })
        tracking_path = tmp_path / "tracking.json"
        tracking_path.write_text(json.dumps(data))

        results = grid_search(
            gt_path,
            tracking_path,
            source="test.mp4",
            fps=30.0,
            proximity_values=[50.0, 100.0],
            min_frames_values=[3, 5],
        )

        assert len(results) == 4  # 2 x 2 combinations
        for i in range(len(results) - 1):
            assert results[i].evaluation.f1 >= results[i + 1].evaluation.f1
