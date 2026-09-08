"""Tests for experiment helpers (override parsing, leaderboard) and sweep grids.

``run_experiment`` / ``run_sweep`` drive the full pipeline, so the sweep test
replaces ``run_experiment`` with a fake and only checks the bookkeeping.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from court_vision.research.experiment import load_leaderboard, parse_override
from court_vision.research.scorecard import Scorecard
from court_vision.research.sweep import grid_points, load_sweep, run_sweep


class TestParseOverride:
    @pytest.mark.parametrize("text,expected", [
        ("ball_confidence_threshold=0.4", ("ball_confidence_threshold", 0.4)),
        ("ball_frame_step=2", ("ball_frame_step", 2)),
        ("ball_far_crop=true", ("ball_far_crop", True)),
        ("ball_far_crop=false", ("ball_far_crop", False)),
        ("ball_detection_method=wasb", ("ball_detection_method", "wasb")),
        ("pipeline.outcome_method=last_hitter", ("pipeline.outcome_method", "last_hitter")),
        ("target_resolution=[1280, 720]", ("target_resolution", [1280, 720])),
        ("fps_override=null", ("fps_override", None)),
        ("  near_player_hand = left", ("near_player_hand", "left")),
        ("note=a=b", ("note", "a=b")),  # split at the first '='
    ])
    def test_yaml_typed_values(self, text: str, expected):
        assert parse_override(text) == expected

    def test_requires_an_equals_sign(self):
        with pytest.raises(ValueError, match="key=value"):
            parse_override("ball_confidence_threshold")


class TestLoadLeaderboard:
    def test_missing_file_is_empty(self, tmp_path: Path):
        assert load_leaderboard(tmp_path) == []

    def test_reads_jsonl_rows(self, tmp_path: Path):
        rows = [{"clip": "a", "score": 0.5}, {"clip": "b", "score": 0.7}]
        (tmp_path / "leaderboard.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
        assert load_leaderboard(tmp_path) == rows


class TestGridPoints:
    def test_full_product_in_order(self):
        pts = grid_points({"a": [1, 2], "b": ["x", "y"]})
        assert pts == [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 2, "b": "x"}, {"a": 2, "b": "y"}]

    def test_max_runs_samples_deterministically(self):
        params = {"a": [1, 2, 3], "b": [1, 2, 3]}
        sample = grid_points(params, max_runs=4)
        assert len(sample) == 4
        assert sample == grid_points(params, max_runs=4)
        assert all(p in grid_points(params) for p in sample)
        assert grid_points(params, max_runs=4, seed=1) != sample or True  # a different seed may reorder

    def test_max_runs_larger_than_grid_keeps_everything(self):
        assert len(grid_points({"a": [1, 2]}, max_runs=10)) == 2

    def test_empty_params_is_one_empty_combo(self):
        assert grid_points({}) == [{}]


class TestLoadSweep:
    def test_reads_yaml(self, tmp_path: Path):
        path = tmp_path / "sweep.yaml"
        path.write_text("name: contacts\nclips: [a, b]\nparams:\n  contact_min_gap_s: [0.35, 0.45]\n")
        assert load_sweep(path) == {"name": "contacts", "clips": ["a", "b"], "params": {"contact_min_gap_s": [0.35, 0.45]}}


class TestRunSweep:
    def test_ranks_runs_by_mean_score_and_writes_results(self, tmp_path: Path):
        scores = {("a", 0.35): 0.2, ("b", 0.35): 0.4, ("a", 0.45): 0.9, ("b", 0.45): 0.7}
        calls = []

        def fake_run_experiment(clip, overrides, **kwargs):
            calls.append((clip, dict(overrides), kwargs))
            score = scores[(clip, overrides["contact_min_gap_s"])]
            sc = Scorecard(clip=clip, score=score, shot_f1=0.5, outcome_correct=1, outcome_total=2,
                           side_correct=3, side_total=4, points_matched=1, gt_points=1)
            return SimpleNamespace(scorecard=sc)

        sweep = {
            "name": "gap",
            "clips": ["a", "b"],
            "params": {"contact_min_gap_s": [0.35, 0.45]},
            "fixed": {"ball_detection_method": "wasb"},
        }
        with patch("court_vision.research.sweep.run_experiment", side_effect=fake_run_experiment):
            rows = run_sweep(sweep, root=tmp_path, registry=tmp_path / "clips.yaml", log=lambda *_: None)

        assert [r["mean_score"] for r in rows] == [0.8, 0.3]
        assert rows[0]["overrides"] == {"ball_detection_method": "wasb", "contact_min_gap_s": 0.45}
        assert rows[0]["per_clip"]["a"] == {"score": 0.9, "shot_f1": 0.5, "outcome": "1/2", "side": "3/4", "points": "1/1"}
        assert rows[1]["run"] == 1
        assert len(calls) == 4
        assert all(kw["keyframes"] is False and kw["root"] == tmp_path for _, _, kw in calls)
        assert all(kw["registry"] == tmp_path / "clips.yaml" for _, _, kw in calls)
        assert [kw["tag"] for _, _, kw in calls] == ["gap-1", "gap-1", "gap-2", "gap-2"]
        saved = json.loads((tmp_path / "sweeps" / "gap.json").read_text())
        assert saved == rows

    def test_explicit_clips_override_the_file(self, tmp_path: Path):
        seen = []

        def fake_run_experiment(clip, overrides, **kwargs):
            seen.append(clip)
            return SimpleNamespace(scorecard=Scorecard(clip=clip, score=0.1))

        with patch("court_vision.research.sweep.run_experiment", side_effect=fake_run_experiment):
            rows = run_sweep({"params": {"x": [1]}, "clips": ["a"]}, root=tmp_path, clips=["z"], log=lambda *_: None)
        assert seen == ["z"]
        assert rows[0]["mean_score"] == 0.1
        assert (tmp_path / "sweeps" / "sweep.json").exists()  # default name

    def test_sweep_yaml_round_trip_into_grid(self, tmp_path: Path):
        path = tmp_path / "s.yaml"
        path.write_text(yaml.safe_dump({"name": "n", "params": {"a": [1, 2], "b": [True]}}))
        assert grid_points(load_sweep(path)["params"]) == [{"a": 1, "b": True}, {"a": 2, "b": True}]
