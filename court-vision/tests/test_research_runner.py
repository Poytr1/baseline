"""Tests for the staged runner's on-disk cache keys (no pipeline stage is executed)."""

import json
from pathlib import Path

import pytest

from court_vision.config import STAGE_DEPS, STAGE_PARAMS, PipelineConfig, apply_overrides
from court_vision.research.runner import (
    STAGE_SOURCES,
    StageCache,
    _source_hash,
    _timeline_from_json,
    _timeline_to_json,
    stage_params,
)
from court_vision.scoreboard import ScoreRow, ScoreSample, ScoreTimeline


@pytest.fixture
def cache(tmp_path: Path) -> StageCache:
    return StageCache(tmp_path / "cache")


class TestStageParams:
    def test_only_the_stage_knobs(self):
        p = stage_params(PipelineConfig(), "ball")
        assert set(p) == set(STAGE_PARAMS["ball"])
        assert p["ball_detection_method"] == "wasb"

    def test_unknown_stage_is_empty(self):
        assert stage_params(PipelineConfig(), "nope") == {}


class TestStageCacheKey:
    def test_stable_for_identical_inputs(self, cache: StageCache):
        a = cache.key("ball", "clipA", PipelineConfig(), {})
        b = cache.key("ball", "clipA", PipelineConfig(), {})
        assert a == b
        assert len(a) == 12 and all(c in "0123456789abcdef" for c in a)

    def test_changes_with_the_stage_knobs(self, cache: StageCache):
        base = cache.key("ball", "clipA", PipelineConfig(), {})
        changed = cache.key("ball", "clipA", apply_overrides(PipelineConfig(), {"ball_confidence_threshold": 0.9}), {})
        assert changed != base

    def test_ignores_knobs_of_other_stages(self, cache: StageCache):
        base = cache.key("ball", "clipA", PipelineConfig(), {})
        tweaked = apply_overrides(PipelineConfig(), {"contact_min_gap_s": 0.9, "outcome_method": "last_hitter"})
        assert cache.key("ball", "clipA", tweaked, {}) == base
        assert cache.key("shots", "clipA", tweaked, {}) != cache.key("shots", "clipA", PipelineConfig(), {})

    def test_changes_with_the_clip_and_the_stage(self, cache: StageCache):
        cfg = PipelineConfig()
        assert cache.key("ball", "clipA", cfg, {}) != cache.key("ball", "clipB", cfg, {})
        assert cache.key("ball", "clipA", cfg, {}) != cache.key("players", "clipA", cfg, {})

    def test_upstream_keys_propagate_through_dependencies(self, cache: StageCache):
        cfg = PipelineConfig()
        up1 = {"ingest": "aaaa", "scene": "bbbb", "court": "cccc"}
        up2 = {**up1, "court": "dddd"}
        # "ball" depends on court...
        assert cache.key("ball", "clipA", cfg, up1) != cache.key("ball", "clipA", cfg, up2)
        # ...but "scoreboard" only depends on ingest
        assert "court" not in STAGE_DEPS["scoreboard"]
        assert cache.key("scoreboard", "clipA", cfg, up1) == cache.key("scoreboard", "clipA", cfg, up2)

    def test_missing_upstream_keys_are_tolerated(self, cache: StageCache):
        cfg = PipelineConfig()
        assert cache.key("shots", "clipA", cfg, {}) == cache.key("shots", "clipA", cfg, {})
        assert cache.key("shots", "clipA", cfg, {}) != cache.key("shots", "clipA", cfg, {"players": "x"})

    def test_source_hash_tracks_the_stage_files(self):
        assert set(STAGE_SOURCES) == set(STAGE_PARAMS)
        for stage in STAGE_SOURCES:
            h = _source_hash(stage)
            assert len(h) == 10
        assert _source_hash("ball") != _source_hash("players")
        assert _source_hash("nope") == _source_hash("also-nope")  # nothing hashed


class TestStageCacheStorage:
    def test_root_is_created(self, tmp_path: Path):
        root = tmp_path / "deep" / "cache"
        StageCache(root)
        assert root.is_dir()

    def test_path_layout(self, cache: StageCache):
        p = cache.path("clipA", "ball", "abc123")
        assert p == cache.root / "clipA" / "ball-abc123.json"
        assert p.parent.is_dir()

    def test_save_and_load_round_trip(self, cache: StageCache):
        assert cache.load("clipA", "ball", "k1") is None
        cache.save("clipA", "ball", "k1", [{"frame_index": 1, "x": 2.5}])
        assert cache.load("clipA", "ball", "k1") == [{"frame_index": 1, "x": 2.5}]
        assert cache.load("clipA", "ball", "k2") is None

    def test_save_handles_numpy_values(self, cache: StageCache):
        import numpy as np

        cache.save("clipA", "court", "k", {"h": np.eye(2), "v": np.float32(0.5), "n": np.int64(3)})
        assert cache.load("clipA", "court", "k") == {"h": [[1.0, 0.0], [0.0, 1.0]], "v": 0.5, "n": 3}


class TestTimelineJson:
    def test_round_trip(self):
        tl = ScoreTimeline(roi=(1, 2, 3, 4), samples=[
            ScoreSample(frame=0, rows=[ScoreRow("A", [6, 4], "30", True, "A 6 4 30"), ScoreRow("B", [3, 5], "15")]),
            ScoreSample(frame=15, rows=[]),
        ])
        raw = json.loads(json.dumps(_timeline_to_json(tl)))
        back = _timeline_from_json(raw)
        assert back == tl
        assert back.roi == (1, 2, 3, 4)

    def test_none_and_empty(self):
        assert _timeline_to_json(None) is None
        assert _timeline_from_json(None) is None
        assert _timeline_from_json({}) is None
        assert _timeline_from_json({"roi": None, "samples": []}) == ScoreTimeline()
