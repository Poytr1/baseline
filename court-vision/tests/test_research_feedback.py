"""Tests for the human feedback log (JSONL per clip) and its review conversion."""

import json
from pathlib import Path

import pytest

from court_vision.research.feedback import (
    add_feedback,
    apply_feedback,
    feedback_path,
    feedback_to_review,
    load_feedback,
)
from court_vision.review_data import load_match_json, save_match_json
from court_vision.shot_classify import MatchData, Point, Shot


class TestFeedbackToReview:
    def test_empty(self):
        assert feedback_to_review([]) == {"reviewer": "feedback", "shots": [], "missing_shots": [], "points": [], "notes": ""}

    def test_stroke_or_player_correction_becomes_a_fixed_wrong_verdict(self):
        review = feedback_to_review([{"frame": 1216, "player": "near_player", "stroke": "forehand"}])
        assert review["shots"] == [{
            "frame": 1216, "verdict": "wrong", "stroke": "forehand", "player": "near_player", "fix_ground_truth": True,
        }]
        assert review["missing_shots"] == [] and review["points"] == []

    def test_delete_becomes_a_false_positive(self):
        review = feedback_to_review([{"frame": "888", "delete": True}])
        assert review["shots"] == [{"frame": 888, "verdict": "false_positive", "fix_ground_truth": True}]

    def test_add_becomes_a_missing_shot(self):
        review = feedback_to_review([{"frame": 129, "add": True, "player": "far_player", "stroke": "slice"}])
        assert review["missing_shots"] == [{"frame": 129, "player": "far_player", "stroke": "slice"}]
        assert review["shots"] == []

    def test_point_winner_and_server(self):
        review = feedback_to_review([{"point": 2, "winner": "far_player"}, {"point": "3", "server": "near_player"}])
        assert review["points"] == [
            {"point_number": 2, "winner": "far_player", "server": None},
            {"point_number": 3, "winner": None, "server": "near_player"},
        ]

    def test_point_without_winner_or_server_is_ignored(self):
        review = feedback_to_review([{"point": 2}])
        assert review["points"] == [] and review["shots"] == []

    def test_notes_are_concatenated(self):
        review = feedback_to_review([{"note": "first"}, {"frame": 5, "stroke": "serve", "note": "second"}])
        assert review["notes"] == "first\nsecond\n"
        assert len(review["shots"]) == 1


class TestFeedbackLog:
    def test_add_and_load_round_trip(self, tmp_path: Path):
        root = tmp_path / "fb"
        p1 = add_feedback("clipA", {"frame": 10, "stroke": "serve"}, root=root)
        p2 = add_feedback("clipA", {"point": 1, "winner": "far_player"}, root=root, author="claude")
        assert p1 == p2 == root / "clipA.jsonl"
        assert feedback_path("clipA", root) == p1
        entries = load_feedback("clipA", root=root)
        assert len(entries) == 2
        assert entries[0]["frame"] == 10 and entries[0]["stroke"] == "serve"
        assert entries[0]["author"] == "human" and "time" in entries[0]
        assert entries[1]["author"] == "claude" and entries[1]["winner"] == "far_player"
        # one JSON object per line
        assert len(p1.read_text().strip().splitlines()) == 2
        assert all(json.loads(line) for line in p1.read_text().splitlines())

    def test_load_unknown_clip_is_empty(self, tmp_path: Path):
        assert load_feedback("nope", root=tmp_path / "fb") == []

    def test_blank_lines_are_skipped(self, tmp_path: Path):
        root = tmp_path / "fb"
        root.mkdir()
        (root / "clipA.jsonl").write_text('{"frame": 1}\n\n   \n{"frame": 2}\n')
        assert [e["frame"] for e in load_feedback("clipA", root=root)] == [1, 2]


class TestApplyFeedback:
    def _setup(self, tmp_path: Path) -> tuple[Path, Path]:
        registry = tmp_path / "research" / "clips.yaml"
        registry.parent.mkdir()
        registry.write_text("clips:\n  clipA:\n    video: clipA.mp4\n    ground_truth: gt.json\n  noGT:\n    video: b.mp4\n")
        gt_path = tmp_path / "gt.json"
        shots = [Shot(1, 100, 100 / 30.0, "near_player", "forehand", None, 0.9)]
        gt = MatchData("m", "clipA.mp4", {"fps": 30.0}, [
            Point(1, 0, 300, 0.0, 10.0, "near_player", shots, None, None, 1),
        ])
        save_match_json(gt, gt_path)
        return registry, gt_path

    def test_applies_recorded_feedback_to_ground_truth(self, tmp_path: Path):
        registry, gt_path = self._setup(tmp_path)
        root = tmp_path / "fb"
        add_feedback("clipA", {"frame": 100, "stroke": "slice"}, root=root)
        add_feedback("clipA", {"point": 1, "winner": "far_player"}, root=root)
        out = apply_feedback("clipA", root=root, registry=registry)
        assert out is not None and Path(out).samefile(gt_path)
        fixed = load_match_json(gt_path)
        assert fixed.points[0].shots[0].stroke == "slice"
        assert fixed.points[0].winner == "far_player"
        assert fixed.metadata["review_history"][0]["reviewer"] == "feedback"

    def test_nothing_to_apply(self, tmp_path: Path):
        registry, _ = self._setup(tmp_path)
        assert apply_feedback("clipA", root=tmp_path / "fb", registry=registry) is None

    def test_clip_without_ground_truth(self, tmp_path: Path):
        registry, _ = self._setup(tmp_path)
        root = tmp_path / "fb"
        add_feedback("noGT", {"frame": 1, "stroke": "serve"}, root=root)
        assert apply_feedback("noGT", root=root, registry=registry) is None

    def test_unknown_clip_raises(self, tmp_path: Path):
        registry, _ = self._setup(tmp_path)
        with pytest.raises(KeyError):
            apply_feedback("missing", root=tmp_path / "fb", registry=registry)
