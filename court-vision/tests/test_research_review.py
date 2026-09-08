"""Tests for review packets and folding review verdicts into ground truth."""

import json
from pathlib import Path

import pytest

from court_vision.research.clips import Clip
from court_vision.research.review import (
    apply_review_to_ground_truth,
    load_review,
    write_review_packet,
)
from court_vision.research.scorecard import Scorecard
from court_vision.review_data import load_match_json, save_match_json
from court_vision.shot_classify import MatchData, Point, Shot

FPS = 30.0


def _shot(n: int, frame: int, player: str, stroke: str) -> Shot:
    return Shot(shot_number=n, frame=frame, time_s=frame / FPS, player=player, stroke=stroke,
                placement=None, confidence=0.9)


def _point(n: int, start: int, end: int, shots, winner=None) -> Point:
    return Point(point_number=n, start_frame=start, end_frame=end, start_time_s=start / FPS, end_time_s=end / FPS,
                 server=shots[0].player if shots else None, shots=list(shots), outcome=None, outcome_player=None,
                 rally_length=len(shots), winner=winner)


def _gt() -> MatchData:
    return MatchData(
        match_id="gt0001", source_url="clip.mp4", metadata={"fps": FPS},
        points=[
            _point(1, 0, 300, [_shot(1, 100, "near_player", "forehand"), _shot(2, 140, "far_player", "backhand")]),
            _point(2, 400, 700, [_shot(1, 500, "far_player", "serve")], winner="far_player"),
        ],
    )


@pytest.fixture
def gt_path(tmp_path: Path) -> Path:
    path = tmp_path / "gt.json"
    save_match_json(_gt(), path)
    return path


class TestApplyReviewToGroundTruth:
    def test_full_round_trip(self, gt_path: Path, tmp_path: Path):
        review = {
            "reviewer": "claude-code",
            "shots": [
                {"frame": 100, "verdict": "wrong", "stroke": "slice", "fix_ground_truth": True},
                {"frame": 141, "verdict": "false_positive", "fix_ground_truth": True},  # nearest GT shot is f140
                {"frame": 500, "verdict": "wrong", "stroke": "volley"},  # prediction verdict only: GT untouched
            ],
            "missing_shots": [{"frame": 200, "player": "far_player", "stroke": "volley"}],
            "points": [{"point_number": 1, "winner": "far_player", "server": "far_player"}],
        }
        out = tmp_path / "gt_fixed.json"
        result = apply_review_to_ground_truth(review, gt_path, out_path=out)
        assert result == out
        assert not gt_path.with_suffix(".bak.json").exists()  # no in-place write, no backup

        fixed = load_match_json(out)
        p1, p2 = fixed.points
        assert [(s.shot_number, s.frame, s.player, s.stroke) for s in p1.shots] == [
            (1, 100, "near_player", "slice"),
            (2, 200, "far_player", "volley"),
        ]
        assert p1.shots[1].time_s == pytest.approx(200 / FPS)
        assert p1.rally_length == 2
        assert p1.winner == "far_player"
        assert p1.outcome == "winner" and p1.outcome_player == "far_player"  # last hitter won
        assert p1.server == "far_player"
        assert p2.shots[0].stroke == "serve"  # no fix_ground_truth flag
        assert all(p.review_status == "corrected" for p in fixed.points)
        history = fixed.metadata["review_history"]
        assert len(history) == 1
        assert history[0]["reviewer"] == "claude-code"
        assert any("slice" in c for c in history[0]["changes"])
        assert any("removed GT shot f140" in c for c in history[0]["changes"])
        assert any("added GT shot f200" in c for c in history[0]["changes"])
        # the original file is untouched
        assert load_match_json(gt_path).points[0].shots[0].stroke == "forehand"

    def test_in_place_write_keeps_a_backup(self, gt_path: Path):
        review = {"shots": [{"frame": 100, "verdict": "wrong", "player": "far_player", "fix_ground_truth": True}]}
        result = apply_review_to_ground_truth(review, gt_path)
        assert result == gt_path
        backup = gt_path.with_suffix(".bak.json")
        assert backup.exists()
        assert load_match_json(backup).points[0].shots[0].player == "near_player"
        assert load_match_json(gt_path).points[0].shots[0].player == "far_player"

    def test_frame_correction_moves_the_shot_and_resorts(self, gt_path: Path, tmp_path: Path):
        review = {"shots": [{"frame": 150, "gt_frame": 100, "verdict": "wrong", "fix_ground_truth": True}]}
        out = tmp_path / "out.json"
        fixed = load_match_json(apply_review_to_ground_truth(review, gt_path, out_path=out))
        shots = fixed.points[0].shots
        assert [(s.shot_number, s.frame) for s in shots] == [(1, 140), (2, 150)]
        assert shots[1].time_s == pytest.approx(150 / FPS)

    def test_shots_outside_tolerance_are_ignored(self, gt_path: Path, tmp_path: Path):
        review = {"shots": [{"frame": 250, "verdict": "wrong", "stroke": "slice", "fix_ground_truth": True}]}
        out = tmp_path / "out.json"
        fixed = load_match_json(apply_review_to_ground_truth(review, gt_path, out_path=out, tolerance_frames=15))
        assert [s.stroke for s in fixed.points[0].shots] == ["forehand", "backhand"]
        assert fixed.metadata["review_history"][0]["changes"] == []

    def test_missing_shot_near_an_existing_one_is_not_duplicated(self, gt_path: Path, tmp_path: Path):
        review = {"missing_shots": [{"frame": 105, "player": "near_player", "stroke": "forehand"}]}
        out = tmp_path / "out.json"
        fixed = load_match_json(apply_review_to_ground_truth(review, gt_path, out_path=out))
        assert [s.frame for s in fixed.points[0].shots] == [100, 140]

    def test_missing_shot_outside_every_point_goes_to_the_nearest(self, gt_path: Path, tmp_path: Path):
        review = {"missing_shots": [{"frame": 380, "player": "near_player"}]}
        out = tmp_path / "out.json"
        fixed = load_match_json(apply_review_to_ground_truth(review, gt_path, out_path=out))
        assert [s.frame for s in fixed.points[1].shots] == [380, 500]
        assert fixed.points[1].shots[0].stroke == "forehand"  # default stroke

    def test_point_winner_error_when_last_hitter_lost(self, gt_path: Path, tmp_path: Path):
        review = {"points": [{"point_number": 1, "winner": "near_player"}, {"point_number": 9, "winner": "near_player"}]}
        out = tmp_path / "out.json"
        fixed = load_match_json(apply_review_to_ground_truth(review, gt_path, out_path=out))
        p1 = fixed.points[0]
        assert p1.winner == "near_player"
        assert p1.outcome == "error" and p1.outcome_player == "far_player"

    def test_explicit_fps_is_used_for_times(self, gt_path: Path, tmp_path: Path):
        review = {"missing_shots": [{"frame": 60, "player": "near_player"}]}
        out = tmp_path / "out.json"
        fixed = load_match_json(apply_review_to_ground_truth(review, gt_path, out_path=out, fps=60.0))
        assert fixed.points[0].shots[0].time_s == pytest.approx(1.0)

    def test_review_history_accumulates(self, gt_path: Path, tmp_path: Path):
        out1 = tmp_path / "one.json"
        out2 = tmp_path / "two.json"
        apply_review_to_ground_truth({"reviewer": "a"}, gt_path, out_path=out1)
        apply_review_to_ground_truth({"reviewer": "b"}, out1, out_path=out2)
        history = load_match_json(out2).metadata["review_history"]
        assert [h["reviewer"] for h in history] == ["a", "b"]


class TestLoadReview:
    def test_reads_json(self, tmp_path: Path):
        path = tmp_path / "review.json"
        path.write_text(json.dumps({"reviewer": "x", "shots": []}))
        assert load_review(path) == {"reviewer": "x", "shots": []}


class TestWriteReviewPacket:
    def test_packet_lists_scorecard_predictions_and_sheets(self, tmp_path: Path):
        clip = Clip(name="clipA", video=tmp_path / "clipA.mp4", notes="rainy day")
        sc = Scorecard(clip="clipA", score=0.5, stroke_confusion={"serve": {"serve": 1, "overhead": 1}})
        pred = MatchData(match_id="p", source_url="clipA.mp4", metadata={}, points=[
            _point(1, 0, 300, [_shot(1, 100, "near_player", "forehand")], winner="near_player"),
        ])
        sheets = [{"file": "keyframes/sheet_01.jpg", "items": [
            {"kind": "pred", "frame": 100, "point": 1, "pred": "near_player forehand", "gt": "GT f100 near_player forehand", "verdict": "OK"},
            {"kind": "missed", "frame": 140, "point": 1, "gt": "far_player backhand"},
            {"kind": "point-start", "frame": 0, "point": 1},
        ]}]
        path = write_review_packet(tmp_path, clip, sc, pred, _gt(), sheets, {"contact_min_gap_s": 0.6})
        assert path == tmp_path / "REVIEW.md"
        text = path.read_text()
        assert "# Review packet — clipA" in text
        assert "rainy day" in text
        assert '"contact_min_gap_s": 0.6' in text
        assert sc.summary() in text
        assert "serve: serve=1, overhead=1" in text
        assert "**Point 1** frames 0-300 server=near_player winner=near_player" in text
        assert "**forehand**" in text
        assert "## Ground truth (human-corrected)" in text
        assert "keyframes/sheet_01.jpg" in text
        assert "-> OK" in text and "MISSED" in text and "point-start" in text

    def test_packet_without_ground_truth(self, tmp_path: Path):
        clip = Clip(name="clipB", video=tmp_path / "b.mp4")
        pred = MatchData(match_id="p", source_url="b.mp4", metadata={}, points=[])
        text = write_review_packet(tmp_path, clip, Scorecard(clip="clipB"), pred, None, [], {}).read_text()
        assert "Ground truth" not in text
        assert "## How to review" in text
