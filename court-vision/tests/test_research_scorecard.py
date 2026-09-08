"""Tests for the research scorecard (point matching + composite scoring)."""

import json
from pathlib import Path

import pytest

from court_vision.ball_tracker import BallDetection
from court_vision.player_detect import FrameTrackingResult, PlayerDetection
from court_vision.research.scorecard import (
    Scorecard,
    _composite,
    _overlap,
    load_scorecard,
    match_points,
    score_run,
)
from court_vision.scene_filter import GameplaySegment
from court_vision.shot_classify import MatchData, Point, Shot

FPS = 30.0


def _shot(n: int, frame: int, player: str, stroke: str) -> Shot:
    return Shot(shot_number=n, frame=frame, time_s=frame / FPS, player=player, stroke=stroke,
                placement=None, confidence=0.9)


def _point(n: int, start: int, end: int, shots=(), winner=None, outcome=None, outcome_player=None,
           server=None, source=None) -> Point:
    shots = list(shots)
    return Point(point_number=n, start_frame=start, end_frame=end, start_time_s=start / FPS, end_time_s=end / FPS,
                 server=server or (shots[0].player if shots else None), shots=shots, outcome=outcome,
                 outcome_player=outcome_player, rally_length=len(shots), winner=winner, outcome_source=source)


def _match(points, source: str = "clip.mp4") -> MatchData:
    return MatchData(match_id="abc123", source_url=source, metadata={"fps": FPS}, points=list(points))


def _tracking(frame: int, ball: str | None, both_players: bool) -> FrameTrackingResult:
    b = None
    if ball == "real":
        b = BallDetection(frame, 1.0, 1.0, 0.9)
    elif ball == "interp":
        b = BallDetection(frame, 1.0, 1.0, 0.4, interpolated=True)
    players = [PlayerDetection(frame, (0.0, 0.0, 10.0, 10.0), 0.9, role="near_player")]
    if both_players:
        players.append(PlayerDetection(frame, (20.0, 0.0, 30.0, 10.0), 0.9, role="far_player"))
    return FrameTrackingResult(frame, b, players, [])


def _rally_shots():
    return [_shot(1, 10, "near_player", "serve"), _shot(2, 40, "far_player", "backhand"), _shot(3, 70, "near_player", "forehand")]


class TestOverlapAndMatchPoints:
    def test_overlap_is_frame_iou(self):
        assert _overlap(_point(1, 0, 99), _point(1, 0, 99)) == 1.0
        assert _overlap(_point(1, 0, 99), _point(1, 50, 149)) == pytest.approx(50 / 150)
        assert _overlap(_point(1, 0, 99), _point(1, 100, 199)) == 0.0

    def test_identical_points_all_match(self):
        gt = _match([_point(1, 0, 100), _point(2, 200, 300)])
        pred = _match([_point(1, 0, 100), _point(2, 200, 300)])
        pairs = match_points(gt, pred)
        assert [(g.point_number, p.point_number) for g, p in pairs] == [(1, 1), (2, 2)]

    def test_each_prediction_is_used_once(self):
        gt = _match([_point(1, 0, 100), _point(2, 60, 160)])
        pred = _match([_point(1, 0, 100)])
        pairs = match_points(gt, pred)
        assert [(g.point_number, p.point_number) for g, p in pairs] == [(1, 1)]

    def test_best_iou_wins(self):
        gt = _match([_point(1, 100, 200)])
        pred = _match([_point(1, 0, 130), _point(2, 90, 210)])
        pairs = match_points(gt, pred)
        assert [(g.point_number, p.point_number) for g, p in pairs] == [(1, 2)]

    def test_low_overlap_is_not_a_match(self):
        gt = _match([_point(1, 0, 100)])
        pred = _match([_point(1, 90, 400)])
        assert match_points(gt, pred) == []
        assert len(match_points(gt, pred, min_iou=0.01)) == 1

    def test_empty_inputs(self):
        assert match_points(_match([]), _match([_point(1, 0, 10)])) == []
        assert match_points(_match([_point(1, 0, 10)]), _match([])) == []


class TestScoreRunWithoutGroundTruth:
    def test_only_stage_health_is_scored(self):
        tracking = [
            _tracking(0, "real", True), _tracking(1, "real", True),
            _tracking(2, "interp", False), _tracking(3, None, False),
        ]
        sc = score_run("clipA", None, _match([]), tracking, [GameplaySegment(0, 3, 0.0, 0.1, 4)], [True, False], FPS)
        assert sc.clip == "clipA"
        assert sc.ball_coverage == pytest.approx(0.5)
        assert sc.ball_coverage_any == pytest.approx(0.75)
        assert sc.players_both_rate == pytest.approx(0.5)
        assert sc.court_success_rate == pytest.approx(0.5)
        assert sc.gt_points == 0 and sc.pred_points == 0 and sc.shot_f1 == 0.0
        assert any("no ground truth" in n for n in sc.notes)
        assert sc.score == pytest.approx(_composite(sc, has_gt=False))
        assert sc.score == pytest.approx(0.6 * 0.5 + 0.3 * 0.5 + 0.1 * 0.5)

    def test_empty_tracking_and_court(self):
        sc = score_run("clipA", None, _match([]), [], [], [], FPS)
        assert sc.ball_coverage == 0.0
        assert sc.court_success_rate == 0.0
        assert sc.score == 0.0


class TestScoreRunWithGroundTruth:
    def _gt(self) -> MatchData:
        return _match([_point(1, 0, 100, _rally_shots(), winner="near_player")])

    def test_perfect_prediction_scores_one(self):
        gt = self._gt()
        pred = _match([_point(1, 0, 100, _rally_shots(), winner="near_player", source="trajectory")])
        tracking = [_tracking(i, "real", True) for i in range(4)]
        sc = score_run("clipA", gt, pred, tracking, [GameplaySegment(0, 100, 0.0, 3.3, 101)], [True], FPS)
        assert (sc.gt_points, sc.pred_points, sc.points_matched) == (1, 1, 1)
        assert sc.point_precision == 1.0 and sc.point_recall == 1.0
        assert (sc.outcome_total, sc.outcome_correct, sc.outcome_accuracy) == (1, 1, 1.0)
        assert sc.outcome_sources == {"trajectory": 1}
        assert (sc.server_total, sc.server_correct) == (1, 1)
        assert (sc.shot_tp, sc.shot_fp, sc.shot_fn) == (3, 0, 0)
        assert sc.shot_precision == sc.shot_recall == sc.shot_f1 == 1.0
        assert sc.player_accuracy == 1.0 and sc.stroke_accuracy == 1.0
        assert (sc.side_total, sc.side_correct, sc.side_accuracy) == (2, 2, 1.0)
        assert (sc.serve_precision, sc.serve_recall) == (1.0, 1.0)
        assert (sc.slice_precision, sc.slice_recall) == (1.0, 1.0)  # no slices anywhere: vacuously right
        assert sc.stroke_confusion == {"serve": {"serve": 1}, "backhand": {"backhand": 1}, "forehand": {"forehand": 1}}
        assert sc.score == pytest.approx(1.0)
        assert sc.ball_detect_rate is None  # no stage labels given

    def test_imperfect_prediction(self):
        gt = self._gt()
        pred_shots = [
            _shot(1, 12, "near_player", "serve"),  # within tolerance (15 frames at 30 fps)
            _shot(2, 40, "far_player", "forehand"),  # wrong side
            _shot(3, 95, "far_player", "slice"),  # false positive; GT forehand at 70 is missed
        ]
        pred = _match([_point(1, 0, 100, pred_shots, winner="far_player", source="last_hitter")])
        sc = score_run("clipA", gt, pred, [], [GameplaySegment(0, 100, 0.0, 3.3, 101)], [True], FPS)
        assert (sc.shot_tp, sc.shot_fp, sc.shot_fn) == (2, 1, 1)
        assert sc.shot_precision == pytest.approx(2 / 3)
        assert sc.shot_recall == pytest.approx(2 / 3)
        assert sc.stroke_accuracy == pytest.approx(0.5)
        assert (sc.side_total, sc.side_correct, sc.side_accuracy) == (1, 0, 0.0)
        assert sc.stroke_confusion == {"serve": {"serve": 1}, "backhand": {"forehand": 1}}
        assert (sc.outcome_total, sc.outcome_correct) == (1, 0)
        assert sc.outcome_sources == {"last_hitter": 1}
        assert (sc.serve_precision, sc.serve_recall) == (1.0, 1.0)
        # A slice was predicted although GT has none: both P and R are charged for it.
        assert (sc.slice_precision, sc.slice_recall) == (0.0, 0.0)
        assert 0.0 < sc.score < 1.0

    def test_winner_falls_back_to_outcome_fields(self):
        gt = _match([_point(1, 0, 100, _rally_shots(), outcome="error", outcome_player="near_player")])  # far won
        pred = _match([_point(1, 0, 100, _rally_shots(), outcome="winner", outcome_player="far_player")])
        sc = score_run("clipA", gt, pred, [], [], [], FPS)
        assert (sc.outcome_total, sc.outcome_correct) == (1, 1)
        assert sc.outcome_sources == {"none": 1}

    def test_unmatched_points_count_against_precision_and_recall(self):
        gt = _match([_point(1, 0, 100, _rally_shots(), winner="near_player"), _point(2, 500, 600, winner="far_player")])
        pred = _match([_point(1, 0, 100, _rally_shots(), winner="near_player"), _point(2, 800, 900)])
        sc = score_run("clipA", gt, pred, [], [], [], FPS)
        assert (sc.gt_points, sc.pred_points, sc.points_matched) == (2, 2, 1)
        assert sc.point_precision == 0.5 and sc.point_recall == 0.5
        assert sc.outcome_total == 1  # only matched points are judged

    def test_server_from_first_shot_or_field(self):
        gt = _match([_point(1, 0, 100, server="far_player", winner="far_player")])
        pred_ok = _match([_point(1, 0, 100, server="far_player")])
        pred_bad = _match([_point(1, 0, 100, server="near_player")])
        assert score_run("c", gt, pred_ok, [], [], [], FPS).server_correct == 1
        assert score_run("c", gt, pred_bad, [], [], [], FPS).server_correct == 0


class TestScorecardIO:
    def test_summary_mentions_the_key_numbers(self):
        sc = Scorecard(clip="clipA", score=0.5, points_matched=2, gt_points=3, pred_points=4, shot_f1=0.75)
        s = sc.summary()
        assert "score=0.500" in s
        assert "points 2/3 (pred 4)" in s
        assert "F1=0.75" in s

    def test_round_trip_through_json(self, tmp_path: Path):
        sc = Scorecard(clip="clipA", score=0.42, stroke_confusion={"serve": {"serve": 2}}, notes=["n"])
        path = tmp_path / "scorecard.json"
        path.write_text(json.dumps(sc.to_dict()))
        assert load_scorecard(path) == sc
