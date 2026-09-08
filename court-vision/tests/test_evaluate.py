"""Tests for pipeline evaluation module."""

from court_vision.evaluate import match_shots
from court_vision.shot_classify import MatchData, Point, Shot


def _make_match(shots_frames: list[tuple[int, str, str]]) -> MatchData:
    """Helper: create a MatchData with one point containing shots at given frames."""
    shots = [
        Shot(
            shot_number=i + 1,
            frame=frame,
            time_s=frame / 30.0,
            player=player,
            stroke=stroke,
            placement=None,
            confidence=1.0,
        )
        for i, (frame, player, stroke) in enumerate(shots_frames)
    ]
    return MatchData(
        match_id="test",
        source_url="test.mp4",
        metadata={},
        points=[
            Point(
                point_number=1,
                start_frame=0,
                end_frame=300,
                start_time_s=0.0,
                end_time_s=10.0,
                server=None,
                shots=shots,
                outcome=None,
                outcome_player=None,
                rally_length=len(shots),
            )
        ],
    )


class TestMatchShots:
    """Tests for match_shots evaluation function."""

    def test_perfect_match(self):
        """All predictions match ground truth exactly."""
        gt = _make_match([(10, "near_player", "forehand"), (50, "far_player", "backhand")])
        pred = _make_match([(10, "near_player", "forehand"), (50, "far_player", "backhand")])
        result = match_shots(gt, pred)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.stroke_accuracy == 1.0
        assert result.player_accuracy == 1.0

    def test_false_positives(self):
        """Extra predictions lower precision but not recall."""
        gt = _make_match([(10, "near_player", "forehand")])
        pred = _make_match([
            (10, "near_player", "forehand"),
            (30, "far_player", "backhand"),
            (60, "near_player", "slice"),
        ])
        result = match_shots(gt, pred)
        assert result.true_positives == 1
        assert result.false_positives == 2
        assert result.false_negatives == 0
        assert result.recall == 1.0
        assert result.precision < 1.0

    def test_false_negatives(self):
        """Missing predictions lower recall."""
        gt = _make_match([(10, "near_player", "forehand"), (50, "far_player", "backhand")])
        pred = _make_match([(10, "near_player", "forehand")])
        result = match_shots(gt, pred)
        assert result.true_positives == 1
        assert result.false_negatives == 1
        assert result.recall < 1.0

    def test_frame_tolerance(self):
        """Shots within tolerance match; those beyond do not."""
        gt = _make_match([(10, "near_player", "forehand")])
        pred = _make_match([(22, "near_player", "forehand")])
        within = match_shots(gt, pred, frame_tolerance=15)
        assert within.true_positives == 1
        beyond = match_shots(gt, pred, frame_tolerance=10)
        assert beyond.true_positives == 0

    def test_stroke_accuracy(self):
        """Stroke accuracy counts only correctly classified strokes."""
        gt = _make_match([(10, "near_player", "forehand"), (50, "far_player", "backhand")])
        pred = _make_match([(10, "near_player", "slice"), (50, "far_player", "backhand")])
        result = match_shots(gt, pred)
        assert result.stroke_correct == 1
        assert result.stroke_total == 2
        assert result.stroke_accuracy == 0.5

    def test_empty_predictions(self):
        """No predictions yield zero precision, recall, and F1."""
        gt = _make_match([(10, "near_player", "forehand")])
        pred = _make_match([])
        result = match_shots(gt, pred)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_no_ground_truth(self):
        """No ground truth with predictions yields all false positives."""
        gt = _make_match([])
        pred = _make_match([(10, "near_player", "forehand")])
        result = match_shots(gt, pred)
        assert result.false_positives == 1
        assert result.recall == 0.0
