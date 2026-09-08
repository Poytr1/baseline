"""Tests for scoreboard parsing and point-outcome inference (no OCR engines are invoked)."""

import numpy as np
import pytest

from court_vision.scoreboard import (
    ScoreRow,
    ScoreSample,
    ScoreTimeline,
    TextBox,
    _cluster_rows,
    _normalize_points,
    _parse_numbers,
    _parse_tokens,
    _row_tokens,
    compare_states,
    infer_point_winner_row,
    locate_scoreboard,
)


def _row(name: str, games, points: str, serving: bool = False) -> ScoreRow:
    return ScoreRow(name=name, games=list(games), points=points, serving=serving)


def _sample(frame: int, a: ScoreRow, b: ScoreRow) -> ScoreSample:
    return ScoreSample(frame=frame, rows=[a, b])


class TestParseTokens:
    @pytest.mark.parametrize("toks,expected", [
        (["6", "4", "30"], ([6, 4], "30")),
        (["6", "4"], ([6, 4], "")),
        (["AD"], ([], "AD")),
        (["40"], ([], "40")),
        (["3", "2", "8"], ([3, 2], "8")),  # tie-break points
        (["6", "7"], ([6, 7], "")),  # 7 is a game count, not a point score
        (["1", "0"], ([1], "0")),
        ([], ([], "")),
    ])
    def test_games_and_points(self, toks, expected):
        assert _parse_tokens(toks) == expected

    def test_parse_numbers_from_text_maps_letter_o_to_zero(self):
        assert _parse_numbers("6 4 3O") == ([6, 4], "30")
        assert _parse_numbers("2 AD") == ([2], "AD")
        assert _parse_numbers("") == ([], "")


class TestScoreRow:
    @pytest.mark.parametrize("points,rank", [("0", 0), ("15", 1), ("30", 2), ("40", 3), ("AD", 4), ("A", 4), ("ADV", 4)])
    def test_point_rank(self, points: str, rank: int):
        assert _row("a", [], points).point_rank() == rank

    def test_tie_break_rank_is_above_game_points(self):
        assert _row("a", [], "5").point_rank() == 15
        assert _row("a", [], "5").point_rank() > _row("a", [], "AD").point_rank()

    def test_unknown_points_have_no_rank(self):
        assert _row("a", [], "").point_rank() is None
        assert _row("a", [], "?").point_rank() is None


class TestScoreSample:
    def test_valid_needs_two_rows_with_something_readable(self):
        assert _sample(0, _row("a", [3], "15"), _row("b", [2], "0")).is_valid()
        assert _sample(0, _row("a", [], "15"), _row("b", [2], "")).is_valid()
        assert not _sample(0, _row("a", [3], ""), _row("b", [], "")).is_valid()
        assert not ScoreSample(0, [_row("a", [3], "15")]).is_valid()
        assert not ScoreSample(0, []).is_valid()

    def test_has_points(self):
        assert _sample(0, _row("a", [3], "15"), _row("b", [2], "0")).has_points()
        assert _sample(0, _row("a", [3], "5"), _row("b", [2], "3")).has_points()
        assert not _sample(0, _row("a", [3], ""), _row("b", [2], "0")).has_points()


class TestClusterRows:
    def test_groups_by_vertical_overlap_and_sorts(self):
        boxes = [
            TextBox(60, 30, 100, 40, "6", 0.9),
            TextBox(10, 10, 50, 20, "DJOKOVIC", 0.9),
            TextBox(60, 10, 70, 20, "4", 0.9),
            TextBox(10, 31, 50, 41, "SONEGO", 0.9),
        ]
        rows = _cluster_rows(boxes)
        assert [[b.text for b in r] for r in rows] == [["DJOKOVIC", "4"], ["SONEGO", "6"]]

    def test_empty(self):
        assert _cluster_rows([]) == []

    def test_textbox_geometry(self):
        b = TextBox(10, 20, 30, 60, "x", 0.5)
        assert (b.cx, b.cy, b.h) == (20, 40, 40)


class TestRowTokens:
    def test_name_and_numbers(self):
        row = [TextBox(0, 0, 1, 1, "Sinner", 0.9), TextBox(0, 0, 1, 1, "6", 0.9), TextBox(0, 0, 1, 1, "3O", 0.9)]
        assert _row_tokens(row) == ("Sinner", ["6", "30"])

    def test_ad_token(self):
        row = [TextBox(0, 0, 1, 1, "Alcaraz", 0.9), TextBox(0, 0, 1, 1, "AD", 0.9)]
        assert _row_tokens(row) == ("Alcaraz", ["AD"])

    def test_no_name(self):
        assert _row_tokens([TextBox(0, 0, 1, 1, "40", 0.9)]) == ("", ["40"])


class TestNormalizePoints:
    def test_blank_opposite_ad_is_forty(self):
        rows = _normalize_points([_row("a", [3], ""), _row("b", [2], "AD")])
        assert [r.points for r in rows] == ["40", "AD"]

    def test_blank_opposite_points_is_zero(self):
        rows = _normalize_points([_row("a", [3], ""), _row("b", [2], "30")])
        assert [r.points for r in rows] == ["0", "30"]
        rows = _normalize_points([_row("a", [3], "15"), _row("b", [2], "")])
        assert [r.points for r in rows] == ["15", "0"]

    def test_both_blank_stay_blank(self):
        rows = _normalize_points([_row("a", [3], ""), _row("b", [2], "")])
        assert [r.points for r in rows] == ["", ""]

    def test_single_row_untouched(self):
        rows = _normalize_points([_row("a", [3], "")])
        assert rows[0].points == ""


class TestCompareStates:
    def test_point_advance_row0(self):
        before = _sample(0, _row("a", [3], "15"), _row("b", [2], "30"))
        after = _sample(1, _row("a", [3], "30"), _row("b", [2], "30"))
        assert compare_states(before, after) == 0

    def test_point_advance_row1(self):
        before = _sample(0, _row("a", [3], "15"), _row("b", [2], "30"))
        after = _sample(1, _row("a", [3], "15"), _row("b", [2], "40"))
        assert compare_states(before, after) == 1

    def test_game_won(self):
        before = _sample(0, _row("a", [3], "40"), _row("b", [2], "30"))
        after = _sample(1, _row("a", [4], "0"), _row("b", [2], "0"))
        assert compare_states(before, after) == 0

    def test_game_won_in_second_set(self):
        before = _sample(0, _row("a", [6, 3], "40"), _row("b", [4, 2], "30"))
        after = _sample(1, _row("a", [6, 4], "0"), _row("b", [4, 2], "0"))
        assert compare_states(before, after) == 0

    def test_advantage_lost_means_other_row_won(self):
        before = _sample(0, _row("a", [3], "AD"), _row("b", [2], "40"))
        after = _sample(1, _row("a", [3], "40"), _row("b", [2], "40"))
        assert compare_states(before, after) == 1
        before = _sample(0, _row("a", [3], "40"), _row("b", [2], "AD"))
        assert compare_states(before, after) == 0

    def test_no_change_is_none(self):
        s = _sample(0, _row("a", [3], "15"), _row("b", [2], "30"))
        assert compare_states(s, _sample(1, _row("a", [3], "15"), _row("b", [2], "30"))) is None

    def test_both_rows_advancing_is_ambiguous(self):
        before = _sample(0, _row("a", [3], "15"), _row("b", [2], "30"))
        after = _sample(1, _row("a", [3], "30"), _row("b", [2], "40"))
        assert compare_states(before, after) is None

    def test_invalid_samples(self):
        after = _sample(1, _row("a", [3], "30"), _row("b", [2], "30"))
        assert compare_states(ScoreSample(0, [_row("a", [3], "15")]), after) is None

    def test_unreadable_points_fall_back_to_game_counts(self):
        before = _sample(0, _row("a", [3], ""), _row("b", [2], ""))
        assert compare_states(before, _sample(1, _row("a", [3], ""), _row("b", [3], ""))) == 1
        assert compare_states(before, _sample(1, _row("a", [4], ""), _row("b", [2], ""))) == 0
        assert compare_states(before, _sample(1, _row("a", [4], ""), _row("b", [3], ""))) is None

    def test_unreadable_points_without_games(self):
        before = _sample(0, _row("a", [], "15"), _row("b", [2], ""))
        after = _sample(1, _row("a", [], "30"), _row("b", [2], ""))
        assert compare_states(before, after) is None


class TestScoreTimeline:
    def _timeline(self) -> ScoreTimeline:
        return ScoreTimeline(samples=[
            _sample(0, _row("a", [3], "15"), _row("b", [2], "30")),
            _sample(15, _row("a", [3], "15"), _row("b", [2], "30")),
            ScoreSample(30, []),  # OCR miss
            _sample(45, _row("a", [3], "15"), _row("b", [2], "0")),  # misread
            _sample(200, _row("a", [3], "30"), _row("b", [2], "30")),
        ])

    def test_sample_at_skips_invalid_samples(self):
        tl = self._timeline()
        assert tl.sample_at(30).frame == 15
        assert tl.sample_at(30, "after").frame == 45
        assert tl.sample_at(-1) is None
        assert tl.sample_at(999, "after") is None

    def test_stable_state_is_the_majority_vote(self):
        tl = self._timeline()
        state = tl.stable_state(0, 60)
        assert state is not None
        assert [r.points for r in state.rows] == ["15", "30"]
        assert tl.stable_state(100, 150) is None

    def test_empty_timeline(self):
        tl = ScoreTimeline()
        assert tl.samples == [] and tl.roi is None
        assert tl.sample_at(0) is None
        assert tl.stable_state(0, 10) is None


class TestInferPointWinnerRow:
    def test_winner_from_score_change_after_the_point(self):
        tl = ScoreTimeline(samples=[
            _sample(0, _row("a", [3], "15"), _row("b", [2], "30")),
            _sample(15, _row("a", [3], "15"), _row("b", [2], "30")),
            _sample(200, _row("a", [3], "30"), _row("b", [2], "30")),
        ])
        assert infer_point_winner_row(tl, 0, 100, None, fps=30.0) == 0

    def test_after_window_bounded_by_next_point(self):
        tl = ScoreTimeline(samples=[
            _sample(0, _row("a", [3], "15"), _row("b", [2], "30")),
            _sample(400, _row("a", [3], "30"), _row("b", [2], "30")),
        ])
        assert infer_point_winner_row(tl, 0, 100, 150, fps=30.0) is None
        assert infer_point_winner_row(tl, 0, 100, None, fps=30.0) == 0

    def test_no_samples(self):
        assert infer_point_winner_row(ScoreTimeline(), 0, 100, None, fps=30.0) is None


class TestLocateScoreboard:
    def test_needs_at_least_three_probe_frames(self):
        frames = [np.zeros((100, 200, 3), dtype=np.uint8)] * 2
        assert locate_scoreboard(frames) is None
