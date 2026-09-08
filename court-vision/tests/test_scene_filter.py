"""Tests for scene filtering — gameplay detection from frames."""

import pytest

from court_vision.scene_filter import (
    GameplaySegment,
    SceneCategory,
    SceneFilterResult,
    filter_gameplay_segments,
)


class TestSceneCategory:
    def test_gameplay_is_a_category(self):
        assert SceneCategory.GAMEPLAY.value == "gameplay"

    def test_all_categories_exist(self):
        categories = {c.value for c in SceneCategory}
        assert categories == {"gameplay", "close_up", "replay", "crowd", "transition"}


class TestFilterGameplaySegments:
    def test_contiguous_gameplay_frames(self):
        """Groups contiguous gameplay frames into segments."""
        results = [
            SceneFilterResult(frame_index=0, category=SceneCategory.GAMEPLAY, confidence=0.9),
            SceneFilterResult(frame_index=1, category=SceneCategory.GAMEPLAY, confidence=0.85),
            SceneFilterResult(frame_index=2, category=SceneCategory.GAMEPLAY, confidence=0.88),
            SceneFilterResult(frame_index=3, category=SceneCategory.CLOSE_UP, confidence=0.95),
            SceneFilterResult(frame_index=4, category=SceneCategory.GAMEPLAY, confidence=0.91),
        ]

        segments = filter_gameplay_segments(results, fps=30.0)

        assert len(segments) == 2
        assert segments[0].start_frame == 0
        assert segments[0].end_frame == 2
        assert segments[1].start_frame == 4
        assert segments[1].end_frame == 4

    def test_no_gameplay_returns_empty(self):
        """Returns empty list when no gameplay frames detected."""
        results = [
            SceneFilterResult(frame_index=0, category=SceneCategory.CLOSE_UP, confidence=0.9),
            SceneFilterResult(frame_index=1, category=SceneCategory.CROWD, confidence=0.8),
        ]

        segments = filter_gameplay_segments(results, fps=30.0)

        assert segments == []

    def test_segment_has_time_range(self):
        """Segments include start/end times in seconds."""
        results = [
            SceneFilterResult(frame_index=30, category=SceneCategory.GAMEPLAY, confidence=0.9),
            SceneFilterResult(frame_index=31, category=SceneCategory.GAMEPLAY, confidence=0.85),
        ]

        segments = filter_gameplay_segments(results, fps=30.0)

        assert len(segments) == 1
        assert segments[0].start_time_s == pytest.approx(1.0)
        assert segments[0].end_time_s == pytest.approx(31 / 30.0)
