"""Tests for scene filtering — gameplay detection from frames."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from court_vision.scene_filter import (
    SceneCategory,
    SceneFilterResult,
    GameplaySegment,
    classify_frame,
    classify_frames,
    filter_gameplay_segments,
    load_scene_model,
)


class TestSceneCategory:
    def test_gameplay_is_a_category(self):
        assert SceneCategory.GAMEPLAY.value == "gameplay"

    def test_all_categories_exist(self):
        categories = {c.value for c in SceneCategory}
        assert categories == {"gameplay", "close_up", "replay", "crowd", "transition"}


class TestClassifyFrame:
    def test_returns_scene_filter_result(self):
        """classify_frame returns category + confidence with correct frame index."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        model = MagicMock()
        # Mock model to return logits for 5 classes
        model.return_value = torch.tensor([[2.0, 0.1, 0.1, 0.1, 0.1]])

        result = classify_frame(frame, model, device=torch.device("cpu"), frame_index=42)

        assert isinstance(result, SceneFilterResult)
        assert isinstance(result.category, SceneCategory)
        assert 0.0 <= result.confidence <= 1.0
        assert result.frame_index == 42


class TestClassifyFrames:
    def test_classifies_all_frames_in_directory(self, tmp_path: Path):
        """classify_frames reads frame images and returns results for each."""
        import cv2

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        # Create 3 synthetic frame images
        for i in range(3):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), frame)

        model = MagicMock()
        model.return_value = torch.tensor([[2.0, 0.1, 0.1, 0.1, 0.1]])

        results = classify_frames(frames_dir, total_frames=3, model=model, device=torch.device("cpu"))

        assert len(results) == 3
        assert results[0].frame_index == 0
        assert results[1].frame_index == 1
        assert results[2].frame_index == 2


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


class TestLoadSceneModel:
    @patch("court_vision.scene_filter.models.resnet18")
    def test_loads_resnet18_with_modified_fc(self, mock_resnet18: MagicMock):
        """Loads ResNet-18 and replaces final FC layer for 5 classes."""
        import torch.nn as nn

        mock_model = MagicMock()
        mock_model.fc = MagicMock(in_features=512)
        mock_resnet18.return_value = mock_model

        model = load_scene_model(weights_path=None, device=torch.device("cpu"))

        mock_resnet18.assert_called_once()
        # Verify fc layer was replaced with nn.Linear for 5 classes
        assert isinstance(mock_model.fc, nn.Linear)
        assert mock_model.fc.out_features == 5
