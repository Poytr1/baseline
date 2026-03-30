"""Tests for heuristic scene filter."""

import cv2
import numpy as np
import pytest

from court_vision.heuristic_scene_filter import (
    HeuristicScores,
    HeuristicWeights,
    classify_frame_heuristic,
    classify_frames_heuristic,
    compute_court_color_ratio,
    compute_court_spatial_score,
    compute_gameplay_score,
    compute_line_score,
)
from court_vision.scene_filter import SceneCategory, SceneFilterResult


class TestComputeCourtColorRatio:
    def test_green_court_has_high_ratio(self):
        """Frame filled with green court color has high ratio."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)  # BGR forest green
        ratio = compute_court_color_ratio(frame)
        assert ratio > 0.8

    def test_black_image_has_zero_ratio(self):
        """All-black frame has zero court color."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        ratio = compute_court_color_ratio(frame)
        assert ratio == 0.0

    def test_blue_court_has_high_ratio(self):
        """Frame filled with blue hard court color has high ratio."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (180, 120, 40)  # BGR blue court
        ratio = compute_court_color_ratio(frame)
        assert ratio > 0.8

    def test_half_green_has_partial_ratio(self):
        """Half green, half black gives ~0.5 ratio."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:360, :] = (34, 139, 34)  # top half green
        ratio = compute_court_color_ratio(frame)
        assert 0.3 < ratio < 0.7


class TestComputeLineScore:
    def test_frame_with_white_lines_scores_high(self):
        """Frame with white lines on dark background scores > 0."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (34, 100, 34)  # dark green background
        # Draw horizontal white lines
        cv2.line(frame, (100, 200), (1100, 200), (255, 255, 255), 3)
        cv2.line(frame, (100, 500), (1100, 500), (255, 255, 255), 3)
        # Draw vertical white lines
        cv2.line(frame, (200, 100), (200, 600), (255, 255, 255), 3)
        cv2.line(frame, (1000, 100), (1000, 600), (255, 255, 255), 3)
        score = compute_line_score(frame)
        assert score > 0.3

    def test_blank_frame_has_zero_score(self):
        """All-black frame has no lines."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        score = compute_line_score(frame)
        assert score == 0.0

    def test_grid_pattern_gets_bonus(self):
        """Lines in both H and V directions get grid bonus."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(frame, (100, 300), (1100, 300), (255, 255, 255), 3)
        cv2.line(frame, (640, 50), (640, 670), (255, 255, 255), 3)
        score = compute_line_score(frame)
        # Should have grid bonus since lines in both directions
        assert score > 0.0


class TestComputeCourtSpatialScore:
    def test_court_in_lower_region_scores_high(self):
        """Green in bottom 2/3, dark on top scores high."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[240:, :] = (34, 139, 34)  # bottom 2/3 green
        score = compute_court_spatial_score(frame)
        assert score >= 0.5

    def test_uniform_green_scores_lower(self):
        """Court color everywhere gives lower spatial signal than bottom-only."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)
        score_uniform = compute_court_spatial_score(frame)

        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame2[240:, :] = (34, 139, 34)
        score_bottom = compute_court_spatial_score(frame2)

        assert score_bottom >= score_uniform


class TestComputeGameplayScore:
    def test_synthetic_court_gets_high_composite(self):
        """Green frame with white lines gets high composite score."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[240:, :] = (34, 139, 34)
        cv2.line(frame, (100, 300), (1100, 300), (255, 255, 255), 3)
        cv2.line(frame, (100, 600), (1100, 600), (255, 255, 255), 3)
        cv2.line(frame, (200, 250), (200, 650), (255, 255, 255), 3)
        cv2.line(frame, (1000, 250), (1000, 650), (255, 255, 255), 3)
        scores = compute_gameplay_score(frame)
        assert isinstance(scores, HeuristicScores)
        assert scores.composite > 0.4

    def test_black_frame_gets_low_composite(self):
        """All-black frame gets near-zero composite."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        scores = compute_gameplay_score(frame)
        assert scores.composite < 0.1

    def test_custom_weights_applied(self):
        """Custom weights change the composite score."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)
        default_scores = compute_gameplay_score(frame)
        custom_weights = HeuristicWeights(court_color=1.0, line_detection=0.0, spatial_distribution=0.0)
        custom_scores = compute_gameplay_score(frame, weights=custom_weights)
        assert custom_scores.composite != default_scores.composite


class TestClassifyFrameHeuristic:
    def test_returns_scene_filter_result(self):
        """Returns a SceneFilterResult."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = classify_frame_heuristic(frame)
        assert isinstance(result, SceneFilterResult)

    def test_gameplay_frame_classified_correctly(self):
        """Synthetic court frame classified as GAMEPLAY."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[240:, :] = (34, 139, 34)
        cv2.line(frame, (100, 300), (1100, 300), (255, 255, 255), 3)
        cv2.line(frame, (100, 600), (1100, 600), (255, 255, 255), 3)
        cv2.line(frame, (200, 250), (200, 650), (255, 255, 255), 3)
        cv2.line(frame, (1000, 250), (1000, 650), (255, 255, 255), 3)
        result = classify_frame_heuristic(frame)
        assert result.category == SceneCategory.GAMEPLAY

    def test_non_gameplay_classified_as_transition(self):
        """Black frame classified as non-gameplay."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = classify_frame_heuristic(frame)
        assert result.category != SceneCategory.GAMEPLAY

    def test_frame_index_preserved(self):
        """frame_index round-trips."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = classify_frame_heuristic(frame, frame_index=42)
        assert result.frame_index == 42


class TestClassifyFramesHeuristic:
    def test_classifies_all_frames(self, tmp_path):
        """Classifies all frames in a directory."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(3):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), frame)
        results = classify_frames_heuristic(frames_dir, total_frames=3)
        assert len(results) == 3
        assert all(isinstance(r, SceneFilterResult) for r in results)
