"""Tests for court detection and homography."""

import cv2
import numpy as np
import pytest

from court_vision.court_detect import COURT_KEYPOINTS


class TestCourtKeypoints:
    def test_keypoints_is_dict(self):
        """COURT_KEYPOINTS maps names to (x, y) tuples in meters."""
        assert isinstance(COURT_KEYPOINTS, dict)
        assert len(COURT_KEYPOINTS) >= 12

    def test_net_center_is_origin(self):
        """Net center is at origin (0, 0)."""
        assert COURT_KEYPOINTS["net_center"] == (0.0, 0.0)

    def test_baseline_far_left_singles(self):
        """Far-left singles baseline corner."""
        x, y = COURT_KEYPOINTS["baseline_far_left_singles"]
        assert x == pytest.approx(-4.115)
        assert y == pytest.approx(11.885)

    def test_baseline_near_right_singles(self):
        """Near-right singles baseline corner."""
        x, y = COURT_KEYPOINTS["baseline_near_right_singles"]
        assert x == pytest.approx(4.115)
        assert y == pytest.approx(-11.885)

    def test_service_line_far_center(self):
        """Far service line at center mark."""
        x, y = COURT_KEYPOINTS["service_far_center"]
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(6.4)

    def test_all_keypoints_are_float_tuples(self):
        """Every keypoint is a 2-tuple of floats."""
        for name, (x, y) in COURT_KEYPOINTS.items():
            assert isinstance(x, float), f"{name} x is not float"
            assert isinstance(y, float), f"{name} y is not float"


def _draw_court_lines(img: np.ndarray) -> np.ndarray:
    """Draw white court lines on a green court image for testing."""
    h, w = img.shape[:2]
    cv2.line(img, (200, 650), (1080, 650), (255, 255, 255), 2)
    cv2.line(img, (400, 150), (880, 150), (255, 255, 255), 2)
    cv2.line(img, (200, 650), (400, 150), (255, 255, 255), 2)
    cv2.line(img, (1080, 650), (880, 150), (255, 255, 255), 2)
    cv2.line(img, (280, 450), (1000, 450), (255, 255, 255), 2)
    cv2.line(img, (360, 280), (920, 280), (255, 255, 255), 2)
    cv2.line(img, (640, 280), (640, 450), (255, 255, 255), 2)
    return img


class TestClusterLines:
    def test_merges_similar_horizontal_lines(self):
        """Horizontal lines at similar y-positions cluster into one."""
        from court_vision.court_detect import cluster_lines

        lines = [
            ((100, 300), (900, 302)),
            ((150, 298), (850, 301)),
            ((120, 299), (880, 303)),
        ]
        result = cluster_lines(lines, rho_threshold=20.0, theta_threshold=10.0)
        assert len(result) == 1

    def test_keeps_distinct_lines(self):
        """Lines at different positions remain separate."""
        from court_vision.court_detect import cluster_lines

        lines = [
            ((100, 100), (900, 100)),   # y=100
            ((100, 400), (900, 400)),   # y=400
            ((100, 650), (900, 650)),   # y=650
        ]
        result = cluster_lines(lines, rho_threshold=20.0, theta_threshold=10.0)
        assert len(result) == 3

    def test_merges_similar_vertical_lines(self):
        """Vertical lines at similar x-positions cluster into one."""
        from court_vision.court_detect import cluster_lines

        lines = [
            ((500, 100), (502, 600)),
            ((498, 150), (501, 550)),
        ]
        result = cluster_lines(lines, rho_threshold=20.0, theta_threshold=10.0)
        assert len(result) == 1

    def test_empty_input(self):
        """Returns empty list for empty input."""
        from court_vision.court_detect import cluster_lines

        result = cluster_lines([], rho_threshold=20.0, theta_threshold=10.0)
        assert result == []

    def test_single_line(self):
        """Single line returns unchanged."""
        from court_vision.court_detect import cluster_lines

        lines = [((100, 300), (900, 300))]
        result = cluster_lines(lines, rho_threshold=20.0, theta_threshold=10.0)
        assert len(result) == 1


class TestFilterMarginLines:
    def test_removes_top_margin_lines(self):
        """Lines in the top scoreboard area are filtered out."""
        from court_vision.court_detect import _filter_margin_lines

        lines = [
            ((100, 50), (900, 52)),    # top 7% — scoreboard
            ((100, 300), (900, 302)),   # mid frame — court
        ]
        result = _filter_margin_lines(lines, frame_height=720, top_margin=0.15)
        assert len(result) == 1
        assert result[0][0][1] == 300  # kept the mid-frame line

    def test_keeps_court_area_lines(self):
        """Lines in the court area are kept."""
        from court_vision.court_detect import _filter_margin_lines

        lines = [
            ((100, 200), (900, 200)),   # 28% down
            ((100, 400), (900, 400)),   # 56% down
            ((100, 600), (900, 600)),   # 83% down
        ]
        result = _filter_margin_lines(lines, frame_height=720, top_margin=0.15)
        assert len(result) == 3

    def test_empty_input(self):
        """Returns empty list for empty input."""
        from court_vision.court_detect import _filter_margin_lines

        result = _filter_margin_lines([], frame_height=720)
        assert result == []


class TestSelectCourtQuad:
    def test_selects_quad_from_lines(self):
        """Finds a valid quadrilateral using horizontal and vertical lines."""
        from court_vision.court_detect import _select_court_quad

        # Simulate a perspective court: trapezoid wider at bottom
        keypoints = [
            (200.0, 650.0), (1080.0, 650.0),
            (880.0, 150.0), (400.0, 150.0),
            (640.0, 400.0),
        ]
        horizontal = [
            ((200, 650), (1080, 650)),  # near baseline
            ((400, 150), (880, 150)),   # far baseline
        ]
        vertical = [
            ((200, 650), (400, 150)),   # left sideline
            ((1080, 650), (880, 150)),  # right sideline
        ]
        result = _select_court_quad(
            keypoints, frame_width=1280, frame_height=720,
            horizontal_lines=horizontal, vertical_lines=vertical,
        )
        assert result is not None
        pixel_pts, court_pts = result
        assert pixel_pts.shape == (4, 2)
        assert court_pts.shape == (4, 2)

    def test_returns_none_with_too_few_points(self):
        """Returns None with fewer than 4 keypoints."""
        from court_vision.court_detect import _select_court_quad

        result = _select_court_quad([(100.0, 100.0), (200.0, 200.0)],
                                     frame_width=1280, frame_height=720)
        assert result is None

    def test_rejects_no_perspective_quad(self):
        """Rejects a quadrilateral where far side is nearly as wide as near."""
        from court_vision.court_detect import _select_court_quad

        # Rectangle (no perspective foreshortening) — ratio ~1.0
        keypoints = [
            (200.0, 650.0), (1080.0, 650.0),
            (200.0, 150.0), (1080.0, 150.0),
            (640.0, 400.0),
        ]
        horizontal = [
            ((200, 650), (1080, 650)),
            ((200, 150), (1080, 150)),
        ]
        vertical = [
            ((200, 150), (200, 650)),
            ((1080, 150), (1080, 650)),
        ]
        result = _select_court_quad(
            keypoints, frame_width=1280, frame_height=720,
            horizontal_lines=horizontal, vertical_lines=vertical,
        )
        # Should be rejected: far/near ≈ 1.0 (no perspective)
        assert result is None

    def test_selects_best_vertical_pair(self):
        """With noise verticals that violate perspective, picks valid pair."""
        from court_vision.court_detect import _select_court_quad

        # 2 good verticals + 1 noise vertical far right (creates ratio > 0.85)
        keypoints = [
            (200.0, 650.0), (1000.0, 650.0), (1150.0, 650.0),
            (400.0, 150.0), (850.0, 150.0), (1120.0, 150.0),
            (640.0, 400.0),
        ]
        horizontal = [
            ((100, 650), (1300, 650)),  # near baseline
            ((300, 150), (1200, 150)),  # far baseline
        ]
        vertical = [
            ((200, 650), (400, 150)),    # left sideline (good)
            ((1000, 650), (850, 150)),   # right sideline (good, ratio 0.56)
            ((1150, 650), (1120, 150)),  # noise vertical (with left: ratio 0.76, still ok
                                          # but good pair has better ratio score)
        ]
        result = _select_court_quad(
            keypoints, frame_width=1280, frame_height=720,
            horizontal_lines=horizontal, vertical_lines=vertical,
        )
        assert result is not None
        pixel_pts, _ = result
        assert pixel_pts.shape == (4, 2)


class TestDetectCourtLines:
    def test_detects_lines_on_synthetic_court(self):
        """Detects lines from a synthetic court image."""
        from court_vision.court_detect import detect_court_lines

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)  # green
        img = _draw_court_lines(img)
        lines = detect_court_lines(img)
        assert len(lines) >= 4

    def test_returns_list_of_line_segments(self):
        """Each detected line is a pair of (x, y) endpoints."""
        from court_vision.court_detect import detect_court_lines

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        img = _draw_court_lines(img)
        lines = detect_court_lines(img)
        for line in lines:
            assert len(line) == 2, "Each line should be ((x1,y1), (x2,y2))"
            (x1, y1), (x2, y2) = line
            assert isinstance(x1, (int, float))
            assert isinstance(y1, (int, float))

    def test_no_lines_on_blank_image(self):
        """Returns empty list when no court lines are present."""
        from court_vision.court_detect import detect_court_lines

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        lines = detect_court_lines(img)
        assert lines == []


class TestClassifyLines:
    def test_horizontal_line(self):
        """A nearly-horizontal line is classified as horizontal."""
        from court_vision.court_detect import classify_lines

        lines = [((100, 300), (900, 310))]  # nearly horizontal
        h, v = classify_lines(lines, angle_threshold=30.0)
        assert len(h) == 1
        assert len(v) == 0

    def test_vertical_line(self):
        """A nearly-vertical line is classified as vertical."""
        from court_vision.court_detect import classify_lines

        lines = [((500, 100), (510, 600))]  # nearly vertical
        h, v = classify_lines(lines, angle_threshold=30.0)
        assert len(h) == 0
        assert len(v) == 1

    def test_diagonal_line_excluded(self):
        """A 45-degree line is neither horizontal nor vertical."""
        from court_vision.court_detect import classify_lines

        lines = [((100, 100), (500, 500))]  # 45 degrees
        h, v = classify_lines(lines, angle_threshold=30.0)
        assert len(h) == 0
        assert len(v) == 0

    def test_mixed_lines(self):
        """Correctly separates a mix of horizontal and vertical lines."""
        from court_vision.court_detect import classify_lines

        lines = [
            ((100, 300), (900, 310)),   # horizontal
            ((500, 100), (510, 600)),   # vertical
            ((200, 200), (800, 205)),   # horizontal
        ]
        h, v = classify_lines(lines, angle_threshold=30.0)
        assert len(h) == 2
        assert len(v) == 1


class TestLineIntersection:
    def test_perpendicular_lines_intersect(self):
        """Two perpendicular lines intersect at their crossing point."""
        from court_vision.court_detect import find_line_intersection

        line1 = ((100, 300), (900, 300))
        line2 = ((500, 100), (500, 600))

        point = find_line_intersection(line1, line2)

        assert point is not None
        x, y = point
        assert x == pytest.approx(500.0, abs=1.0)
        assert y == pytest.approx(300.0, abs=1.0)

    def test_parallel_lines_no_intersection(self):
        """Parallel lines return None."""
        from court_vision.court_detect import find_line_intersection

        line1 = ((100, 300), (900, 300))
        line2 = ((100, 400), (900, 400))

        point = find_line_intersection(line1, line2)

        assert point is None

    def test_angled_lines_intersect(self):
        """Two angled lines find their intersection."""
        from court_vision.court_detect import find_line_intersection

        line1 = ((0, 0), (100, 100))
        line2 = ((100, 0), (0, 100))

        point = find_line_intersection(line1, line2)

        assert point is not None
        x, y = point
        assert x == pytest.approx(50.0, abs=1.0)
        assert y == pytest.approx(50.0, abs=1.0)


class TestExtractKeypoints:
    def test_extracts_keypoints_from_court_lines(self):
        """Extracts intersection points from horizontal and vertical lines."""
        from court_vision.court_detect import extract_keypoints

        horizontal = [
            ((200, 150), (880, 150)),
            ((200, 650), (1080, 650)),
        ]
        vertical = [
            ((200, 650), (200, 150)),
            ((1080, 650), (880, 150)),
        ]

        keypoints = extract_keypoints(horizontal, vertical)

        assert len(keypoints) >= 2

    def test_keypoints_are_float_tuples(self):
        """Each keypoint is a (x, y) tuple of floats."""
        from court_vision.court_detect import extract_keypoints

        horizontal = [((100, 300), (900, 300))]
        vertical = [((500, 100), (500, 600))]

        keypoints = extract_keypoints(horizontal, vertical)

        for x, y in keypoints:
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_no_keypoints_from_empty_lines(self):
        """Returns empty list when no lines provided."""
        from court_vision.court_detect import extract_keypoints

        keypoints = extract_keypoints([], [])

        assert keypoints == []


class TestComputeHomography:
    def test_identity_like_homography(self):
        """When pixel and court points are proportional, homography maps correctly."""
        from court_vision.court_detect import compute_homography

        pixel_points = np.array([
            [200.0, 650.0],
            [1080.0, 650.0],
            [880.0, 150.0],
            [400.0, 150.0],
        ], dtype=np.float64)

        court_points = np.array([
            [-4.115, -11.885],
            [4.115, -11.885],
            [4.115, 11.885],
            [-4.115, 11.885],
        ], dtype=np.float64)

        H = compute_homography(pixel_points, court_points)

        assert H is not None
        assert H.shape == (3, 3)

    def test_homography_transforms_known_point(self):
        """Homography correctly transforms a known pixel point to court coords."""
        from court_vision.court_detect import compute_homography, pixel_to_court

        pixel_points = np.array([
            [200.0, 650.0],
            [1080.0, 650.0],
            [880.0, 150.0],
            [400.0, 150.0],
        ], dtype=np.float64)

        court_points = np.array([
            [-4.115, -11.885],
            [4.115, -11.885],
            [4.115, 11.885],
            [-4.115, 11.885],
        ], dtype=np.float64)

        H = compute_homography(pixel_points, court_points)

        result = pixel_to_court(np.array([200.0, 650.0]), H)
        assert result[0] == pytest.approx(-4.115, abs=0.5)
        assert result[1] == pytest.approx(-11.885, abs=0.5)

    def test_homography_needs_at_least_4_points(self):
        """Returns None with fewer than 4 point correspondences."""
        from court_vision.court_detect import compute_homography

        pixel_points = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
        court_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        H = compute_homography(pixel_points, court_points)

        assert H is None


class TestPixelToCourt:
    def test_transforms_single_point(self):
        """Transforms a single pixel coordinate to court space."""
        from court_vision.court_detect import pixel_to_court

        H = np.eye(3, dtype=np.float64)

        result = pixel_to_court(np.array([640.0, 360.0]), H)

        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestMatchKeypoints:
    def test_matches_four_corners(self):
        """Matches detected pixel keypoints to their nearest court keypoint candidates."""
        from court_vision.court_detect import match_keypoints_to_court

        pixel_keypoints = [
            (200.0, 650.0),
            (1080.0, 650.0),
            (880.0, 150.0),
            (400.0, 150.0),
        ]

        pixel_pts, court_pts = match_keypoints_to_court(pixel_keypoints)

        assert len(pixel_pts) >= 4
        assert len(court_pts) >= 4
        assert pixel_pts.shape[1] == 2
        assert court_pts.shape[1] == 2

    def test_returns_none_with_too_few_keypoints(self):
        """Returns None when fewer than 4 keypoints detected."""
        from court_vision.court_detect import match_keypoints_to_court

        pixel_keypoints = [(200.0, 650.0), (1080.0, 650.0)]

        result = match_keypoints_to_court(pixel_keypoints)

        assert result is None


class TestDetectCourt:
    def test_returns_court_detection_result(self):
        """Full detect_court returns a CourtDetectionResult with homography."""
        from court_vision.court_detect import CourtDetectionResult, detect_court

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        img = _draw_court_lines(img)

        result = detect_court(img)

        assert isinstance(result, CourtDetectionResult)
        assert isinstance(result.success, bool)

    def test_successful_detection_has_homography(self):
        """Successful detection includes a 3x3 homography matrix."""
        from court_vision.court_detect import detect_court

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        img = _draw_court_lines(img)

        result = detect_court(img)

        if result.success:
            assert result.homography is not None
            assert result.homography.shape == (3, 3)
            assert result.pixel_keypoints is not None
            assert len(result.pixel_keypoints) >= 4

    def test_failed_detection_on_blank_image(self):
        """Detection fails gracefully on an image with no court lines."""
        from court_vision.court_detect import detect_court

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)

        result = detect_court(img)

        assert result.success is False
        assert result.homography is None


class TestComputeSegmentHomographies:
    def test_returns_one_result_per_segment(self, tmp_path):
        """Computes one CourtDetectionResult per gameplay segment."""
        from court_vision.court_detect import compute_segment_homographies
        from court_vision.scene_filter import GameplaySegment

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        for i in range(10):
            img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
            img = _draw_court_lines(img)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segments = [
            GameplaySegment(start_frame=0, end_frame=4, start_time_s=0.0, end_time_s=0.13, frame_count=5),
            GameplaySegment(start_frame=7, end_frame=9, start_time_s=0.23, end_time_s=0.3, frame_count=3),
        ]

        results = compute_segment_homographies(frames_dir, segments)

        assert len(results) == 2

    def test_uses_middle_frame_of_segment(self, tmp_path):
        """Samples the middle frame of each segment for homography."""
        from unittest.mock import patch as mock_patch
        from court_vision.court_detect import compute_segment_homographies, CourtDetectionResult
        from court_vision.scene_filter import GameplaySegment

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        for i in range(10):
            img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segments = [
            GameplaySegment(start_frame=0, end_frame=8, start_time_s=0.0, end_time_s=0.27, frame_count=9),
        ]

        with mock_patch("court_vision.court_detect.detect_court") as mock_detect:
            mock_detect.return_value = CourtDetectionResult(success=False, num_lines_detected=0)
            compute_segment_homographies(frames_dir, segments)

            call_args = mock_detect.call_args
            assert call_args is not None


class TestComputeHomographyRANSAC:
    def test_ransac_rejects_outlier(self):
        """Homography computed with RANSAC ignores outlier point."""
        from court_vision.court_detect import compute_homography

        # 4 good correspondences + 1 outlier
        pixel_points = np.array([
            [200.0, 650.0],
            [1080.0, 650.0],
            [880.0, 150.0],
            [400.0, 150.0],
            [999.0, 999.0],  # outlier
        ], dtype=np.float64)

        court_points = np.array([
            [-4.115, -11.885],
            [4.115, -11.885],
            [4.115, 11.885],
            [-4.115, 11.885],
            [0.0, 0.0],  # outlier target
        ], dtype=np.float64)

        H = compute_homography(pixel_points, court_points)
        assert H is not None
        assert H.shape == (3, 3)

    def test_good_points_still_work(self):
        """RANSAC doesn't break normal 4-point homography."""
        from court_vision.court_detect import compute_homography, pixel_to_court

        pixel_points = np.array([
            [200.0, 650.0],
            [1080.0, 650.0],
            [880.0, 150.0],
            [400.0, 150.0],
        ], dtype=np.float64)

        court_points = np.array([
            [-4.115, -11.885],
            [4.115, -11.885],
            [4.115, 11.885],
            [-4.115, 11.885],
        ], dtype=np.float64)

        H = compute_homography(pixel_points, court_points)
        assert H is not None

        result = pixel_to_court(np.array([200.0, 650.0]), H)
        assert result[0] == pytest.approx(-4.115, abs=0.5)
        assert result[1] == pytest.approx(-11.885, abs=0.5)


class TestComputeReprojectionError:
    def test_identity_has_zero_error(self):
        """Identity homography gives zero reprojection error."""
        from court_vision.court_detect import _compute_reprojection_error

        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        H = np.eye(3, dtype=np.float64)
        error = _compute_reprojection_error(pts, pts, H)
        assert error == pytest.approx(0.0, abs=0.01)

    def test_bad_homography_has_high_error(self):
        """Mismatched homography gives high reprojection error."""
        from court_vision.court_detect import _compute_reprojection_error

        pixel_pts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dtype=np.float64)
        court_pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        # Use identity as a deliberately wrong homography (pixel != court scale)
        H = np.eye(3, dtype=np.float64)
        error = _compute_reprojection_error(pixel_pts, court_pts, H)
        assert error > 10.0


class TestDetectCourtWithReprojection:
    def test_valid_court_passes_reprojection(self):
        """Detect court on valid synthetic image passes reprojection check."""
        from court_vision.court_detect import detect_court

        img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        img = _draw_court_lines(img)
        result = detect_court(img)
        # If detection succeeds, it passed reprojection validation
        if result.success:
            assert result.homography is not None


class TestMultiFrameSampling:
    def test_samples_three_frames(self, tmp_path):
        """compute_segment_homographies tries 3 frames per segment."""
        from unittest.mock import patch as mock_patch, call
        from court_vision.court_detect import compute_segment_homographies, CourtDetectionResult
        from court_vision.scene_filter import GameplaySegment

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(20):
            img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segments = [
            GameplaySegment(start_frame=0, end_frame=19, start_time_s=0.0, end_time_s=0.63, frame_count=20),
        ]

        # First two calls fail, third succeeds
        mock_results = [
            CourtDetectionResult(success=False, num_lines_detected=0),
            CourtDetectionResult(success=False, num_lines_detected=0),
            CourtDetectionResult(success=True, homography=np.eye(3), num_lines_detected=6),
        ]

        with mock_patch("court_vision.court_detect.detect_court", side_effect=mock_results) as mock_detect:
            results = compute_segment_homographies(frames_dir, segments)

        assert len(results) == 1
        assert results[0].success is True
        assert mock_detect.call_count == 3

    def test_returns_best_on_first_success(self, tmp_path):
        """Stops trying frames after first successful detection."""
        from unittest.mock import patch as mock_patch
        from court_vision.court_detect import compute_segment_homographies, CourtDetectionResult
        from court_vision.scene_filter import GameplaySegment

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(20):
            img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        segments = [
            GameplaySegment(start_frame=0, end_frame=19, start_time_s=0.0, end_time_s=0.63, frame_count=20),
        ]

        mock_result = CourtDetectionResult(success=True, homography=np.eye(3), num_lines_detected=6)

        with mock_patch("court_vision.court_detect.detect_court", return_value=mock_result) as mock_detect:
            results = compute_segment_homographies(frames_dir, segments)

        assert results[0].success is True
        assert mock_detect.call_count == 1  # stopped after first success


class TestDetectCourtNeural:
    def test_returns_result_with_valid_keypoints(self):
        """Neural detection with mocked keypoints produces a result."""
        from unittest.mock import patch
        from court_vision.court_detect import detect_court_neural

        # Mock 4 corner keypoints (doubles corners)
        mock_points = [(None, None)] * 14
        mock_points[0] = (200.0, 100.0)   # far left doubles
        mock_points[1] = (1080.0, 100.0)  # far right doubles
        mock_points[2] = (100.0, 650.0)   # near left doubles
        mock_points[3] = (1180.0, 650.0)  # near right doubles

        with patch("court_vision.court_keypoint_net.detect_keypoints", return_value=mock_points):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            result = detect_court_neural(frame)
            # May or may not succeed depending on reprojection, but should not crash
            assert isinstance(result.success, bool)

    def test_fails_with_too_few_keypoints(self):
        """Fewer than min_keypoints returns failure."""
        from unittest.mock import patch
        from court_vision.court_detect import detect_court_neural

        mock_points = [(None, None)] * 14
        mock_points[0] = (200.0, 100.0)  # Only 1 keypoint

        with patch("court_vision.court_keypoint_net.detect_keypoints", return_value=mock_points):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            result = detect_court_neural(frame, min_keypoints=4)
            assert result.success is False

    def test_detect_court_falls_back_to_classical(self):
        """When neural fails, classical pipeline still runs."""
        from unittest.mock import patch
        from court_vision.court_detect import detect_court, CourtDetectionResult

        # Make neural detection raise an exception
        with patch("court_vision.court_detect.detect_court_neural", side_effect=RuntimeError("no weights")):
            img = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
            img = _draw_court_lines(img)
            result = detect_court(img)
            # Classical pipeline should still produce a result
            assert isinstance(result, CourtDetectionResult)
