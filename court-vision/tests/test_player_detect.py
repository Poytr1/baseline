"""Tests for the player detection module (YOLO-pose + court-aware tracking)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from court_vision.ball_tracker import BallDetection, BallTrajectory
from court_vision.player_detect import (
    COCO_KEYPOINT_NAMES,
    FrameTrackingResult,
    PersonCandidate,
    PlayerDetection,
    PlayerTracker,
    PoseKeypoints,
    _iou,
    build_ball_trajectory,
    court_to_pixel,
    detect_persons,
    detect_persons_with_far_crop,
    detect_players_segment,
    far_court_roi,
    finalize_ball_by_frame,
    map_player_to_court,
    pixel_to_court,
)
from court_vision.scene_filter import GameplaySegment

FRAME_SHAPE = (720, 1280, 3)

# Linear pixel->court homography: court_x = (px - 640) / 50, court_y = -(py - 360) / 25
# (bottom of the image == near baseline == negative court y).
H_LINEAR = np.array([[1 / 50, 0.0, -12.8], [0.0, -1 / 25, 14.4], [0.0, 0.0, 1.0]])


def _perspective_homography() -> np.ndarray:
    """Pixel->court homography from a plausible broadcast view of the doubles corners."""
    px = np.array([[200, 650], [1080, 650], [830, 250], [450, 250]], dtype=np.float32)
    court = np.array([[-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885]], dtype=np.float32)
    return cv2.getPerspectiveTransform(px, court).astype(np.float64)


def _cand(bbox, conf: float = 0.9, keypoints=None) -> PersonCandidate:
    return PersonCandidate(bbox=tuple(float(v) for v in bbox), confidence=conf, keypoints=dict(keypoints or {}))


def _segment(start: int, end: int) -> GameplaySegment:
    return GameplaySegment(start_frame=start, end_frame=end, start_time_s=start / 30.0,
                           end_time_s=end / 30.0, frame_count=end - start + 1)


class TestDataclasses:
    def test_player_detection_fields(self):
        det = PlayerDetection(frame_index=0, bbox=(100.0, 200.0, 150.0, 400.0), confidence=0.92)
        assert det.frame_index == 0
        assert det.bbox == (100.0, 200.0, 150.0, 400.0)
        assert det.confidence == 0.92
        assert det.court_position is None
        assert det.role is None

    def test_player_detection_with_court_position(self):
        det = PlayerDetection(frame_index=0, bbox=(100.0, 200.0, 150.0, 400.0), confidence=0.92,
                              court_position=(2.0, -8.0), role="near_player")
        assert det.court_position == (2.0, -8.0)
        assert det.role == "near_player"

    def test_pose_keypoints_fields(self):
        pk = PoseKeypoints(frame_index=0, role="far_player", keypoints={"left_wrist": (100.0, 200.0, 0.95)})
        assert pk.role == "far_player"
        assert pk.keypoints["left_wrist"] == (100.0, 200.0, 0.95)

    def test_frame_tracking_fields(self):
        result = FrameTrackingResult(
            frame_index=0,
            ball=BallDetection(frame_index=0, x=300.0, y=200.0, confidence=0.8),
            players=[], poses=[],
        )
        assert result.ball is not None
        assert result.players == [] and result.poses == []

    def test_person_candidate_defaults(self):
        c = PersonCandidate(bbox=(0.0, 0.0, 1.0, 1.0), confidence=0.5, keypoints={})
        assert c.court_position is None

    def test_coco_keypoint_names(self):
        assert len(COCO_KEYPOINT_NAMES) == 17
        assert COCO_KEYPOINT_NAMES[0] == "nose"
        for name in ("left_wrist", "right_wrist", "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            assert name in COCO_KEYPOINT_NAMES


def _yolo_result(xyxy, confs, kxy=None, kconf=None) -> MagicMock:
    """Mimic an ultralytics Results object: .boxes.xyxy/.conf and .keypoints.xy/.conf tensors."""
    r = MagicMock()
    r.boxes.__len__.return_value = len(xyxy)
    r.boxes.xyxy.cpu.return_value.numpy.return_value = np.array(xyxy, dtype=np.float32)
    r.boxes.conf.cpu.return_value.numpy.return_value = np.array(confs, dtype=np.float32)
    if kxy is None:
        r.keypoints = None
    else:
        r.keypoints.xy.cpu.return_value.numpy.return_value = np.array(kxy, dtype=np.float32)
        if kconf is None:
            r.keypoints.conf = None
        else:
            r.keypoints.conf.cpu.return_value.numpy.return_value = np.array(kconf, dtype=np.float32)
    return r


class TestDetectPersons:
    def _frame(self) -> np.ndarray:
        return np.zeros(FRAME_SHAPE, dtype=np.uint8)

    @patch("court_vision.player_detect._get_pose_model")
    def test_returns_bbox_and_named_coco_keypoints(self, mock_get_model: MagicMock):
        kxy = [[[10.0 * j, 20.0 * j] for j in range(17)]]
        kconf = [[0.5 + 0.01 * j for j in range(17)]]
        model = MagicMock(return_value=[_yolo_result([[100.0, 200.0, 200.0, 500.0]], [0.92], kxy, kconf)])
        mock_get_model.return_value = model

        cands = detect_persons(self._frame(), model_name="custom-pose.pt", imgsz=640, conf=0.2)

        assert len(cands) == 1
        c = cands[0]
        assert c.bbox == (100.0, 200.0, 200.0, 500.0)
        assert c.confidence == pytest.approx(0.92, abs=1e-3)
        assert set(c.keypoints) == set(COCO_KEYPOINT_NAMES)
        assert c.keypoints["nose"] == pytest.approx((0.0, 0.0, 0.5), abs=1e-3)
        assert c.keypoints["right_ankle"] == pytest.approx((160.0, 320.0, 0.66), abs=1e-3)
        mock_get_model.assert_called_once_with("custom-pose.pt")
        kwargs = model.call_args.kwargs
        assert kwargs["conf"] == 0.2
        assert kwargs["imgsz"] == 640
        assert kwargs["classes"] == [0]  # person class only

    @patch("court_vision.player_detect._get_pose_model")
    def test_multiple_persons(self, mock_get_model: MagicMock):
        kxy = [[[0.0, 0.0]] * 17, [[5.0, 5.0]] * 17]
        kconf = [[0.9] * 17, [0.8] * 17]
        mock_get_model.return_value = MagicMock(return_value=[
            _yolo_result([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 60.0]], [0.9, 0.4], kxy, kconf),
        ])
        cands = detect_persons(self._frame())
        assert [c.bbox for c in cands] == [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 40.0, 60.0)]
        assert [round(c.confidence, 2) for c in cands] == [0.9, 0.4]
        assert cands[1].keypoints["nose"] == pytest.approx((5.0, 5.0, 0.8), abs=1e-3)

    @patch("court_vision.player_detect._get_pose_model")
    def test_no_detections(self, mock_get_model: MagicMock):
        empty = MagicMock()
        empty.boxes.__len__.return_value = 0
        none_boxes = MagicMock()
        none_boxes.boxes = None
        mock_get_model.return_value = MagicMock(return_value=[empty, none_boxes])
        assert detect_persons(self._frame()) == []

    @patch("court_vision.player_detect._get_pose_model")
    def test_missing_keypoints_gives_empty_dict(self, mock_get_model: MagicMock):
        mock_get_model.return_value = MagicMock(return_value=[_yolo_result([[1.0, 2.0, 3.0, 4.0]], [0.5])])
        cands = detect_persons(self._frame())
        assert cands[0].keypoints == {}

    @patch("court_vision.player_detect._get_pose_model")
    def test_missing_keypoint_confidence_defaults_to_one(self, mock_get_model: MagicMock):
        kxy = [[[1.0, 2.0]] * 17]
        mock_get_model.return_value = MagicMock(return_value=[_yolo_result([[1.0, 2.0, 3.0, 4.0]], [0.5], kxy, None)])
        cands = detect_persons(self._frame())
        assert cands[0].keypoints["nose"] == (1.0, 2.0, 1.0)


class TestGeometry:
    def test_pixel_court_round_trip(self):
        court = pixel_to_court(640.0, 660.0, H_LINEAR)
        assert court == pytest.approx((0.0, -12.0))
        back = court_to_pixel(*court, np.linalg.inv(H_LINEAR))
        assert back == pytest.approx((640.0, 660.0))

    def test_pixel_to_court_degenerate(self):
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        assert pixel_to_court(1.0, 1.0, H) is None

    def test_iou(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
        assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


class TestMapPlayerToCourt:
    def test_maps_bbox_center_bottom_to_court(self):
        det = PlayerDetection(frame_index=0, bbox=(100.0, 200.0, 200.0, 500.0), confidence=0.9)
        court_pos = map_player_to_court(det, np.eye(3, dtype=np.float64))
        assert court_pos == pytest.approx((150.0, 500.0))

    def test_returns_none_for_none_homography(self):
        det = PlayerDetection(frame_index=0, bbox=(100.0, 200.0, 200.0, 500.0), confidence=0.9)
        assert map_player_to_court(det, None) is None


class TestFarCourtRoi:
    def test_roi_covers_far_half_only(self):
        H = _perspective_homography()
        roi = far_court_roi(H, FRAME_SHAPE)
        assert roi is not None
        x1, y1, x2, y2 = roi
        assert 0 <= x1 < x2 <= 1280
        assert 0 <= y1 < y2 <= 720
        assert y1 <= 250  # reaches the far baseline (and behind it)
        net_y = court_to_pixel(0.0, 0.0, np.linalg.inv(H))[1]
        assert y2 >= net_y  # includes the net
        assert y2 < 650  # but not the near baseline

    def test_singular_homography(self):
        assert far_court_roi(np.zeros((3, 3)), FRAME_SHAPE) is None

    def test_degenerate_depth_is_rejected(self):
        # Far baseline and net project within a couple of pixels of each other.
        H = np.array([[1.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
        assert far_court_roi(H, FRAME_SHAPE) is None


class TestDetectPersonsWithFarCrop:
    def _frame(self) -> np.ndarray:
        return np.zeros(FRAME_SHAPE, dtype=np.uint8)

    def test_no_homography_runs_full_frame_only(self):
        full = [_cand((600, 400, 700, 700))]
        with patch("court_vision.player_detect.detect_persons", return_value=full) as dp:
            out = detect_persons_with_far_crop(self._frame(), None)
        assert out == full
        assert dp.call_count == 1

    def test_far_crop_can_be_disabled(self):
        full = [_cand((600, 400, 700, 700))]
        with patch("court_vision.player_detect.detect_persons", return_value=full) as dp:
            out = detect_persons_with_far_crop(self._frame(), np.eye(3), far_crop=False)
        assert out == full
        assert dp.call_count == 1

    def test_crop_candidates_are_mapped_back_and_merged(self):
        full = [
            _cand((600, 400, 700, 700), 0.9, {"nose": (650.0, 420.0, 0.9)}),
            _cand((110, 60, 120, 80), 0.3),  # same person as the crop finds -> dropped as duplicate
        ]
        crop = [_cand((40, 40, 80, 120), 0.8, {"nose": (60.0, 48.0, 0.7)})]
        with patch("court_vision.player_detect.detect_persons", side_effect=[full, crop]) as dp, \
             patch("court_vision.player_detect.far_court_roi", return_value=(100, 50, 420, 290)):
            out = detect_persons_with_far_crop(self._frame(), np.eye(3), imgsz=1280)

        assert dp.call_count == 2
        # 320px-wide ROI is upscaled x4 (capped) before the second pass
        assert dp.call_args_list[1].args[0].shape == (960, 1280, 3)
        assert len(out) == 2
        mapped, kept = out
        assert mapped.bbox == pytest.approx((110.0, 60.0, 120.0, 80.0))
        assert mapped.confidence == 0.8
        assert mapped.keypoints["nose"] == pytest.approx((115.0, 62.0, 0.7))
        assert kept.bbox == (600.0, 400.0, 700.0, 700.0)

    def test_no_roi_falls_back_to_full_frame(self):
        full = [_cand((600, 400, 700, 700))]
        with patch("court_vision.player_detect.detect_persons", return_value=full) as dp, \
             patch("court_vision.player_detect.far_court_roi", return_value=None):
            out = detect_persons_with_far_crop(self._frame(), np.eye(3))
        assert out == full
        assert dp.call_count == 1


class TestPlayerTrackerWithoutHomography:
    def test_splits_frame_into_near_and_far(self):
        tracker = PlayerTracker(FRAME_SHAPE, None)
        near = _cand((600, 400, 700, 700), 0.9, {"nose": (650.0, 420.0, 0.9)})
        far = _cand((620, 100, 660, 200), 0.8)
        players, poses = tracker.update(0, [near, far])
        assert [(p.role, p.bbox) for p in players] == [("near_player", near.bbox), ("far_player", far.bbox)]
        assert all(p.frame_index == 0 for p in players)
        assert [(p.role, p.keypoints) for p in poses] == [("near_player", near.keypoints), ("far_player", {})]
        assert players[0].court_position is None

    def test_tiny_margin_blob_is_not_a_player(self):
        tracker = PlayerTracker(FRAME_SHAPE, None)
        blob = _cand((10, 600, 40, 660), 0.7)
        players, poses = tracker.update(0, [blob])
        assert players == [] and poses == []

    def test_large_margin_candidate_is_kept(self):
        tracker = PlayerTracker(FRAME_SHAPE, None)
        players, _ = tracker.update(0, [_cand((1100, 300, 1250, 700), 0.7)])
        assert [p.role for p in players] == ["near_player"]

    def test_best_candidate_per_half(self):
        tracker = PlayerTracker(FRAME_SHAPE, None)
        strong = _cand((600, 400, 700, 700), 0.9)
        weak = _cand((300, 500, 340, 600), 0.3)
        players, _ = tracker.update(0, [weak, strong])
        assert [(p.role, p.bbox) for p in players] == [("near_player", strong.bbox)]

    def test_missing_detection_carries_previous_track(self):
        tracker = PlayerTracker(FRAME_SHAPE, None)
        near = _cand((600, 400, 700, 700), 0.9, {"nose": (650.0, 420.0, 0.9)})
        tracker.update(0, [near])
        players, poses = tracker.update(1, [])
        assert [(p.role, p.bbox, p.frame_index) for p in players] == [("near_player", near.bbox, 1)]
        assert [(p.role, p.frame_index) for p in poses] == [("near_player", 1)]
        assert tracker.missing["near_player"] == 1

    def test_track_expires_after_max_missing(self):
        tracker = PlayerTracker(FRAME_SHAPE, None, max_missing=3)
        tracker.update(0, [_cand((600, 400, 700, 700), 0.9)])
        for i in range(1, 4):
            players, _ = tracker.update(i, [])
            assert len(players) == 1
        players, poses = tracker.update(4, [])
        assert players == [] and poses == []

    def test_implausible_jump_is_rejected_while_track_is_trusted(self):
        tracker = PlayerTracker(FRAME_SHAPE, None)
        near = _cand((600, 400, 700, 700), 0.9)
        tracker.update(0, [near])
        jumped = _cand((100, 400, 200, 700), 0.95)
        players, _ = tracker.update(1, [jumped])
        assert [(p.role, p.bbox, p.frame_index) for p in players] == [("near_player", near.bbox, 1)]

    def test_nearby_candidate_continues_the_track(self):
        tracker = PlayerTracker(FRAME_SHAPE, None)
        tracker.update(0, [_cand((600, 400, 700, 700), 0.9)])
        moved = _cand((620, 400, 720, 700), 0.6)
        players, _ = tracker.update(1, [moved])
        assert [(p.role, p.bbox) for p in players] == [("near_player", moved.bbox)]


class TestPlayerTrackerWithHomography:
    def test_court_half_picks_the_role(self):
        tracker = PlayerTracker(FRAME_SHAPE, H_LINEAR)
        near = _cand((600, 400, 680, 660), 0.9)  # feet at (640, 660) -> court (0, -12)
        far = _cand((620, 40, 660, 100), 0.8)  # feet at (640, 100) -> court (0, 10.4)
        players, _ = tracker.update(0, [far, near])
        assert [(p.role, p.bbox) for p in players] == [("near_player", near.bbox), ("far_player", far.bbox)]
        assert players[0].court_position == pytest.approx((0.0, -12.0))
        assert players[1].court_position == pytest.approx((0.0, 10.4))

    def test_candidates_outside_the_court_box_are_dropped(self):
        tracker = PlayerTracker(FRAME_SHAPE, H_LINEAR, max_court_x=8.0)
        outside = _cand((1150, 400, 1250, 660), 0.99)  # feet at (1200, 660) -> court x = 11.2
        players, _ = tracker.update(0, [outside])
        assert players == []

    def test_body_at_the_net_post_is_not_a_player(self):
        tracker = PlayerTracker(FRAME_SHAPE, H_LINEAR)
        ball_kid = _cand((970, 200, 1010, 335), 0.95)  # feet -> court (7.0, 1.0)
        players, _ = tracker.update(0, [ball_kid])
        assert players == []

    def test_baseline_body_preferred_over_net_body_without_history(self):
        tracker = PlayerTracker(FRAME_SHAPE, H_LINEAR)
        baseline = _cand((620, 20, 660, 110), 0.8)  # court (0, 10)
        at_net = _cand((620, 232.5, 660, 322.5), 0.8)  # court (0, 1.5)
        players, _ = tracker.update(0, [at_net, baseline])
        assert [(p.role, p.bbox) for p in players] == [("far_player", baseline.bbox)]

    def test_degenerate_projection_is_dropped(self):
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        tracker = PlayerTracker(FRAME_SHAPE, H)
        players, _ = tracker.update(0, [_cand((600, 400, 700, 700), 0.9)])
        assert players == []


class TestFinalizeBallByFrame:
    def test_builds_per_frame_lookup_and_trims_weak_edges(self):
        seg = _segment(10, 19)
        dets = [
            BallDetection(10, 0.0, 0.0, 0.3),  # weak before the first strong -> dropped
            BallDetection(11, 20.0, 20.0, 0.9),
            BallDetection(12, 40.0, 40.0, 0.9),
            BallDetection(13, 60.0, 60.0, 0.9),
            BallDetection(14, 80.0, 80.0, 0.9),
            BallDetection(15, 100.0, 100.0, 0.3),  # weak after the last strong -> dropped
            BallDetection(30, 1.0, 1.0, 0.9),  # outside the segment -> ignored
        ]
        out = finalize_ball_by_frame(dets, seg)
        assert sorted(out) == list(range(10, 20))
        assert {k for k, v in out.items() if v is not None} == {11, 12, 13, 14}
        assert out[12] is dets[2]

    def test_stationary_track_is_rejected(self):
        """The stationary window is one second long: a ball parked for 40 frames at 30 fps goes."""
        seg = _segment(0, 39)
        dets = [BallDetection(i, 100.0 + (i % 2), 200.0, 0.9) for i in range(40)]
        out = finalize_ball_by_frame(dets, seg, stationary_std_px=3.0)
        assert all(v is None for v in out.values())

    def test_stationary_window_scales_with_fps(self):
        seg = _segment(0, 9)
        dets = [BallDetection(i, 100.0 + (i % 2), 200.0, 0.9) for i in range(10)]
        # ten frames are less than a second at 30 fps: nothing can be judged stationary...
        assert all(v is not None for v in finalize_ball_by_frame(dets, seg, stationary_std_px=3.0, fps=30.0).values())
        # ...but they are two seconds at 5 fps
        assert all(v is None for v in finalize_ball_by_frame(dets, seg, stationary_std_px=3.0, fps=5.0).values())

    def test_strong_confidence_threshold(self):
        seg = _segment(0, 4)
        dets = [BallDetection(i, 20.0 * i, 20.0 * i, 0.5) for i in range(5)]
        assert all(v is not None for v in finalize_ball_by_frame(dets, seg, strong_confidence=0.4).values())
        # no detection is "strong" -> nothing trimmed either
        assert all(v is not None for v in finalize_ball_by_frame(dets, seg, strong_confidence=0.9).values())


class TestBuildBallTrajectory:
    def test_threads_knobs_and_returns_lookup(self):
        seg = _segment(10, 19)
        traj = BallTrajectory(
            detections=[BallDetection(f, 20.0 * (f - 10), 20.0 * (f - 10), 0.9) for f in range(11, 16)],
            fps=60.0,
        )
        with patch("court_vision.player_detect.build_trajectory", return_value=traj) as bt:
            out = build_ball_trajectory(
                Path("/frames"), seg, fps=60.0, ball_method="tracknet", confidence_threshold=0.4,
                frame_step=2, max_speed_px=99.0, max_gap_s=0.7, smooth_window=5,
            )
        assert bt.call_args.args == (Path("/frames"), 10, 19)
        kwargs = bt.call_args.kwargs
        assert kwargs["fps"] == 60.0
        assert kwargs["method"] == "tracknet"
        assert kwargs["confidence_threshold"] == 0.4
        assert kwargs["frame_step"] == 2
        assert kwargs["max_speed_px"] == 99.0
        assert kwargs["max_gap_s"] == 0.7
        assert kwargs["smooth_window"] == 5
        assert sorted(out) == list(range(10, 20))
        assert {k for k, v in out.items() if v is not None} == {11, 12, 13, 14, 15}


class TestDetectPlayersSegment:
    def _frames_dir(self, tmp_path: Path, indices) -> Path:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in indices:
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), np.zeros(FRAME_SHAPE, dtype=np.uint8))
        return frames_dir

    def _cands(self) -> list[PersonCandidate]:
        return [
            _cand((600, 400, 700, 700), 0.9, {"nose": (650.0, 420.0, 0.9)}),
            _cand((620, 100, 660, 200), 0.8),
        ]

    def test_tracks_every_frame_and_merges_ball(self, tmp_path: Path):
        frames_dir = self._frames_dir(tmp_path, range(3))
        seg = _segment(0, 2)
        ball = {0: BallDetection(0, 1.0, 1.0, 0.9), 1: None, 2: BallDetection(2, 3.0, 3.0, 0.9)}
        with patch("court_vision.player_detect.detect_persons_with_far_crop", return_value=self._cands()) as det:
            results = detect_players_segment(frames_dir, seg, ball, homography=None)

        assert det.call_count == 3
        assert [r.frame_index for r in results] == [0, 1, 2]
        assert [r.ball is not None for r in results] == [True, False, True]
        for r in results:
            assert [p.role for p in r.players] == ["near_player", "far_player"]
            assert [p.frame_index for p in r.players] == [r.frame_index] * 2
            assert [p.role for p in r.poses] == ["near_player", "far_player"]
        assert results[0].poses[0].keypoints == {"nose": (650.0, 420.0, 0.9)}

    def test_stride_reuses_previous_detection(self, tmp_path: Path):
        frames_dir = self._frames_dir(tmp_path, range(5))
        seg = _segment(0, 4)
        progress = []
        with patch("court_vision.player_detect.detect_persons_with_far_crop", return_value=self._cands()) as det:
            results = detect_players_segment(
                frames_dir, seg, {}, homography=None, player_detect_stride=2,
                progress_callback=lambda cur, tot: progress.append((cur, tot)),
                model_name="x-pose.pt", imgsz=640, conf=0.3, far_crop=False,
            )
        assert det.call_count == 3  # frames 0, 2, 4
        assert det.call_args.kwargs == {"model_name": "x-pose.pt", "imgsz": 640, "conf": 0.3, "far_crop": False, "far_tiles": False}
        assert [r.frame_index for r in results] == [0, 1, 2, 3, 4]
        assert [p.frame_index for p in results[1].players] == [1, 1]  # copied forward with the new index
        assert progress == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]

    def test_missing_frames_are_skipped(self, tmp_path: Path):
        frames_dir = self._frames_dir(tmp_path, [0, 2])
        seg = _segment(0, 2)
        with patch("court_vision.player_detect.detect_persons_with_far_crop", return_value=[]):
            results = detect_players_segment(frames_dir, seg, {}, homography=None)
        assert [r.frame_index for r in results] == [0, 2]
        assert all(r.players == [] for r in results)

    def test_homography_is_forwarded_to_detector_and_tracker(self, tmp_path: Path):
        frames_dir = self._frames_dir(tmp_path, [0])
        seg = _segment(0, 0)
        near = _cand((600, 400, 680, 660), 0.9)  # feet -> court (0, -12)
        with patch("court_vision.player_detect.detect_persons_with_far_crop", return_value=[near]) as det:
            results = detect_players_segment(frames_dir, seg, {}, homography=H_LINEAR)
        assert det.call_args.args[1] is H_LINEAR
        assert results[0].players[0].court_position == pytest.approx((0.0, -12.0))


class TestFarCropTiles:
    def test_tiling_is_off_unless_asked(self):
        import inspect
        from court_vision.player_detect import detect_persons_with_far_crop, detect_players_segment

        assert inspect.signature(detect_persons_with_far_crop).parameters["far_tiles"].default is False
        assert inspect.signature(detect_players_segment).parameters["far_tiles"].default is False

    def test_narrow_roi_is_one_tile(self):
        from court_vision.player_detect import far_crop_tiles

        assert far_crop_tiles(300, 900, 1280) == [(300, 900)]

    def test_wide_roi_is_split_into_overlapping_tiles(self):
        from court_vision.player_detect import far_crop_tiles

        tiles = far_crop_tiles(0, 905, 1280)
        assert len(tiles) == 2
        assert tiles[0][0] == 0 and tiles[-1][1] == 905
        assert tiles[0][1] > tiles[1][0]                     # neighbours overlap
        assert all(t2 - t1 <= 0.5 * 1280 * 1.25 for t1, t2 in tiles)  # each gets ≥2x zoom at imgsz 1280
