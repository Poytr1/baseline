"""Tests for the pipeline orchestrator and its stage functions."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from court_vision.config import PipelineConfig, PipelineSettings
from court_vision.court_detect import CourtDetectionResult
from court_vision.ingest import FrameSequence
from court_vision.pipeline import (
    PipelineResult,
    auto_stride,
    run_pipeline,
    stage_ball,
    stage_court,
    stage_players,
    stage_scene,
    stage_scoreboard,
    stage_shots,
)
from court_vision.player_detect import FrameTrackingResult
from court_vision.scene_filter import GameplaySegment
from court_vision.scoreboard import ScoreTimeline
from court_vision.shot_classify import MatchData


@pytest.fixture(autouse=True)
def _no_ocr():
    """Never run scoreboard OCR in tests."""
    with patch("court_vision.pipeline.read_scoreboard_timeline", return_value=ScoreTimeline()):
        yield


def _segment(start: int, end: int, fps: float = 30.0) -> GameplaySegment:
    return GameplaySegment(start_frame=start, end_frame=end, start_time_s=start / fps,
                           end_time_s=end / fps, frame_count=end - start + 1)


def _frame_seq(frames_dir: Path, fps: float = 30.0, total_frames: int = 100) -> FrameSequence:
    return FrameSequence(frames_dir=frames_dir, fps=fps, total_frames=total_frames, resolution=(1280, 720))


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig(pipeline=PipelineSettings(**overrides))


@contextmanager
def patched_pipeline(tmp_path: Path, config: PipelineConfig | None = None, fps: float = 30.0,
                     total_frames: int = 100, segments=None, court=None, tracking=None, match=None):
    """Patch every heavy stage of ``run_pipeline`` and yield the mocks."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(exist_ok=True)
    with ExitStack() as stack:
        p = lambda name, **kw: stack.enter_context(patch(f"court_vision.pipeline.{name}", **kw))  # noqa: E731
        m = SimpleNamespace(frames_dir=frames_dir)
        m.load_config = p("load_config", return_value=config or _config())
        m.classify_frames_heuristic = p("classify_frames_heuristic", return_value=[])
        m.smooth_classifications = p("smooth_classifications", return_value=[])
        m.filter_gameplay_segments = p("filter_gameplay_segments", return_value=list(segments or []))
        m.download_video = p("download_video")
        m.extract_frames = p("extract_frames", return_value=_frame_seq(frames_dir, fps, total_frames))
        m.compute_segment_homographies = p("compute_segment_homographies", return_value=list(court or []))
        m.build_ball_trajectory = p("build_ball_trajectory", return_value={})
        m.detect_players_segment = p("detect_players_segment", return_value=list(tracking or []))
        m.read_scoreboard_timeline = p("read_scoreboard_timeline", return_value=ScoreTimeline())
        m.build_match_data = p("build_match_data", return_value=match)
        yield m


class TestAutoStride:
    @pytest.mark.parametrize("fps,expected", [(30.0, 1), (29.97, 1), (24.0, 1), (50.0, 2), (59.94, 2), (60.0, 2), (120.0, 4)])
    def test_auto_rounds_fps_over_30(self, fps: float, expected: int):
        assert auto_stride("auto", fps) == expected

    def test_explicit_values_pass_through(self):
        assert auto_stride(3, 60.0) == 3
        assert auto_stride(1, 60.0) == 1

    def test_never_below_one(self):
        assert auto_stride(0, 30.0) == 1
        assert auto_stride("auto", 5.0) == 1


class TestRunPipeline:
    def test_local_file_pipeline(self, tmp_path: Path):
        """Pipeline orchestrates ingest -> scene filter for a local file."""
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        with patched_pipeline(tmp_path, segments=[_segment(0, 50)]) as m:
            result = run_pipeline(str(video_path), config_path=None)

        assert isinstance(result, PipelineResult)
        assert result.total_frames == 100
        assert result.fps == 30.0
        assert result.frames_dir == m.frames_dir
        assert len(result.gameplay_segments) == 1
        assert result.gameplay_frame_count == 51
        m.extract_frames.assert_called_once()
        m.classify_frames_heuristic.assert_called_once()
        m.download_video.assert_not_called()

    def test_youtube_url_triggers_download(self, tmp_path: Path):
        downloaded = tmp_path / "video.mp4"
        downloaded.touch()
        with patched_pipeline(tmp_path, total_frames=50) as m:
            m.download_video.return_value = downloaded
            result = run_pipeline("https://www.youtube.com/watch?v=abc123", config_path=None, output_dir=tmp_path)

        m.download_video.assert_called_once()
        assert m.download_video.call_args.args[0] == "https://www.youtube.com/watch?v=abc123"
        assert m.extract_frames.call_args.args[0] == downloaded
        assert result.total_frames == 50

    def test_prebuilt_config_takes_precedence_over_config_path(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        config = _config(ball_detection_method="tracknet")
        with patched_pipeline(tmp_path, segments=[_segment(0, 50)]) as m:
            run_pipeline(str(video_path), config_path=tmp_path / "ignored.yaml", config=config)

        m.load_config.assert_not_called()
        assert m.build_ball_trajectory.call_args.kwargs["ball_method"] == "tracknet"

    def test_output_dir_defaults_to_config_and_is_created(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        config = PipelineConfig(pipeline=PipelineSettings())
        config.output.directory = str(tmp_path / "out" / "nested")
        with patched_pipeline(tmp_path, config=config):
            run_pipeline(str(video_path), config_path=None)
        assert (tmp_path / "out" / "nested").is_dir()

    def test_short_segments_are_dropped(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        segments = [_segment(0, 10), _segment(100, 200)]  # 11 frames < 1 s at 30 fps
        with patched_pipeline(tmp_path, segments=segments):
            result = run_pipeline(str(video_path), config_path=None)
        assert [(s.start_frame, s.end_frame) for s in result.gameplay_segments] == [(100, 200)]


class TestRunPipelineWithCourtDetection:
    def test_pipeline_runs_court_detection_after_scene_filter(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        segments = [_segment(0, 50)]
        court_result = CourtDetectionResult(success=True, homography=np.eye(3), num_lines_detected=6)
        with patched_pipeline(tmp_path, segments=segments, court=[court_result]) as m:
            result = run_pipeline(str(video_path), config_path=None)

        m.compute_segment_homographies.assert_called_once_with(m.frames_dir, segments, method="auto")
        assert result.court_detections is not None
        assert len(result.court_detections) == 1
        assert result.court_detections[0].success is True

    def test_court_method_from_config(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        with patched_pipeline(tmp_path, config=_config(court_method="classical"), segments=[_segment(0, 50)]) as m:
            run_pipeline(str(video_path), config_path=None)
        assert m.compute_segment_homographies.call_args.kwargs == {"method": "classical"}


class TestRunPipelineWithTracking:
    def test_pipeline_runs_tracking_after_court_detection(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        tracking = [FrameTrackingResult(frame_index=0, ball=None, players=[], poses=[])]
        with patched_pipeline(tmp_path, segments=[_segment(0, 50)],
                              court=[CourtDetectionResult(success=True, num_lines_detected=6)],
                              tracking=tracking) as m:
            result = run_pipeline(str(video_path), config_path=None)

        m.build_ball_trajectory.assert_called_once()
        m.detect_players_segment.assert_called_once()
        assert result.tracking_results is not None
        assert len(result.tracking_results) == 1

    def test_ball_detection_method_threaded_to_build_ball_trajectory(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        config = _config(ball_detection_method="tracknet", ball_confidence_threshold=0.42)
        with patched_pipeline(tmp_path, config=config, segments=[_segment(0, 50)]) as m:
            run_pipeline(str(video_path), config_path=None)

        kwargs = m.build_ball_trajectory.call_args.kwargs
        assert kwargs["ball_method"] == "tracknet"
        assert kwargs["confidence_threshold"] == 0.42

    def test_one_tracking_call_per_segment(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        segments = [_segment(0, 50), _segment(100, 200)]
        with patched_pipeline(tmp_path, segments=segments) as m:
            run_pipeline(str(video_path), config_path=None)
        assert m.build_ball_trajectory.call_count == 2
        assert m.detect_players_segment.call_count == 2
        assert [c.args[1] for c in m.detect_players_segment.call_args_list] == segments


class TestRunPipelineWithShotClassification:
    def test_pipeline_runs_shot_classification(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        match = MatchData(match_id="test", source_url="test.mp4", metadata={}, points=[])
        with patched_pipeline(tmp_path, segments=[_segment(0, 50)],
                              court=[CourtDetectionResult(success=True, num_lines_detected=6)], match=match) as m:
            result = run_pipeline(str(video_path), config_path=None)

        m.build_match_data.assert_called_once()
        assert result.match_data is match

    def test_homography_and_settings_passed_to_build_match_data(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        fake_H = np.eye(3, dtype=np.float64)
        config = _config(outcome_method="last_hitter")
        court = [CourtDetectionResult(success=False), CourtDetectionResult(success=True, homography=fake_H, num_lines_detected=6)]
        with patched_pipeline(tmp_path, config=config, segments=[_segment(0, 50), _segment(100, 200)], court=court) as m:
            run_pipeline(str(video_path), config_path=None)

        kwargs = m.build_match_data.call_args.kwargs
        assert np.array_equal(kwargs["court_homography"], fake_H)  # first successful homography
        assert kwargs["settings"] is config.pipeline
        assert kwargs["segment_homographies"][0] is None
        assert np.array_equal(kwargs["segment_homographies"][1], fake_H)
        assert kwargs["fps"] == 30.0
        assert kwargs["source"] == str(video_path)


class TestRunPipelineScoreboard:
    def test_scoreboard_read_and_forwarded_in_auto_mode(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        timeline = ScoreTimeline(roi=(1, 2, 3, 4))
        with patched_pipeline(tmp_path, segments=[_segment(0, 50)]) as m:
            m.read_scoreboard_timeline.return_value = timeline
            result = run_pipeline(str(video_path), config_path=None)

        m.read_scoreboard_timeline.assert_called_once()
        assert m.read_scoreboard_timeline.call_args.args[:3] == (m.frames_dir, 100, 30.0)
        assert m.read_scoreboard_timeline.call_args.kwargs["sample_s"] == 0.5
        assert result.scoreboard is timeline
        assert m.build_match_data.call_args.kwargs["scoreboard"] is timeline

    @pytest.mark.parametrize("method", ["last_hitter", "trajectory"])
    def test_scoreboard_skipped_when_not_needed(self, tmp_path: Path, method: str):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        with patched_pipeline(tmp_path, config=_config(outcome_method=method), segments=[_segment(0, 50)]) as m:
            result = run_pipeline(str(video_path), config_path=None)
        m.read_scoreboard_timeline.assert_not_called()
        assert result.scoreboard is None
        assert m.build_match_data.call_args.kwargs["scoreboard"] is None

    def test_ocr_failure_never_fails_the_pipeline(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        with patched_pipeline(tmp_path, segments=[_segment(0, 50)]) as m:
            m.read_scoreboard_timeline.side_effect = RuntimeError("tesseract exploded")
            result = run_pipeline(str(video_path), config_path=None)
        assert result.scoreboard is None
        m.build_match_data.assert_called_once()


class TestRunPipelineSceneFilter:
    def test_heuristic_filter_gets_config_knobs(self, tmp_path: Path):
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        with patched_pipeline(tmp_path, config=_config(scene_filter_stride=3, gameplay_threshold=0.5)) as m:
            run_pipeline(str(video_path), config_path=None)

        m.classify_frames_heuristic.assert_called_once()
        assert m.classify_frames_heuristic.call_args.kwargs["stride"] == 3
        assert m.classify_frames_heuristic.call_args.kwargs["gameplay_threshold"] == 0.5
        m.smooth_classifications.assert_called_once()
        assert m.smooth_classifications.call_args.kwargs["window_size"] == 5


class TestStageScene:
    def test_classifies_smooths_and_drops_short_segments(self, tmp_path: Path):
        frame_seq = _frame_seq(tmp_path)
        segments = [_segment(0, 10), _segment(100, 200)]
        with patch("court_vision.pipeline.classify_frames_heuristic", return_value=["r"]) as classify, \
             patch("court_vision.pipeline.smooth_classifications", return_value=["s"]) as smooth, \
             patch("court_vision.pipeline.filter_gameplay_segments", return_value=segments) as filt:
            out = stage_scene(frame_seq, _config(scene_filter_stride=2, min_segment_s=1.0))

        assert classify.call_args.args[:2] == (tmp_path, 100)
        assert classify.call_args.kwargs["stride"] == 2
        smooth.assert_called_once_with(["r"], window_size=5)
        filt.assert_called_once_with(["s"], 30.0)
        assert out == [segments[1]]  # 11-frame segment is shorter than min_segment_s


class TestStageCourt:
    def test_passes_method_from_config(self, tmp_path: Path):
        frame_seq = _frame_seq(tmp_path)
        segments = [_segment(0, 50)]
        with patch("court_vision.pipeline.compute_segment_homographies", return_value=["cd"]) as m:
            out = stage_court(frame_seq, segments, _config(court_method="neural"))
        m.assert_called_once_with(tmp_path, segments, method="neural")
        assert out == ["cd"]


class TestStageBall:
    def test_threads_config_knobs_and_auto_stride(self, tmp_path: Path):
        frame_seq = _frame_seq(tmp_path, fps=60.0)
        segments = [_segment(0, 9, 60.0), _segment(100, 109, 60.0)]
        config = _config(
            ball_detection_method="tracknet", ball_confidence_threshold=0.42, ball_frame_step="auto",
            ball_max_speed_px=120.0, ball_max_gap_s=0.4, ball_smooth_window=5,
            ball_stationary_std_px=3.0, ball_strong_confidence=0.8,
        )
        progress = []
        with patch("court_vision.pipeline.build_ball_trajectory", return_value={1: None}) as m:
            out = stage_ball(frame_seq, segments, config, progress_callback=lambda c, t: progress.append((c, t)))

        assert out == [{1: None}, {1: None}]
        assert m.call_count == 2
        first = m.call_args_list[0]
        assert first.args == (tmp_path, segments[0])
        kw = first.kwargs
        assert kw["fps"] == 60.0
        assert kw["ball_method"] == "tracknet"
        assert kw["confidence_threshold"] == 0.42
        assert kw["frame_step"] == 2  # auto at 60 fps
        assert kw["max_speed_px"] == 120.0
        assert kw["max_gap_s"] == 0.4
        assert kw["smooth_window"] == 5
        assert kw["stationary_std_px"] == 3.0
        assert kw["strong_confidence"] == 0.8
        # per-segment progress is offset into a single [0, total] range
        second_cb = m.call_args_list[1].kwargs["progress_callback"]
        second_cb(3, 10)
        assert progress[-1] == (13, 20)

    def test_explicit_frame_step(self, tmp_path: Path):
        frame_seq = _frame_seq(tmp_path, fps=60.0)
        with patch("court_vision.pipeline.build_ball_trajectory", return_value={}) as m:
            stage_ball(frame_seq, [_segment(0, 9)], _config(ball_frame_step=3))
        assert m.call_args.kwargs["frame_step"] == 3
        assert m.call_args.kwargs["progress_callback"] is None

    def test_no_segments(self, tmp_path: Path):
        with patch("court_vision.pipeline.build_ball_trajectory") as m:
            assert stage_ball(_frame_seq(tmp_path), [], _config()) == []
        m.assert_not_called()


class TestStagePlayers:
    def test_threads_homography_ball_and_knobs(self, tmp_path: Path):
        frame_seq = _frame_seq(tmp_path, fps=60.0)
        segments = [_segment(0, 9, 60.0), _segment(100, 109, 60.0)]
        H = np.eye(3)
        court = [CourtDetectionResult(success=True, homography=H), CourtDetectionResult(success=False)]
        balls = [{0: None}, {100: None}]
        config = _config(player_model="yolov8n-pose.pt", player_imgsz=960, player_conf=0.2,
                            player_detect_stride="auto", player_far_crop=False,
                            player_max_court_x=9.0, player_max_court_y=18.0)
        t0 = FrameTrackingResult(0, None, [], [])
        t1 = FrameTrackingResult(100, None, [], [])
        with patch("court_vision.pipeline.detect_players_segment", side_effect=[[t0], [t1]]) as m:
            out = stage_players(frame_seq, segments, court, balls, config)

        assert out == [t0, t1]
        first, second = m.call_args_list
        assert first.args == (tmp_path, segments[0], balls[0])
        assert first.kwargs["homography"] is H
        assert second.args == (tmp_path, segments[1], balls[1])
        assert second.kwargs["homography"] is None  # failed court detection
        kw = first.kwargs
        assert kw["player_detect_stride"] == 2
        assert kw["model_name"] == "yolov8n-pose.pt"
        assert kw["imgsz"] == 960
        assert kw["conf"] == 0.2
        assert kw["far_crop"] is False
        assert kw["max_court_x"] == 9.0
        assert kw["max_court_y"] == 18.0

    def test_missing_court_and_ball_data_are_tolerated(self, tmp_path: Path):
        frame_seq = _frame_seq(tmp_path)
        with patch("court_vision.pipeline.detect_players_segment", return_value=[]) as m:
            stage_players(frame_seq, [_segment(0, 9)], None, [], _config())
        assert m.call_args.args[2] == {}
        assert m.call_args.kwargs["homography"] is None


class TestStageScoreboard:
    def test_reads_timeline_with_sample_interval(self, tmp_path: Path):
        frame_seq = _frame_seq(tmp_path, fps=25.0, total_frames=500)
        timeline = ScoreTimeline()
        with patch("court_vision.pipeline.read_scoreboard_timeline", return_value=timeline) as m:
            assert stage_scoreboard(frame_seq, _config(scoreboard_sample_s=1.5)) is timeline
        m.assert_called_once_with(tmp_path, 500, 25.0, sample_s=1.5)


class TestStageShots:
    def test_uses_first_successful_homography_and_per_segment_list(self, tmp_path: Path):
        frame_seq = _frame_seq(tmp_path)
        segments = [_segment(0, 50), _segment(100, 200)]
        H1, H2 = np.eye(3), np.eye(3) * 2
        court = [CourtDetectionResult(success=False, homography=H1), CourtDetectionResult(success=True, homography=H2)]
        config = _config(contact_method="proximity")
        timeline = ScoreTimeline()
        match = MatchData("m", "src", {}, [])
        with patch("court_vision.pipeline.build_match_data", return_value=match) as m:
            out = stage_shots("src.mp4", frame_seq, segments, court, [], config, scoreboard=timeline)

        assert out is match
        kw = m.call_args.kwargs
        assert kw["source"] == "src.mp4"
        assert kw["segments"] == segments
        assert kw["tracking_results"] == []
        assert kw["fps"] == 30.0
        assert kw["court_homography"] is H2
        assert kw["settings"] is config.pipeline
        assert kw["scoreboard"] is timeline
        assert kw["segment_homographies"][0] is None
        assert kw["segment_homographies"][1] is H2

    def test_no_court_detections(self, tmp_path: Path):
        with patch("court_vision.pipeline.build_match_data", return_value=None) as m:
            stage_shots("src.mp4", _frame_seq(tmp_path), [], None, [], _config())
        assert m.call_args.kwargs["court_homography"] is None
        assert m.call_args.kwargs["segment_homographies"] == []


class TestFixedCamera:
    def test_whole_clip_segment_spans_every_frame(self):
        from court_vision.pipeline import whole_clip_segment

        seg = whole_clip_segment(SimpleNamespace(total_frames=300, fps=60.0))
        assert (seg.start_frame, seg.end_frame, seg.frame_count) == (0, 299, 300)
        assert seg.end_time_s == pytest.approx(299 / 60.0)

    def test_load_fixed_court_rescales_to_the_run_resolution(self, tmp_path):
        import json
        from court_vision.pipeline import load_fixed_court

        H = np.array([[0.01, 0.0, -6.4], [0.0, -0.05, 30.0], [0.0, 0.0, 1.0]])  # calibrated on 1280x720
        path = tmp_path / "court.json"
        path.write_text(json.dumps({"homography": H.tolist(), "frame_size": [1280, 720], "points": [{"pixel": [640, 360], "court": [0, 12]}]}))
        same = load_fixed_court(path, 2, (720, 1280))
        assert len(same) == 2 and all(c.success for c in same)
        np.testing.assert_allclose(same[0].homography, H)
        assert same[0].pixel_keypoints == [(640.0, 360.0)]
        # at half resolution a pixel maps to the same court point as twice the pixel did
        half = load_fixed_court(path, 1, (360, 640))[0].homography
        p = half @ np.array([320.0, 180.0, 1.0]); q = H @ np.array([640.0, 360.0, 1.0])
        np.testing.assert_allclose(p[:2] / p[2], q[:2] / q[2])
