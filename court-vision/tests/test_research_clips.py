"""Tests for the clip registry."""

from pathlib import Path

import pytest

from court_vision.research.clips import Clip, get_clip, load_clips

REGISTRY_YAML = """
clips:
  houston28s:
    video: data/houston.mp4
    frames_dir: data/houston_frames
    ground_truth: data/houston_gt.json
    stage_labels: data/houston_labels.json
    notes: "60fps, far player left-handed"
    overrides:
      far_player_hand: left
  vienna7s:
    video: {abs_video}
"""


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    path = tmp_path / "research" / "clips.yaml"
    path.parent.mkdir()
    path.write_text(REGISTRY_YAML.format(abs_video=tmp_path / "elsewhere" / "vienna.mp4"))
    return path


class TestLoadClips:
    def test_relative_paths_resolve_against_the_registry_grandparent(self, registry: Path, tmp_path: Path):
        clips = load_clips(registry)
        assert set(clips) == {"houston28s", "vienna7s"}
        c = clips["houston28s"]
        root = registry.resolve().parent.parent
        assert c.name == "houston28s"
        assert c.video == root / "data" / "houston.mp4"
        assert c.frames_dir == root / "data" / "houston_frames"
        assert c.ground_truth == root / "data" / "houston_gt.json"
        assert c.stage_labels == root / "data" / "houston_labels.json"
        assert c.notes == "60fps, far player left-handed"
        assert c.extra == {"overrides": {"far_player_hand": "left"}}

    def test_absolute_paths_are_kept(self, registry: Path, tmp_path: Path):
        c = load_clips(registry)["vienna7s"]
        assert c.video == tmp_path / "elsewhere" / "vienna.mp4"
        assert c.frames_dir is None and c.ground_truth is None and c.stage_labels is None
        assert c.notes == "" and c.extra == {}

    def test_explicit_root(self, registry: Path, tmp_path: Path):
        c = load_clips(registry, root=tmp_path / "other")["houston28s"]
        assert c.video == tmp_path / "other" / "data" / "houston.mp4"

    def test_has_ground_truth_requires_the_file(self, registry: Path):
        clips = load_clips(registry)
        assert clips["vienna7s"].has_ground_truth is False
        c = clips["houston28s"]
        assert c.has_ground_truth is False
        c.ground_truth.parent.mkdir(parents=True, exist_ok=True)
        c.ground_truth.write_text("{}")
        assert c.has_ground_truth is True

    def test_missing_registry_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_clips(tmp_path / "nope.yaml")

    def test_empty_registry(self, tmp_path: Path):
        path = tmp_path / "clips.yaml"
        path.write_text("")
        assert load_clips(path) == {}
        path.write_text("clips:\n")
        assert load_clips(path) == {}


class TestGetClip:
    def test_returns_named_clip(self, registry: Path):
        assert get_clip("vienna7s", registry).name == "vienna7s"

    def test_unknown_clip_lists_known_names(self, registry: Path):
        with pytest.raises(KeyError, match="houston28s"):
            get_clip("nope", registry)


class TestClipDataclass:
    def test_defaults(self, tmp_path: Path):
        c = Clip(name="x", video=tmp_path / "x.mp4")
        assert c.has_ground_truth is False
        assert c.extra == {} and c.notes == ""


class TestFixedCameraOptions:
    def test_calibration_path_and_single_segment(self, tmp_path: Path):
        reg = tmp_path / "research" / "clips.yaml"
        reg.parent.mkdir()
        reg.write_text(
            "clips:\n  cam:\n    video: data/cam.mov\n    calibration: data/cam_court.json\n    single_segment: true\n"
            "  tv:\n    video: data/tv.mp4\n"
        )
        clips = load_clips(reg)
        assert clips["cam"].calibration == tmp_path / "data" / "cam_court.json"
        assert clips["cam"].single_segment is True
        assert "calibration" not in clips["cam"].extra
        assert clips["tv"].calibration is None and clips["tv"].single_segment is False
