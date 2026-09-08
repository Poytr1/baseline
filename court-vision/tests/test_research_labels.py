"""Tests for the per-stage label fixture loader."""

import json
from pathlib import Path

from court_vision.research.labels import (
    CORNER_ORDER,
    BallLabel,
    CourtLabel,
    empty_labels,
    image_path_for,
    load_labels,
)


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(raw))
    return path


class TestLoadLabels:
    def test_full_fixture(self, tmp_path: Path):
        raw = {
            "frames_dir": "data/frames",
            "image_size": [1920, 1080],
            "court": [{"frame": 100, "corners": {"near_left": [1, 2], "near_right": [3, 4], "far_left": [5, 6], "far_right": [7, 8]}}],
            "ball": [
                {"frame": 9, "x": 640.0, "y": 360.0, "visible": True, "status": 1},
                {"frame": 30, "x": None, "y": None, "visible": False},
                {"frame": 31, "x": 10, "y": 20},  # visibility inferred from x
                {"frame": 32},
            ],
            "players": [{"frame": 50, "boxes": [{"role": "near_player", "bbox": [1, 2, 3, 4]}], "image_path": "/abs/50.jpg"}],
        }
        labels = load_labels(_write(tmp_path, raw))
        assert labels.frames_dir == "data/frames"
        assert labels.image_size == (1920, 1080)
        assert labels.court[0].frame == 100
        assert labels.court[0].corners["far_right"] == (7.0, 8.0)
        assert labels.court[0].ordered_pixels() == [(1.0, 2.0), (3.0, 4.0), (7.0, 8.0), (5.0, 6.0)]
        b9, b30, b31, b32 = labels.ball
        assert (b9.x, b9.y, b9.visible, b9.status) == (640.0, 360.0, True, 1)
        assert (b30.x, b30.visible) == (None, False)
        assert (b31.x, b31.y, b31.visible, b31.status) == (10.0, 20.0, True, None)
        assert (b32.x, b32.visible) == (None, False)
        p = labels.players[0]
        assert p.frame == 50 and p.image_path == "/abs/50.jpg"
        assert p.boxes[0].role == "near_player" and p.boxes[0].bbox == (1.0, 2.0, 3.0, 4.0)
        assert "1 court frame(s), 4 ball frame(s) (2 visible), 1 player frame(s)" == labels.summary()

    def test_defaults_for_an_empty_fixture(self, tmp_path: Path):
        labels = load_labels(_write(tmp_path, {}))
        assert labels.frames_dir == "data/frames"
        assert labels.image_size == (1280, 720)
        assert labels.court == [] and labels.ball == [] and labels.players == []

    def test_incomplete_corners_have_no_order(self):
        assert CourtLabel(frame=1, corners={"near_left": (0.0, 0.0)}).ordered_pixels() is None
        assert CORNER_ORDER == ("near_left", "near_right", "far_right", "far_left")


class TestHelpers:
    def test_empty_labels(self):
        labels = empty_labels("x/frames", (640, 480))
        assert labels.frames_dir == "x/frames"
        assert labels.image_size == (640, 480)
        assert labels.summary().startswith("0 court frame(s)")

    def test_image_path_for(self):
        assert image_path_for(BallLabel(frame=7, x=None, y=None, visible=False), "data/frames") == Path("data/frames/frame_000007.jpg")
        assert image_path_for(BallLabel(frame=7, x=None, y=None, visible=False, image_path="/abs/7.png"), "data/frames") == Path("/abs/7.png")
