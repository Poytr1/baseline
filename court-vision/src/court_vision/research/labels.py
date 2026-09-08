"""Per-stage ground-truth labels (court corners, ball centres, player boxes).

The final shot list (match_data.json) only scores the end of the pipeline.
To tune the vision stages independently we keep a small pixel-space label
fixture per clip:

{
  "frames_dir": "data/frames",
  "image_size": [1280, 720],
  "court":   [{"frame": 100, "corners": {"near_left": [x, y], "near_right": [x, y],
                                          "far_left": [x, y], "far_right": [x, y]}}],
  "ball":    [{"frame": 9, "x": 640.0, "y": 360.0, "visible": true},
              {"frame": 30, "x": null, "y": null, "visible": false}],
  "players": [{"frame": 50, "boxes": [{"role": "near_player", "bbox": [x1, y1, x2, y2]}]}]
}

Labels may carry an absolute ``image_path`` (external datasets) instead of
resolving to ``frames_dir/frame_<n>.jpg``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CORNER_ORDER = ("near_left", "near_right", "far_right", "far_left")


@dataclass
class CourtLabel:
    frame: int
    corners: dict[str, tuple[float, float]]
    image_path: str | None = None

    def ordered_pixels(self) -> list[tuple[float, float]] | None:
        if not all(k in self.corners for k in CORNER_ORDER):
            return None
        return [tuple(self.corners[k]) for k in CORNER_ORDER]


@dataclass
class BallLabel:
    frame: int
    x: float | None
    y: float | None
    visible: bool
    image_path: str | None = None
    status: int | None = None  # TrackNet: 0 flying, 1 hit, 2 bounce


@dataclass
class PlayerBox:
    role: str
    bbox: tuple[float, float, float, float]


@dataclass
class PlayerLabel:
    frame: int
    boxes: list[PlayerBox]
    image_path: str | None = None


@dataclass
class LabelSet:
    frames_dir: str
    image_size: tuple[int, int]
    court: list[CourtLabel] = field(default_factory=list)
    ball: list[BallLabel] = field(default_factory=list)
    players: list[PlayerLabel] = field(default_factory=list)

    def summary(self) -> str:
        ball_visible = sum(1 for b in self.ball if b.visible)
        return (f"{len(self.court)} court frame(s), {len(self.ball)} ball frame(s) "
                f"({ball_visible} visible), {len(self.players)} player frame(s)")


def load_labels(path: Path) -> LabelSet:
    raw = json.loads(Path(path).read_text())
    court = [
        CourtLabel(frame=int(c["frame"]),
                   corners={k: (float(v[0]), float(v[1])) for k, v in c["corners"].items()},
                   image_path=c.get("image_path"))
        for c in raw.get("court", [])
    ]
    ball = [
        BallLabel(frame=int(b["frame"]),
                  x=None if b.get("x") is None else float(b["x"]),
                  y=None if b.get("y") is None else float(b["y"]),
                  visible=bool(b.get("visible", b.get("x") is not None)),
                  image_path=b.get("image_path"),
                  status=b.get("status"))
        for b in raw.get("ball", [])
    ]
    players = [
        PlayerLabel(frame=int(p["frame"]),
                    boxes=[PlayerBox(role=str(bx["role"]), bbox=tuple(float(v) for v in bx["bbox"]))
                           for bx in p.get("boxes", [])],
                    image_path=p.get("image_path"))
        for p in raw.get("players", [])
    ]
    size = raw.get("image_size", [1280, 720])
    return LabelSet(frames_dir=raw.get("frames_dir", "data/frames"),
                    image_size=(int(size[0]), int(size[1])),
                    court=court, ball=ball, players=players)


def empty_labels(frames_dir: str = "data/frames", image_size=(1280, 720)) -> LabelSet:
    return LabelSet(frames_dir=frames_dir, image_size=tuple(image_size))


def image_path_for(label, frames_dir: str) -> Path:
    if getattr(label, "image_path", None):
        return Path(label.image_path)
    return Path(frames_dir) / f"frame_{int(label.frame):06d}.jpg"
