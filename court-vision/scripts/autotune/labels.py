"""Per-stage ground-truth labels for auto-tuning.

The repo only ships final-stage shot labels (data/match_data.json). To score the
court / ball / player stages independently, the auto-tune workflow has vision
agents inspect a sample of the extracted frames and emit pixel-space labels.
Those labels are persisted here as a single JSON fixture and loaded back to
drive the per-stage metrics in metrics_stages.py.

Fixture schema (data/autotune_labels.json):

{
  "frames_dir": "data/frames",
  "image_size": [1280, 720],            # [width, height] the labels were made against
  "court": [
    {
      "frame": 100,
      "corners": {                       # pixel (x, y) of doubles-court corners
        "near_left":  [x, y],
        "near_right": [x, y],
        "far_left":   [x, y],
        "far_right":  [x, y]
      }
    }
  ],
  "ball": [
    {"frame": 9,  "x": 640.0, "y": 360.0, "visible": true},
    {"frame": 30, "x": null,  "y": null,  "visible": false}  # ball not visible
  ],
  "players": [
    {
      "frame": 50,
      "boxes": [                         # one entry per visible player
        {"role": "near_player", "bbox": [x1, y1, x2, y2]},
        {"role": "far_player",  "bbox": [x1, y1, x2, y2]}
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Canonical doubles-court corner order used everywhere downstream. Matches the
# ordering in court_detect.match_keypoints_to_court (near_left, near_right,
# far_right, far_left) so a labelled homography is directly comparable.
CORNER_ORDER = ("near_left", "near_right", "far_right", "far_left")


@dataclass
class CourtLabel:
    frame: int
    corners: dict[str, tuple[float, float]]  # role -> (x, y) pixel
    image_path: str | None = None  # absolute path; None means data/frames/frame_<frame>.jpg

    def ordered_pixels(self) -> list[tuple[float, float]] | None:
        """Return corners in CORNER_ORDER, or None if any are missing."""
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


@dataclass
class PlayerBox:
    role: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)


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
        return (
            f"{len(self.court)} court frame(s), "
            f"{len(self.ball)} ball frame(s) ({ball_visible} visible), "
            f"{len(self.players)} player frame(s)"
        )


def load_labels(path: Path) -> LabelSet:
    """Load the per-stage label fixture from JSON."""
    with open(path) as f:
        raw = json.load(f)

    court = [
        CourtLabel(
            frame=int(c["frame"]),
            corners={k: (float(v[0]), float(v[1])) for k, v in c["corners"].items()},
            image_path=c.get("image_path"),
        )
        for c in raw.get("court", [])
    ]
    ball = [
        BallLabel(
            frame=int(b["frame"]),
            x=None if b.get("x") is None else float(b["x"]),
            y=None if b.get("y") is None else float(b["y"]),
            visible=bool(b.get("visible", b.get("x") is not None)),
            image_path=b.get("image_path"),
        )
        for b in raw.get("ball", [])
    ]
    players = [
        PlayerLabel(
            frame=int(p["frame"]),
            boxes=[
                PlayerBox(role=str(box["role"]), bbox=tuple(float(v) for v in box["bbox"]))
                for box in p.get("boxes", [])
            ],
            image_path=p.get("image_path"),
        )
        for p in raw.get("players", [])
    ]

    size = raw.get("image_size", [1280, 720])
    return LabelSet(
        frames_dir=raw.get("frames_dir", "data/frames"),
        image_size=(int(size[0]), int(size[1])),
        court=court,
        ball=ball,
        players=players,
    )


def empty_labels(frames_dir: str = "data/frames", image_size=(1280, 720)) -> LabelSet:
    """An empty label set — lets the scorecard run before labels exist."""
    return LabelSet(frames_dir=frames_dir, image_size=tuple(image_size))


def image_path_for(label, frames_dir: str) -> Path:
    """Resolve the image file for a label.

    External-dataset labels carry an explicit absolute image_path; local-clip
    labels leave it None and fall back to frames_dir/frame_<frame>.jpg.
    """
    if getattr(label, "image_path", None):
        return Path(label.image_path)
    return Path(frames_dir) / f"frame_{int(label.frame):06d}.jpg"


def merge(*sets: LabelSet) -> LabelSet:
    """Combine several label sets (e.g. local clip + public dataset) into one.

    Uses the first set's frames_dir/image_size as the base. Labels whose
    image_path is None are assumed to belong to the base frames_dir; labels from
    other sets should carry absolute image_path values so they remain resolvable.
    """
    if not sets:
        return empty_labels()
    base = sets[0]
    out = LabelSet(frames_dir=base.frames_dir, image_size=base.image_size)
    for s in sets:
        out.court.extend(s.court)
        out.ball.extend(s.ball)
        out.players.extend(s.players)
    return out
