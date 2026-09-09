"""Clip registry: which example videos exist and where their labels live."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_REGISTRY = Path("research/clips.yaml")


@dataclass
class Clip:
    name: str
    video: Path
    frames_dir: Path | None = None
    ground_truth: Path | None = None
    stage_labels: Path | None = None
    calibration: Path | None = None  # fixed-camera court calibration JSON (see pipeline.load_fixed_court)
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def single_segment(self) -> bool:
        """A fixed camera with no cuts: treat the whole video as one gameplay segment."""
        return bool(self.extra.get("single_segment", False))

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth is not None and self.ground_truth.exists()


def load_clips(registry: Path | None = None, root: Path | None = None) -> dict[str, Clip]:
    """Load the clip registry. Relative paths resolve against ``root``
    (default: the registry file's parent's parent, i.e. court-vision/)."""
    registry = Path(registry or DEFAULT_REGISTRY)
    if not registry.exists():
        raise FileNotFoundError(f"clip registry not found: {registry}")
    root = Path(root) if root else registry.resolve().parent.parent
    raw = yaml.safe_load(registry.read_text()) or {}
    clips: dict[str, Clip] = {}
    for name, c in (raw.get("clips") or {}).items():
        def p(key: str) -> Path | None:
            v = c.get(key)
            if not v:
                return None
            path = Path(v)
            return path if path.is_absolute() else root / path
        clips[name] = Clip(
            name=name,
            video=p("video"),
            frames_dir=p("frames_dir"),
            ground_truth=p("ground_truth"),
            stage_labels=p("stage_labels"),
            calibration=p("calibration"),
            notes=str(c.get("notes", "")),
            extra={k: v for k, v in c.items() if k not in ("video", "frames_dir", "ground_truth", "stage_labels", "calibration", "notes")},
        )
    return clips


def get_clip(name: str, registry: Path | None = None) -> Clip:
    clips = load_clips(registry)
    if name not in clips:
        raise KeyError(f"unknown clip {name!r}; known: {sorted(clips)}")
    return clips[name]
