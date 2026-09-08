"""Pipeline configuration with YAML loading.

All tunable knobs live flat under ``pipeline:`` so the research harness can
address any of them with a dotted override such as
``pipeline.ball_confidence_threshold=0.4``. ``STAGE_PARAMS`` maps each pipeline
stage to the knobs that influence it; the harness uses it to build cache keys
so that changing a Stage-5 knob never re-runs neural inference.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel


class PipelineSettings(BaseModel):
    # ── ingest ──
    target_resolution: list[int] = [1280, 720]
    fps_override: int | None = None

    # ── scene filter / point segmentation ──
    gameplay_threshold: float = 0.45
    scene_filter_stride: int = 5
    scene_smooth_window: int = 5
    min_segment_s: float = 1.0

    # ── court detection ──
    court_method: Literal["auto", "neural", "classical"] = "auto"

    # ── ball detection / trajectory ──
    ball_detection_method: Literal["wasb", "tracknet"] = "wasb"
    ball_confidence_threshold: float = 0.3
    ball_frame_step: int | Literal["auto"] = "auto"  # temporal gap of the 3-frame window (auto = fps/30)
    ball_far_crop: bool = True  # second detector pass on an upscaled far-court crop
    ball_max_speed_px: float = 150.0  # per 30fps-frame; scaled by fps
    ball_max_gap_s: float = 0.5
    ball_smooth_window: int = 3
    ball_stationary_std_px: float = 3.0  # over a 1 s window
    ball_strong_confidence: float = 0.75

    # ── player detection + pose ──
    player_model: str = "yolov8s-pose.pt"
    player_imgsz: int = 1280
    player_conf: float = 0.15
    player_detect_stride: int | Literal["auto"] = "auto"  # auto = fps/30
    player_far_crop: bool = True
    player_max_court_x: float = 7.0  # metres; candidates beyond this are not players
    player_max_court_y: float = 17.0

    # ── contact (hit) detection ──
    contact_method: Literal["trajectory", "proximity"] = "trajectory"
    contact_min_gap_s: float = 0.6  # sweep 2026-09-08 (research/sweeps/contacts.yaml)
    contact_player_margin: float = 0.6  # bbox expansion ratio for hit gating
    contact_min_turn_deg: float = 30.0
    contact_min_speed_px: float = 1.0  # per 30fps-frame (absolute floor)
    contact_min_speed_norm: float = 0.02  # post-hit speed in hitter bbox-heights per 30fps-frame
    contact_swing_min: float = 0.12  # wrist speed (bbox-heights per 30fps-frame) for a swing candidate
    contact_use_swing: bool = True
    close_up_ratio: float = 0.55  # a "player" taller than this fraction of the frame is a close-up
    proximity_threshold: float = 100.0  # legacy proximity method
    min_frames_between_contacts: int = 10  # legacy proximity method

    # ── point segmentation within a gameplay segment ──
    point_split_gap_s: float = 4.0
    point_pad_before_s: float = 1.0
    point_pad_after_s: float = 2.5

    # ── stroke classification ──
    near_player_hand: Literal["auto", "right", "left"] = "auto"
    far_player_hand: Literal["auto", "right", "left"] = "auto"
    slice_lookback_s: float = 0.3
    slice_drop_ratio: float = 0.3
    slice_min_torso_px: float = 30.0  # pose too small to judge the swing path below this
    volley_max_court_y: float = 4.5

    # ── outcome ──
    outcome_method: Literal["auto", "scoreboard", "trajectory", "last_hitter"] = "auto"
    scoreboard_sample_s: float = 0.5


class OutputSettings(BaseModel):
    directory: str = "output/"
    format: str = "json"


class PipelineConfig(BaseModel):
    pipeline: PipelineSettings = PipelineSettings()
    output: OutputSettings = OutputSettings()
    device: str = "auto"


# Which pipeline knobs feed which stage (used for cache keys in the harness).
STAGE_PARAMS: dict[str, tuple[str, ...]] = {
    "ingest": ("target_resolution", "fps_override"),
    "scene": ("gameplay_threshold", "scene_filter_stride", "scene_smooth_window", "min_segment_s"),
    "court": ("court_method",),
    "ball": ("ball_detection_method", "ball_confidence_threshold", "ball_frame_step", "ball_far_crop"),
    "ball_post": ("ball_max_speed_px", "ball_max_gap_s", "ball_smooth_window",
                  "ball_stationary_std_px", "ball_strong_confidence"),
    "players": ("player_model", "player_imgsz", "player_conf", "player_detect_stride",
                "player_far_crop", "player_max_court_x", "player_max_court_y"),
    "scoreboard": ("scoreboard_sample_s",),
    "shots": ("contact_method", "contact_min_gap_s", "contact_player_margin",
              "contact_min_turn_deg", "contact_min_speed_px", "contact_min_speed_norm", "contact_swing_min",
              "contact_use_swing", "close_up_ratio", "proximity_threshold",
              "min_frames_between_contacts", "point_split_gap_s", "point_pad_before_s",
              "point_pad_after_s", "near_player_hand", "far_player_hand",
              "slice_lookback_s", "slice_drop_ratio", "slice_min_torso_px", "volley_max_court_y",
              "outcome_method"),
}

# Stage dependency order: a stage's cache key includes its upstream stages' keys.
STAGE_DEPS: dict[str, tuple[str, ...]] = {
    "ingest": (),
    "scene": ("ingest",),
    "court": ("ingest", "scene"),
    "ball": ("ingest", "scene", "court"),
    "ball_post": ("ball",),
    "players": ("ingest", "scene", "court"),
    "scoreboard": ("ingest",),
    "shots": ("scene", "court", "ball_post", "players", "scoreboard"),
}


def load_config(config_path: Path | None) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to a YAML config file. If None, returns defaults.
    """
    if config_path is None or not config_path.exists():
        return PipelineConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    return PipelineConfig.model_validate(raw)


def apply_overrides(config: PipelineConfig, overrides: dict[str, Any]) -> PipelineConfig:
    """Return a copy of ``config`` with dotted-key overrides applied.

    Keys may be ``pipeline.foo``, ``output.foo``, ``device`` or bare ``foo``
    (interpreted as ``pipeline.foo``).
    """
    data = config.model_dump()
    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) == 1 and parts[0] != "device":
            parts = ["pipeline", parts[0]]
        node = data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return PipelineConfig.model_validate(data)
