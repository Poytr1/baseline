"""Pipeline configuration with YAML loading."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class PipelineSettings(BaseModel):
    target_resolution: list[int] = [1280, 720]
    fps_override: int | None = None
    confidence_threshold: float = 0.7
    max_interpolation_gap_s: float = 0.5
    scene_filter_mode: Literal["heuristic", "ml"] = "heuristic"
    gameplay_threshold: float = 0.45
    ball_detection_method: Literal["wasb", "tracknet", "hsv"] = "tracknet"
    player_detect_stride: int = 3
    scene_filter_stride: int = 5


class OutputSettings(BaseModel):
    directory: str = "output/"
    format: str = "json"


class PipelineConfig(BaseModel):
    pipeline: PipelineSettings = PipelineSettings()
    output: OutputSettings = OutputSettings()
    device: str = "auto"


def load_config(config_path: Path | None) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to a YAML config file. If None, returns defaults.
    """
    if config_path is None or not config_path.exists():
        return PipelineConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return PipelineConfig.model_validate(raw)
