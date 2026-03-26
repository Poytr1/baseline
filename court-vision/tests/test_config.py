"""Tests for configuration loading."""

from pathlib import Path

from court_vision.config import PipelineConfig, load_config


def test_load_config_from_yaml(sample_config_path: Path):
    """Loads config from a YAML file."""
    config = load_config(sample_config_path)
    assert isinstance(config, PipelineConfig)
    assert config.pipeline.target_resolution == [1280, 720]
    assert config.pipeline.confidence_threshold == 0.7
    assert config.device == "auto"


def test_load_config_defaults():
    """Returns default config when no file is provided."""
    config = load_config(None)
    assert isinstance(config, PipelineConfig)
    assert config.pipeline.target_resolution == [1280, 720]
    assert config.pipeline.confidence_threshold == 0.7
    assert config.device == "auto"


def test_config_output_directory_default():
    """Default output directory is 'output/'."""
    config = load_config(None)
    assert config.output.directory == "output/"


def test_config_device_override():
    """Device field accepts explicit device strings."""
    config = load_config(None)
    config.device = "cpu"
    assert config.device == "cpu"
