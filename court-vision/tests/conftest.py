"""Shared test fixtures for Court Vision tests."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory for test artifacts."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    """Create a minimal config file for testing."""
    config_content = """
pipeline:
  target_resolution: [1280, 720]
  fps_override: null
  confidence_threshold: 0.7
  max_interpolation_gap_s: 0.5

output:
  directory: output/
  format: json

device: auto
"""
    config_file = tmp_path / "court-vision.yaml"
    config_file.write_text(config_content)
    return config_file
