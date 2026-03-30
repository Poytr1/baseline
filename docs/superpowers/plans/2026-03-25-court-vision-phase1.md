# Court Vision Phase 1 — Project Scaffolding, Ingestion & Scene Filter

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the Court Vision Python subproject with a working partial pipeline: accept a video input → extract frames → classify frames as gameplay or non-gameplay.

**Architecture:** A Python package (`court_vision`) inside `court-vision/` at the project root. Uses `pyproject.toml` with `hatchling` for packaging and `typer` for CLI. Sequential pipeline where each stage is a separate module. Device-aware for Apple Silicon MPS, CUDA, and CPU fallback.

**Tech Stack:** Python 3.11+, PyTorch + torchvision (ResNet-18), OpenCV, yt-dlp, typer, pydantic (config), pytest

**Spec:** `docs/superpowers/specs/2026-03-25-court-vision-design.md`

---

## Phase 1 Scope

Phase 1 delivers:
1. Python project scaffolding with all config/tooling
2. Device selection module (MPS/CUDA/CPU)
3. Video ingestion (YouTube download + frame extraction)
4. Scene filter (ResNet-18 gameplay classification)
5. CLI to run the partial pipeline
6. YAML configuration loading

Phase 1 does NOT include: court detection, ball tracking, player detection, shot classification, export, or review UI. Those are Phase 2+.

**Note on spec refinements:** The spec's project structure lists `pipeline.py`, `scene_filter.py`, `court_detect.py`, `ball_tracker.py`, `player_detect.py`, `shot_classify.py`, `export.py`, `device.py`. Phase 1 refines this by adding `config.py` (Pydantic config model), `ingest.py` (video download + frame extraction, extracted from what would otherwise be inline in `pipeline.py`), and `cli.py` (Typer entry point). These are natural decompositions that keep modules focused.

**Deferred from spec (Phase 2+):** The spec mentions "court line detection as secondary signal" for the scene filter (Stage 2). Phase 1 implements pure ResNet-18 classification only. Court line detection as a supplementary gameplay signal will be added when the court detection module (Stage 3) is built, as it naturally shares that infrastructure.

---

## File Structure

```
court-vision/
├── src/
│   └── court_vision/
│       ├── __init__.py          # Package init, version
│       ├── config.py            # Pydantic config model, YAML loading
│       ├── device.py            # Device selection (MPS/CUDA/CPU)
│       ├── ingest.py            # Video download + frame extraction
│       ├── scene_filter.py      # ResNet-18 scene classifier
│       ├── pipeline.py          # Orchestrator (Phase 1: ingest + scene filter)
│       └── cli.py               # Typer CLI entry point
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_config.py           # Config loading tests
│   ├── test_device.py           # Device selection tests
│   ├── test_ingest.py           # Video ingestion tests
│   ├── test_scene_filter.py     # Scene filter tests
│   ├── test_pipeline.py         # Pipeline orchestration tests
│   └── test_cli.py              # CLI entry point tests
├── models/                      # Pre-trained weights (gitignored)
├── data/                        # Sample videos + test fixtures (gitignored)
├── output/                      # Pipeline output (gitignored)
├── court-vision.yaml            # Default config
├── pyproject.toml               # Project metadata, deps, scripts
└── README.md                    # Setup + usage instructions
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `court-vision/pyproject.toml`
- Create: `court-vision/src/court_vision/__init__.py`
- Create: `court-vision/tests/conftest.py`
- Create: `court-vision/court-vision.yaml`
- Create: `court-vision/README.md`
- Modify: `.gitignore` — add court-vision specific ignores

### Steps

- [ ] **Step 1: Create `court-vision/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "court-vision"
version = "0.1.0"
description = "Automated shot-by-shot tennis data from broadcast video"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "numpy>=1.24",
    "opencv-python>=4.8",
    "yt-dlp>=2024.0",
    "typer>=0.12",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[project.scripts]
court-vision = "court_vision.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `court-vision/src/court_vision/__init__.py`**

```python
"""Court Vision — Automated shot-by-shot tennis data from broadcast video."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create `court-vision/tests/conftest.py`**

```python
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
```

- [ ] **Step 4: Create `court-vision/court-vision.yaml`**

```yaml
pipeline:
  target_resolution: [1280, 720]
  fps_override: null
  confidence_threshold: 0.7
  max_interpolation_gap_s: 0.5

output:
  directory: output/
  format: json

device: auto
```

- [ ] **Step 5: Create `court-vision/README.md`**

```markdown
# Court Vision

Automated shot-by-shot tennis data extraction from broadcast video.

## Setup

```bash
cd court-vision
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Process a YouTube match (prints gameplay segments to stdout)
court-vision process "https://youtube.com/watch?v=abc123"

# Process a local video file
court-vision process match.mp4
```

## Development

```bash
pytest
```
```

- [ ] **Step 6: Add court-vision ignores to root `.gitignore`**

Append to `.gitignore`:
```
# court-vision
court-vision/models/
court-vision/data/
court-vision/output/
court-vision/.venv/
court-vision/__pycache__/
court-vision/src/*.egg-info/
```

- [ ] **Step 7: Verify project installs**

```bash
cd court-vision && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

Expected: Installs successfully, `court-vision --help` runs (will show "No commands" until CLI is built).

- [ ] **Step 8: Commit**

```bash
git add court-vision/pyproject.toml court-vision/src/court_vision/__init__.py court-vision/tests/conftest.py court-vision/court-vision.yaml court-vision/README.md .gitignore
git commit -m "feat(court-vision): scaffold Python subproject with pyproject.toml and config"
```

---

## Task 2: Device Selection Module

**Files:**
- Create: `court-vision/src/court_vision/device.py`
- Create: `court-vision/tests/test_device.py`

### Steps

- [ ] **Step 1: Write the failing test for `test_device.py`**

```python
"""Tests for device selection logic."""

from unittest.mock import patch

from court_vision.device import get_device


def test_get_device_returns_torch_device():
    """get_device returns a torch.device object."""
    import torch

    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_cpu_fallback():
    """Falls back to CPU when MPS and CUDA are unavailable."""
    import torch

    with patch.object(torch.backends.mps, "is_available", return_value=False):
        with patch("torch.cuda.is_available", return_value=False):
            device = get_device()
            assert device.type == "cpu"


def test_get_device_cuda_when_available():
    """Selects CUDA when available and MPS is not."""
    import torch

    with patch.object(torch.backends.mps, "is_available", return_value=False):
        with patch("torch.cuda.is_available", return_value=True):
            device = get_device()
            assert device.type == "cuda"


def test_get_device_mps_preferred():
    """Selects MPS when available (preferred over CUDA)."""
    import torch

    with patch.object(torch.backends.mps, "is_available", return_value=True):
        device = get_device()
        assert device.type == "mps"


def test_get_device_override():
    """Explicit device override bypasses auto-detection."""
    import torch

    device = get_device(override="cpu")
    assert device.type == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_device.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'court_vision.device'`

- [ ] **Step 3: Write `device.py`**

```python
"""Device selection for PyTorch inference."""

import torch


def get_device(override: str | None = None) -> torch.device:
    """Select the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA GPU) > CPU.

    Args:
        override: Force a specific device ("mps", "cuda", "cpu").
                  If None or "auto", auto-detect.
    """
    if override and override != "auto":
        return torch.device(override)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_device.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/device.py court-vision/tests/test_device.py
git commit -m "feat(court-vision): add device selection module with MPS/CUDA/CPU support"
```

---

## Task 3: Configuration Module

**Files:**
- Create: `court-vision/src/court_vision/config.py`
- Create: `court-vision/tests/test_config.py`

### Steps

- [ ] **Step 1: Write the failing test for `test_config.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'court_vision.config'`

- [ ] **Step 3: Write `config.py`**

```python
"""Pipeline configuration with YAML loading."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class PipelineSettings(BaseModel):
    target_resolution: list[int] = [1280, 720]
    fps_override: int | None = None
    confidence_threshold: float = 0.7
    max_interpolation_gap_s: float = 0.5


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_config.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/config.py court-vision/tests/test_config.py
git commit -m "feat(court-vision): add Pydantic config model with YAML loading"
```

---

## Task 4: Video Ingestion Module

**Files:**
- Create: `court-vision/src/court_vision/ingest.py`
- Create: `court-vision/tests/test_ingest.py`

### Steps

- [ ] **Step 1: Write the failing tests for `test_ingest.py`**

```python
"""Tests for video ingestion — download and frame extraction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from court_vision.ingest import (
    FrameSequence,
    download_video,
    extract_frames,
    is_youtube_url,
)


class TestIsYoutubeUrl:
    def test_standard_url(self):
        assert is_youtube_url("https://www.youtube.com/watch?v=abc123") is True

    def test_short_url(self):
        assert is_youtube_url("https://youtu.be/abc123") is True

    def test_local_file(self):
        assert is_youtube_url("/path/to/match.mp4") is False

    def test_other_url(self):
        assert is_youtube_url("https://example.com/video.mp4") is False


class TestDownloadVideo:
    @patch("court_vision.ingest.subprocess.run")
    def test_download_calls_ytdlp(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(returncode=0)
        output_path = tmp_path / "video.mp4"
        # Create the file as yt-dlp would
        output_path.touch()

        result = download_video(
            "https://www.youtube.com/watch?v=abc123", tmp_path
        )

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "yt-dlp" in call_args
        assert "https://www.youtube.com/watch?v=abc123" in call_args
        assert result == output_path

    @patch("court_vision.ingest.subprocess.run")
    def test_download_raises_on_failure(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        with pytest.raises(RuntimeError, match="yt-dlp download failed"):
            download_video("https://www.youtube.com/watch?v=abc123", tmp_path)


class TestExtractFrames:
    def test_extract_frames_returns_frame_sequence(self, tmp_path: Path):
        """Extract frames from a synthetic video file."""
        video_path = _create_test_video(tmp_path / "test.mp4", num_frames=10)

        result = extract_frames(video_path, target_resolution=(1280, 720))

        assert isinstance(result, FrameSequence)
        assert result.fps > 0
        assert result.total_frames == 10
        assert result.resolution == (1280, 720)
        assert (result.frames_dir / "frame_000000.jpg").exists()
        assert (result.frames_dir / "frame_000009.jpg").exists()

    def test_extract_frames_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            extract_frames(Path("/nonexistent/video.mp4"))


def _create_test_video(path: Path, num_frames: int = 10, fps: int = 30) -> Path:
    """Create a minimal synthetic video for testing."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for _ in range(num_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_ingest.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'court_vision.ingest'`

- [ ] **Step 3: Write `ingest.py`**

```python
"""Video ingestion — YouTube download and frame extraction."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameSequence:
    """Container for extracted video frames with metadata."""

    frames_dir: Path
    fps: float
    total_frames: int
    resolution: tuple[int, int]

    def frame_path(self, index: int) -> Path:
        """Get the path to a specific frame image."""
        return self.frames_dir / f"frame_{index:06d}.jpg"


def is_youtube_url(source: str) -> bool:
    """Check if a source string is a YouTube URL."""
    return "youtube.com/watch" in source or "youtu.be/" in source


def download_video(url: str, output_dir: Path) -> Path:
    """Download a YouTube video using yt-dlp.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded video.

    Returns:
        Path to the downloaded video file.

    Raises:
        RuntimeError: If yt-dlp download fails.
    """
    output_path = output_dir / "video.mp4"
    result = subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", str(output_path),
            url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr}")

    return output_path


def extract_frames(
    video_path: Path,
    target_resolution: tuple[int, int] = (1280, 720),
    output_dir: Path | None = None,
) -> FrameSequence:
    """Extract frames from a video file, resizing to target resolution.

    Args:
        video_path: Path to the input video file.
        target_resolution: (width, height) to resize frames to.
        output_dir: Directory to write frame images. If None, uses
                    a 'frames' subdirectory next to the video.

    Returns:
        FrameSequence with metadata and paths to extracted frames.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = target_resolution

    if output_dir is None:
        output_dir = video_path.parent / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height))
        frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(frame_path), resized)
        frame_count += 1

    cap.release()

    return FrameSequence(
        frames_dir=output_dir,
        fps=fps,
        total_frames=frame_count,
        resolution=target_resolution,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_ingest.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/ingest.py court-vision/tests/test_ingest.py
git commit -m "feat(court-vision): add video ingestion with yt-dlp download and frame extraction"
```

---

## Task 5: Scene Filter Module

**Files:**
- Create: `court-vision/src/court_vision/scene_filter.py`
- Create: `court-vision/tests/test_scene_filter.py`

### Steps

- [ ] **Step 1: Write the failing tests for `test_scene_filter.py`**

```python
"""Tests for scene filtering — gameplay detection from frames."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from court_vision.scene_filter import (
    SceneCategory,
    SceneFilterResult,
    GameplaySegment,
    classify_frame,
    classify_frames,
    filter_gameplay_segments,
    load_scene_model,
)


class TestSceneCategory:
    def test_gameplay_is_a_category(self):
        assert SceneCategory.GAMEPLAY.value == "gameplay"

    def test_all_categories_exist(self):
        categories = {c.value for c in SceneCategory}
        assert categories == {"gameplay", "close_up", "replay", "crowd", "transition"}


class TestClassifyFrame:
    def test_returns_scene_filter_result(self):
        """classify_frame returns category + confidence with correct frame index."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        model = MagicMock()
        # Mock model to return logits for 5 classes
        model.return_value = torch.tensor([[2.0, 0.1, 0.1, 0.1, 0.1]])

        result = classify_frame(frame, model, device=torch.device("cpu"), frame_index=42)

        assert isinstance(result, SceneFilterResult)
        assert isinstance(result.category, SceneCategory)
        assert 0.0 <= result.confidence <= 1.0
        assert result.frame_index == 42


class TestClassifyFrames:
    def test_classifies_all_frames_in_directory(self, tmp_path: Path):
        """classify_frames reads frame images and returns results for each."""
        import cv2

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        # Create 3 synthetic frame images
        for i in range(3):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), frame)

        model = MagicMock()
        model.return_value = torch.tensor([[2.0, 0.1, 0.1, 0.1, 0.1]])

        results = classify_frames(frames_dir, total_frames=3, model=model, device=torch.device("cpu"))

        assert len(results) == 3
        assert results[0].frame_index == 0
        assert results[1].frame_index == 1
        assert results[2].frame_index == 2


class TestFilterGameplaySegments:
    def test_contiguous_gameplay_frames(self):
        """Groups contiguous gameplay frames into segments."""
        results = [
            SceneFilterResult(frame_index=0, category=SceneCategory.GAMEPLAY, confidence=0.9),
            SceneFilterResult(frame_index=1, category=SceneCategory.GAMEPLAY, confidence=0.85),
            SceneFilterResult(frame_index=2, category=SceneCategory.GAMEPLAY, confidence=0.88),
            SceneFilterResult(frame_index=3, category=SceneCategory.CLOSE_UP, confidence=0.95),
            SceneFilterResult(frame_index=4, category=SceneCategory.GAMEPLAY, confidence=0.91),
        ]

        segments = filter_gameplay_segments(results, fps=30.0)

        assert len(segments) == 2
        assert segments[0].start_frame == 0
        assert segments[0].end_frame == 2
        assert segments[1].start_frame == 4
        assert segments[1].end_frame == 4

    def test_no_gameplay_returns_empty(self):
        """Returns empty list when no gameplay frames detected."""
        results = [
            SceneFilterResult(frame_index=0, category=SceneCategory.CLOSE_UP, confidence=0.9),
            SceneFilterResult(frame_index=1, category=SceneCategory.CROWD, confidence=0.8),
        ]

        segments = filter_gameplay_segments(results, fps=30.0)

        assert segments == []

    def test_segment_has_time_range(self):
        """Segments include start/end times in seconds."""
        results = [
            SceneFilterResult(frame_index=30, category=SceneCategory.GAMEPLAY, confidence=0.9),
            SceneFilterResult(frame_index=31, category=SceneCategory.GAMEPLAY, confidence=0.85),
        ]

        segments = filter_gameplay_segments(results, fps=30.0)

        assert len(segments) == 1
        assert segments[0].start_time_s == pytest.approx(1.0)
        assert segments[0].end_time_s == pytest.approx(31 / 30.0)


class TestLoadSceneModel:
    @patch("court_vision.scene_filter.models.resnet18")
    def test_loads_resnet18_with_modified_fc(self, mock_resnet18: MagicMock):
        """Loads ResNet-18 and replaces final FC layer for 5 classes."""
        import torch.nn as nn

        mock_model = MagicMock()
        mock_model.fc = MagicMock(in_features=512)
        mock_resnet18.return_value = mock_model

        model = load_scene_model(weights_path=None, device=torch.device("cpu"))

        mock_resnet18.assert_called_once()
        # Verify fc layer was replaced with nn.Linear for 5 classes
        assert isinstance(mock_model.fc, nn.Linear)
        assert mock_model.fc.out_features == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_scene_filter.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'court_vision.scene_filter'`

- [ ] **Step 3: Write `scene_filter.py`**

```python
"""Scene filter — classify video frames as gameplay or non-gameplay."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class SceneCategory(Enum):
    GAMEPLAY = "gameplay"
    CLOSE_UP = "close_up"
    REPLAY = "replay"
    CROWD = "crowd"
    TRANSITION = "transition"


# Ordered list matching model output indices
CATEGORY_ORDER = [
    SceneCategory.GAMEPLAY,
    SceneCategory.CLOSE_UP,
    SceneCategory.REPLAY,
    SceneCategory.CROWD,
    SceneCategory.TRANSITION,
]


@dataclass
class SceneFilterResult:
    frame_index: int
    category: SceneCategory
    confidence: float


@dataclass
class GameplaySegment:
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    frame_count: int


# Standard ImageNet normalization applied to input frames
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_scene_model(
    weights_path: Path | None,
    device: torch.device,
) -> nn.Module:
    """Load a ResNet-18 model modified for 5-class scene classification.

    Args:
        weights_path: Path to fine-tuned weights. If None, loads ImageNet
                      pre-trained weights (useful for initial testing before
                      fine-tuning).
        device: Torch device to load model onto.

    Returns:
        The model in eval mode on the specified device.
    """
    num_classes = len(CATEGORY_ORDER)

    if weights_path is not None and weights_path.exists():
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    model.eval()
    return model


def classify_frame(
    frame: np.ndarray,
    model: nn.Module,
    device: torch.device,
    frame_index: int = 0,
) -> SceneFilterResult:
    """Classify a single frame into a scene category.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        model: Loaded scene classification model.
        device: Torch device for inference.
        frame_index: Index of this frame in the video sequence.

    Returns:
        SceneFilterResult with predicted category and confidence.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    confidence, predicted = torch.max(probs, dim=1)
    category = CATEGORY_ORDER[predicted.item()]

    return SceneFilterResult(
        frame_index=frame_index,
        category=category,
        confidence=confidence.item(),
    )


def classify_frames(
    frames_dir: Path,
    total_frames: int,
    model: nn.Module,
    device: torch.device,
) -> list[SceneFilterResult]:
    """Classify all frames in a directory.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        total_frames: Number of frames to process.
        model: Loaded scene classification model.
        device: Torch device for inference.

    Returns:
        List of SceneFilterResult, one per frame.
    """
    results = []
    for i in range(total_frames):
        frame_path = frames_dir / f"frame_{i:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        result = classify_frame(frame, model, device, frame_index=i)
        results.append(result)

    return results


def filter_gameplay_segments(
    results: list[SceneFilterResult],
    fps: float,
) -> list[GameplaySegment]:
    """Group contiguous gameplay frames into segments.

    Args:
        results: Ordered list of per-frame classification results.
        fps: Video frame rate for computing timestamps.

    Returns:
        List of GameplaySegment for contiguous runs of gameplay frames.
    """
    segments: list[GameplaySegment] = []
    current_start: int | None = None
    last_gameplay_frame: int | None = None

    for result in results:
        if result.category == SceneCategory.GAMEPLAY:
            if current_start is None:
                current_start = result.frame_index
            last_gameplay_frame = result.frame_index
        else:
            if current_start is not None:
                segments.append(_make_segment(current_start, last_gameplay_frame, fps))
                current_start = None

    # Close final segment if it ends with gameplay
    if current_start is not None:
        segments.append(_make_segment(current_start, last_gameplay_frame, fps))

    return segments


def _make_segment(start_frame: int, end_frame: int, fps: float) -> GameplaySegment:
    return GameplaySegment(
        start_frame=start_frame,
        end_frame=end_frame,
        start_time_s=start_frame / fps,
        end_time_s=end_frame / fps,
        frame_count=end_frame - start_frame + 1,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_scene_filter.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/scene_filter.py court-vision/tests/test_scene_filter.py
git commit -m "feat(court-vision): add ResNet-18 scene filter for gameplay detection"
```

---

## Task 6: Pipeline Orchestrator

**Files:**
- Create: `court-vision/src/court_vision/pipeline.py`
- Create: `court-vision/tests/test_pipeline.py`

### Steps

- [ ] **Step 1: Write the failing tests for `test_pipeline.py`**

```python
"""Tests for the pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from court_vision.pipeline import PipelineResult, run_pipeline


class TestRunPipeline:
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.classify_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_local_file_pipeline(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ):
        """Pipeline orchestrates ingest → scene filter for a local file."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.ingest import FrameSequence
        from court_vision.scene_filter import GameplaySegment

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        mock_load_config.return_value = PipelineConfig()
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        mock_extract.return_value = FrameSequence(
            frames_dir=tmp_path / "frames",
            fps=30.0,
            total_frames=100,
            resolution=(1280, 720),
        )
        mock_classify.return_value = []
        mock_filter.return_value = [
            GameplaySegment(
                start_frame=0, end_frame=50,
                start_time_s=0.0, end_time_s=1.67,
                frame_count=51,
            )
        ]

        result = run_pipeline(str(video_path), config_path=None)

        assert isinstance(result, PipelineResult)
        assert result.total_frames == 100
        assert len(result.gameplay_segments) == 1
        mock_extract.assert_called_once()
        mock_classify.assert_called_once()

    @patch("court_vision.pipeline.download_video")
    @patch("court_vision.pipeline.extract_frames")
    @patch("court_vision.pipeline.classify_frames")
    @patch("court_vision.pipeline.filter_gameplay_segments")
    @patch("court_vision.pipeline.load_scene_model")
    @patch("court_vision.pipeline.get_device")
    @patch("court_vision.pipeline.load_config")
    def test_youtube_url_triggers_download(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_load_model: MagicMock,
        mock_filter: MagicMock,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_download: MagicMock,
        tmp_path: Path,
    ):
        """YouTube URLs trigger yt-dlp download before frame extraction."""
        import torch

        from court_vision.config import PipelineConfig
        from court_vision.ingest import FrameSequence

        mock_load_config.return_value = PipelineConfig()
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        downloaded = tmp_path / "video.mp4"
        downloaded.touch()
        mock_download.return_value = downloaded
        mock_extract.return_value = FrameSequence(
            frames_dir=tmp_path / "frames",
            fps=30.0,
            total_frames=50,
            resolution=(1280, 720),
        )
        mock_classify.return_value = []
        mock_filter.return_value = []

        result = run_pipeline(
            "https://www.youtube.com/watch?v=abc123",
            config_path=None,
            output_dir=tmp_path,
        )

        mock_download.assert_called_once()
        assert result.total_frames == 50
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'court_vision.pipeline'`

- [ ] **Step 3: Write `pipeline.py`**

```python
"""Pipeline orchestrator — runs all stages in sequence."""

from dataclasses import dataclass
from pathlib import Path

from court_vision.config import PipelineConfig, load_config
from court_vision.device import get_device
from court_vision.ingest import (
    FrameSequence,
    download_video,
    extract_frames,
    is_youtube_url,
)
from court_vision.scene_filter import (
    GameplaySegment,
    classify_frames,
    filter_gameplay_segments,
    load_scene_model,
)


@dataclass
class PipelineResult:
    """Result of a pipeline run (Phase 1: ingest + scene filter)."""

    source: str
    total_frames: int
    fps: float
    gameplay_segments: list[GameplaySegment]
    gameplay_frame_count: int
    frames_dir: Path


def run_pipeline(
    source: str,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    scene_weights_path: Path | None = None,
) -> PipelineResult:
    """Run the Court Vision pipeline on a video source.

    Phase 1 stages: video ingestion → scene filter.

    Args:
        source: YouTube URL or local video file path.
        config_path: Path to court-vision.yaml config. None for defaults.
        output_dir: Directory for pipeline output. None for config default.
        scene_weights_path: Path to fine-tuned scene filter weights.
                            None uses ImageNet pre-trained base.

    Returns:
        PipelineResult with frame data and gameplay segments.
    """
    config = load_config(config_path)
    device = get_device(override=config.device)

    if output_dir is None:
        output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Video Ingestion
    if is_youtube_url(source):
        video_path = download_video(source, output_dir)
    else:
        video_path = Path(source)

    resolution = tuple(config.pipeline.target_resolution)
    frame_seq = extract_frames(video_path, target_resolution=resolution)

    # Stage 2: Scene Filter
    model = load_scene_model(scene_weights_path, device)
    results = classify_frames(frame_seq.frames_dir, frame_seq.total_frames, model, device)
    segments = filter_gameplay_segments(results, frame_seq.fps)

    gameplay_frames = sum(seg.frame_count for seg in segments)

    return PipelineResult(
        source=source,
        total_frames=frame_seq.total_frames,
        fps=frame_seq.fps,
        gameplay_segments=segments,
        gameplay_frame_count=gameplay_frames,
        frames_dir=frame_seq.frames_dir,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_pipeline.py -v
```

Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add court-vision/src/court_vision/pipeline.py court-vision/tests/test_pipeline.py
git commit -m "feat(court-vision): add pipeline orchestrator for ingest + scene filter"
```

---

## Task 7: CLI Entry Point

**Files:**
- Create: `court-vision/src/court_vision/cli.py`
- Create: `court-vision/tests/test_cli.py`

### Steps

- [ ] **Step 1: Write the failing test for `test_cli.py`**

```python
"""Tests for the CLI entry point."""

from typer.testing import CliRunner

from court_vision.cli import app

runner = CliRunner()


def test_version_command():
    """Version command prints the current version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help_shows_commands():
    """Top-level help lists available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "process" in result.output
    assert "version" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'court_vision.cli'`

- [ ] **Step 3: Write `cli.py`**

```python
"""Typer CLI for Court Vision."""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="court-vision", help="Automated shot-by-shot tennis data from broadcast video.")


@app.command()
def process(
    source: str = typer.Argument(help="YouTube URL or path to a local video file."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to court-vision.yaml config file."),
    scene_weights: Optional[Path] = typer.Option(None, "--scene-weights", help="Path to fine-tuned scene filter weights."),
) -> None:
    """Process a tennis match video through the CV pipeline.

    Phase 1: extracts frames and classifies gameplay segments.
    JSON/CSV export will be added in Phase 2.
    """
    from court_vision.pipeline import run_pipeline

    result = run_pipeline(
        source=source,
        config_path=config,
        scene_weights_path=scene_weights,
    )

    typer.echo(f"Processed {result.total_frames} frames at {result.fps:.1f} FPS")
    typer.echo(f"Found {len(result.gameplay_segments)} gameplay segments ({result.gameplay_frame_count} frames)")

    for i, seg in enumerate(result.gameplay_segments, 1):
        typer.echo(f"  Segment {i}: frames {seg.start_frame}-{seg.end_frame} ({seg.start_time_s:.1f}s - {seg.end_time_s:.1f}s)")


@app.command()
def version() -> None:
    """Print the Court Vision version."""
    from court_vision import __version__

    typer.echo(f"court-vision {__version__}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd court-vision && source .venv/bin/activate && pytest tests/test_cli.py -v
```

Expected: Both tests PASS.

- [ ] **Step 5: Verify CLI works interactively**

```bash
court-vision version
```

Expected: `court-vision 0.1.0`

- [ ] **Step 6: Commit**

```bash
git add court-vision/src/court_vision/cli.py court-vision/tests/test_cli.py
git commit -m "feat(court-vision): add Typer CLI with process and version commands"
```

---

## Task 8: Run Full Test Suite & Final Commit

**Files:**
- No new files.

### Steps

- [ ] **Step 1: Run the full test suite**

```bash
cd court-vision && source .venv/bin/activate && pytest -v --tb=short
```

Expected: All tests pass (≥22 tests across test_device, test_config, test_ingest, test_scene_filter, test_pipeline, test_cli).

- [ ] **Step 2: Fix any failures**

If any tests fail, fix the issues and re-run until all pass.

- [ ] **Step 3: Verify CLI end-to-end with `version` command**

```bash
court-vision version
```

Expected: `court-vision 0.1.0`

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A court-vision/
git commit -m "fix(court-vision): address test failures from Phase 1 integration"
```

(Skip this step if no fixes were needed.)
