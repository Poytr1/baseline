"""Scene filter — classify video frames as gameplay or non-gameplay."""

from collections.abc import Callable
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
    progress_callback: Callable[[int, int], None] | None = None,
    stride: int = 1,
) -> list[SceneFilterResult]:
    """Classify all frames in a directory.

    Args:
        frames_dir: Directory containing frame_NNNNNN.jpg files.
        total_frames: Number of frames to process.
        model: Loaded scene classification model.
        device: Torch device for inference.
        progress_callback: Optional (current, total) callback for progress.
        stride: Classify every Nth frame; intermediate frames inherit the label.

    Returns:
        List of SceneFilterResult, one per frame.
    """
    sampled: dict[int, SceneFilterResult] = {}
    for i in range(0, total_frames, stride):
        frame_path = frames_dir / f"frame_{i:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        result = classify_frame(frame, model, device, frame_index=i)
        sampled[i] = result
        if progress_callback:
            progress_callback(min(i + stride, total_frames), total_frames)

    # Fill all frames by propagating nearest sampled result
    results: list[SceneFilterResult] = []
    last_result: SceneFilterResult | None = None
    for i in range(total_frames):
        if i in sampled:
            last_result = sampled[i]
        if last_result is not None:
            results.append(SceneFilterResult(
                frame_index=i,
                category=last_result.category,
                confidence=last_result.confidence,
            ))
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
