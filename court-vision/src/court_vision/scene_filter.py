"""Scene filter types and gameplay segment grouping.

Frame classification itself lives in heuristic_scene_filter.py (court colour +
line signals); this module holds the shared result types and turns per-frame
labels into contiguous gameplay segments.
"""

from dataclasses import dataclass
from enum import Enum


class SceneCategory(Enum):
    GAMEPLAY = "gameplay"
    CLOSE_UP = "close_up"
    REPLAY = "replay"
    CROWD = "crowd"
    TRANSITION = "transition"


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
