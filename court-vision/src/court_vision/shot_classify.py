"""Shot classification — stroke type, placement zones, and point outcomes."""

from dataclasses import dataclass, field

_SERVICE_LINE_Y = 6.4  # Service line distance from net in meters
_SINGLES_WIDTH = 4.115  # Singles sideline to center


@dataclass
class ShotPlacement:
    """Ball landing placement on court."""

    x: float  # court x in meters
    y: float  # court y in meters
    zone: str  # e.g. "crosscourt_deep", "wide", "t"


@dataclass
class Shot:
    """A single shot within a point."""

    shot_number: int
    frame: int
    time_s: float
    player: str  # "near_player" or "far_player"
    stroke: str  # forehand, backhand, serve, volley, overhead, slice
    placement: ShotPlacement | None
    confidence: float


@dataclass
class Point:
    """A single point in the match."""

    point_number: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    server: str | None  # "near_player" or "far_player"
    shots: list[Shot]
    outcome: str | None  # "winner", "error", "unforced_error"
    outcome_player: str | None
    rally_length: int
    review_status: str = "pending"


@dataclass
class MatchData:
    """Full match data for export."""

    match_id: str
    source_url: str
    metadata: dict
    points: list[Point]


def compute_placement_zone(
    x: float,
    y: float,
    is_serve: bool = False,
    server_side: str | None = None,
    hitter: str | None = None,
) -> str:
    """Compute the placement zone from ball landing court coordinates.

    Args:
        x: Ball x-coordinate in meters (court system).
        y: Ball y-coordinate in meters (court system).
        is_serve: Whether this shot is a serve.
        server_side: "deuce" or "ad" (for serve zone computation).
        hitter: "near_player" or "far_player" (for rally direction).

    Returns:
        Zone string: serve zones ("wide", "body", "t") or
        rally zones ("crosscourt_deep", "down_the_line_short", etc.)
    """
    if is_serve:
        return _compute_serve_zone(x, y, server_side or "deuce")
    return _compute_rally_zone(x, y, hitter or "near_player")


def _compute_serve_zone(x: float, y: float, server_side: str) -> str:
    """Compute serve placement zone within the service box.

    Deuce side: right service box (positive x relative to server).
    Ad side: left service box (negative x relative to server).
    """
    abs_x = abs(x)

    # T zone: near the center service line
    if abs_x < _SINGLES_WIDTH / 3:
        return "t"

    # Wide zone: near the singles sideline
    if abs_x > _SINGLES_WIDTH * 2 / 3:
        return "wide"

    # Body zone: in between
    return "body"


def _compute_rally_zone(x: float, y: float, hitter: str) -> str:
    """Compute rally placement zone (direction + depth).

    Direction is relative to the hitter:
    - Near player hits to positive y (far side): crosscourt = positive x, DTL = negative x
    - Far player hits to negative y (near side): crosscourt = negative x, DTL = positive x
    """
    abs_y = abs(y)
    abs_x = abs(x)

    # Depth: short = inside service line, deep = beyond service line
    depth = "short" if abs_y < _SERVICE_LINE_Y else "deep"

    # Direction based on x position relative to hitter
    if abs_x < _SINGLES_WIDTH / 3:
        direction = "middle"
    elif hitter == "near_player":
        # Near player hits to far side: crosscourt = same sign as x (positive x = right)
        direction = "crosscourt" if x > 0 else "down_the_line"
    else:
        # Far player hits to near side: crosscourt = negative x
        direction = "crosscourt" if x < 0 else "down_the_line"

    return f"{direction}_{depth}"
