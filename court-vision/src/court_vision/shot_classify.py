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


from court_vision.player_detect import PoseKeypoints


def classify_stroke(
    pose: PoseKeypoints,
) -> tuple[str, float]:
    """Classify the stroke type from pose keypoints at ball contact.

    Rule-based heuristics using wrist, elbow, and shoulder positions:
    - Serve: both wrists above head
    - Overhead: dominant wrist above head, other at body
    - Forehand: dominant wrist extended to dominant side
    - Backhand: dominant wrist extended to non-dominant side
    - Volley: compact arm position, wrists near shoulders
    - Slice: dominant wrist below elbow with arm extended

    Assumes right-handed player for MVP.

    Args:
        pose: PoseKeypoints from MediaPipe.

    Returns:
        Tuple of (stroke_type, confidence).
    """
    kp = pose.keypoints

    # Extract key positions (with safe defaults)
    nose = kp.get("nose", (400.0, 200.0, 0.0))
    r_wrist = kp.get("right_wrist", (450.0, 350.0, 0.0))
    l_wrist = kp.get("left_wrist", (350.0, 350.0, 0.0))
    r_elbow = kp.get("right_elbow", (440.0, 280.0, 0.0))
    r_shoulder = kp.get("right_shoulder", (420.0, 200.0, 0.0))
    l_shoulder = kp.get("left_shoulder", (380.0, 200.0, 0.0))
    r_hip = kp.get("right_hip", (410.0, 400.0, 0.0))

    nose_y = nose[1]
    r_wrist_x, r_wrist_y = r_wrist[0], r_wrist[1]
    l_wrist_y = l_wrist[1]
    r_elbow_y = r_elbow[1]
    r_shoulder_x = r_shoulder[0]
    l_shoulder_x = l_shoulder[0]
    body_center_x = (r_shoulder_x + l_shoulder_x) / 2

    # Rule 1: Serve — both wrists above nose
    if r_wrist_y < nose_y and l_wrist_y < nose_y:
        return ("serve", 0.8)

    # Rule 2: Overhead — dominant wrist above nose, other below
    if r_wrist_y < nose_y and l_wrist_y > nose_y:
        return ("overhead", 0.7)

    # Rule 3: Volley — wrist close to shoulder height, compact position
    wrist_shoulder_dist = abs(r_wrist_y - r_shoulder[1])
    wrist_body_dist_x = abs(r_wrist_x - body_center_x)
    if wrist_shoulder_dist < 60 and wrist_body_dist_x < 80:
        return ("volley", 0.6)

    # Rule 4: Slice — wrist below elbow with arm extended sideways
    if r_wrist_y > r_elbow_y and wrist_body_dist_x > 80:
        return ("slice", 0.6)

    # Rule 5: Forehand vs Backhand — wrist side relative to body center
    if r_wrist_x > body_center_x + 30:
        return ("forehand", 0.7)
    elif r_wrist_x < body_center_x - 30:
        return ("backhand", 0.7)

    # Default: forehand with low confidence
    return ("forehand", 0.4)
