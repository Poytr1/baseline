"""Review data — load and save match JSON with full deserialization."""

import json
from dataclasses import asdict
from pathlib import Path

from court_vision.shot_classify import MatchData, Point, Shot, ShotPlacement


def load_match_json(path: Path) -> MatchData:
    """Load a match JSON file and deserialize into MatchData.

    Reconstructs the full object hierarchy: MatchData -> Point -> Shot -> ShotPlacement.

    Args:
        path: Path to match JSON file.

    Returns:
        Fully deserialized MatchData.
    """
    with open(path) as f:
        raw = json.load(f)

    points: list[Point] = []
    for p in raw.get("points", []):
        shots: list[Shot] = []
        for s in p.get("shots", []):
            placement = None
            if s.get("placement") is not None:
                pl = s["placement"]
                placement = ShotPlacement(x=pl["x"], y=pl["y"], zone=pl["zone"])

            shots.append(Shot(
                shot_number=s["shot_number"],
                frame=s["frame"],
                time_s=s["time_s"],
                player=s["player"],
                stroke=s["stroke"],
                placement=placement,
                confidence=s["confidence"],
                speed_kmh=s.get("speed_kmh"),
            ))

        points.append(Point(
            point_number=p["point_number"],
            start_frame=p["start_frame"],
            end_frame=p["end_frame"],
            start_time_s=p["start_time_s"],
            end_time_s=p["end_time_s"],
            server=p.get("server"),
            shots=shots,
            outcome=p.get("outcome"),
            outcome_player=p.get("outcome_player"),
            rally_length=p["rally_length"],
            review_status=p.get("review_status", "pending"),
            winner=p.get("winner"),
            outcome_source=p.get("outcome_source"),
        ))

    return MatchData(
        match_id=raw["match_id"],
        source_url=raw["source_url"],
        metadata=raw.get("metadata", {}),
        points=points,
    )


def save_match_json(match: MatchData, path: Path) -> None:
    """Save MatchData to JSON file.

    Args:
        match: Match data to save.
        path: Output file path.
    """
    data = asdict(match)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
