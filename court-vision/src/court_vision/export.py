"""Export — JSON and CSV output for match data."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from court_vision.shot_classify import MatchData


def export_json(match: MatchData, output_path: Path) -> None:
    """Export match data to JSON in the spec-defined format.

    Args:
        match: Complete match data.
        output_path: Path to write JSON file.
    """
    data = asdict(match)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def export_csv(match: MatchData, output_path: Path) -> None:
    """Export aggregated match statistics to CSV.

    Produces Sackmann-compatible match-level stats per player.

    Args:
        match: Complete match data.
        output_path: Path to write CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict] = {}
    for player in ("near_player", "far_player"):
        stats[player] = {
            "match_id": match.match_id,
            "player": player,
            "total_points": 0,
            "points_won": 0,
            "winners": 0,
            "errors": 0,
            "aces": 0,
            "forehands": 0,
            "backhands": 0,
            "serves": 0,
            "volleys": 0,
            "total_shots": 0,
        }

    for point in match.points:
        # Count points
        if point.server:
            stats[point.server]["total_points"] += 1
            other = "far_player" if point.server == "near_player" else "near_player"
            stats[other]["total_points"] += 1

        # Count outcome
        if point.outcome == "winner" and point.outcome_player:
            stats[point.outcome_player]["points_won"] += 1
            stats[point.outcome_player]["winners"] += 1
        elif point.outcome == "error" and point.outcome_player:
            other = "far_player" if point.outcome_player == "near_player" else "near_player"
            stats[other]["points_won"] += 1
            stats[point.outcome_player]["errors"] += 1

        # Count shots by type
        for shot in point.shots:
            p = shot.player
            stats[p]["total_shots"] += 1
            if shot.stroke == "forehand":
                stats[p]["forehands"] += 1
            elif shot.stroke == "backhand":
                stats[p]["backhands"] += 1
            elif shot.stroke == "serve":
                stats[p]["serves"] += 1
                # Check for ace (serve that ends the point)
                if shot.shot_number == len(point.shots) and point.outcome == "winner":
                    stats[p]["aces"] += 1
            elif shot.stroke == "volley":
                stats[p]["volleys"] += 1

    rows = list(stats.values())
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
