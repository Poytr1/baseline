"""Export — JSON and CSV output for match data."""

import json
from dataclasses import asdict
from pathlib import Path

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
