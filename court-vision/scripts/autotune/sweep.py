"""Generalized parameter sweep for auto-tuning.

tune.grid_search only ranks Stage-5 contact params by F1. This generalizes it:
sweep an arbitrary grid of Stage-5 knobs, score each candidate with the unified
scorecard (final F1 + per-stage), and rank by the blended `score`.

Cached mode only (Stage 5) — this is the cheap inner loop the workflow runs
many times. Sweeps that touch Stages 1-4 (court/ball thresholds, model choice)
go through a config file + evaluate_all.py --mode full instead, which is far
slower and driven directly by the workflow rather than this module.

CLI:
    python -m scripts.autotune.sweep \
        --ground-truth data/match_data.json \
        --tracking data/tracking_data.json \
        --labels data/autotune_labels.json \
        --top 10 --out sweep_results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from scripts.autotune.evaluate_all import Scorecard, evaluate_cached
from scripts.autotune.labels import empty_labels, load_labels

DEFAULT_PROXIMITY = [50.0, 75.0, 100.0, 125.0, 150.0, 200.0]
DEFAULT_MIN_FRAMES = [3, 5, 8, 10, 15, 20, 25, 30]


def sweep_cached(
    ground_truth_path: Path,
    tracking_path: Path,
    labels,
    source: str = "data/test_input_video.mp4",
    fps: float = 30.0,
    proximity_values: list[float] | None = None,
    min_frames_values: list[int] | None = None,
) -> list[Scorecard]:
    proximity_values = proximity_values or DEFAULT_PROXIMITY
    min_frames_values = min_frames_values or DEFAULT_MIN_FRAMES

    cards: list[Scorecard] = []
    for prox, minf in itertools.product(proximity_values, min_frames_values):
        cards.append(
            evaluate_cached(
                ground_truth_path, tracking_path, labels,
                source=source, fps=fps,
                proximity_threshold=prox,
                min_frames_between_contacts=minf,
            )
        )
    cards.sort(key=lambda c: (c.score, c.f1, c.precision), reverse=True)
    return cards


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep Stage-5 params, rank by blended scorecard.")
    ap.add_argument("--ground-truth", type=Path, default=Path("data/match_data.json"))
    ap.add_argument("--tracking", type=Path, default=Path("data/tracking_data.json"))
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--source", type=str, default="data/test_input_video.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    labels = load_labels(args.labels) if args.labels and args.labels.exists() else empty_labels()
    cards = sweep_cached(args.ground_truth, args.tracking, labels, source=args.source, fps=args.fps)

    top = cards[: args.top]
    payload = {
        "candidates_evaluated": len(cards),
        "best": {"params": cards[0].params, "score": cards[0].score, "f1": cards[0].f1} if cards else None,
        "top": [
            {
                "params": c.params,
                "score": round(c.score, 4),
                "f1": round(c.f1, 4),
                "precision": round(c.precision, 4),
                "recall": round(c.recall, 4),
                "stroke_accuracy": round(c.stroke_accuracy, 4),
            }
            for c in top
        ],
    }
    out_json = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(out_json)
    print(out_json)


if __name__ == "__main__":
    main()
