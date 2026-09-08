"""Parameter sweeps over one or more clips, ranked by mean scorecard score.

A sweep file is YAML:

    name: contact-gap
    clips: [vienna7s, houston28s]
    params:
      contact_min_gap_s: [0.35, 0.45, 0.6]
      contact_min_turn_deg: [30, 40, 55]
    max_runs: 20        # optional; random subset of the grid
    fixed:              # optional overrides applied to every run
      ball_detection_method: wasb

Thanks to the stage cache, sweeping Stage-5 knobs costs seconds per run.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import yaml

from court_vision.research.experiment import run_experiment


def load_sweep(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def grid_points(params: dict[str, list], max_runs: int | None = None, seed: int = 0) -> list[dict]:
    keys = list(params)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(params[k] for k in keys))]
    if max_runs and len(combos) > max_runs:
        rng = random.Random(seed)
        combos = rng.sample(combos, max_runs)
    return combos


def run_sweep(
    sweep: dict,
    root: Path = Path("runs"),
    registry: Path | None = None,
    clips: list[str] | None = None,
    log=print,
) -> list[dict]:
    clips = clips or sweep.get("clips") or []
    fixed = sweep.get("fixed") or {}
    name = sweep.get("name", "sweep")
    combos = grid_points(sweep.get("params") or {}, sweep.get("max_runs"))
    rows: list[dict] = []
    for i, combo in enumerate(combos, 1):
        overrides = {**fixed, **combo}
        scores = {}
        for clip in clips:
            res = run_experiment(clip, overrides, tag=f"{name}-{i}", root=root, registry=registry, keyframes=False, log=lambda *_: None)
            scores[clip] = res.scorecard
        mean = sum(s.score for s in scores.values()) / max(len(scores), 1)
        row = {
            "run": i, "overrides": overrides, "mean_score": round(mean, 4),
            "per_clip": {c: {"score": round(s.score, 4), "shot_f1": round(s.shot_f1, 3), "outcome": f"{s.outcome_correct}/{s.outcome_total}",
                             "side": f"{s.side_correct}/{s.side_total}", "points": f"{s.points_matched}/{s.gt_points}"}
                         for c, s in scores.items()},
        }
        rows.append(row)
        log(f"[sweep {name}] {i}/{len(combos)} mean={mean:.4f} {overrides}")
    rows.sort(key=lambda r: r["mean_score"], reverse=True)
    out = Path(root) / "sweeps"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rows, indent=2))
    return rows
