"""Golden fixtures: per-frame JSON in, shots and points out.

Each fixture under tests/fixtures/golden was exported from a validated
experiment (see the harness). They pin the analysis layer's behaviour and
double as the reference a port to another language must reproduce:
feed the same tracking JSON to the Swift implementation, expect the same
shots. Tolerances are loose enough for float rounding, tight enough to
catch a logic change.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from court_vision.config import PipelineSettings
from court_vision.scene_filter import GameplaySegment
from court_vision.serialize import tracking_list_from_json
from court_vision.shot_classify import build_match_data
from court_vision.trajectory import BallDetection, postprocess_trajectory, select_ball_path

FIXTURES = Path(__file__).parent / "fixtures" / "golden"


def _timeline(d):
    if not d:
        return None
    from court_vision.scoreboard import ScoreRow, ScoreSample, ScoreTimeline
    tl = ScoreTimeline(roi=tuple(d["roi"]) if d.get("roi") else None)
    for s in d.get("samples", []):
        tl.samples.append(ScoreSample(frame=int(s["frame"]), rows=[ScoreRow(**r) for r in s["rows"]]))
    return tl


@pytest.mark.parametrize("name", ["vienna7s", "houston28s_point1"])
def test_shots_and_points_match_the_golden_fixture(name):
    fx = json.loads((FIXTURES / f"{name}.json").read_text())
    settings = PipelineSettings(**{k: v for k, v in fx["settings"].items() if k in PipelineSettings.model_fields})
    segments = [GameplaySegment(**s) for s in fx["segments"]]
    homographies = [np.asarray(c["homography"]) if c.get("success") else None for c in fx["court"]]
    tracking = tracking_list_from_json(fx["tracking"])
    match = build_match_data(
        name, segments, tracking, fx["fps"], court_homography=homographies[0], settings=settings,
        scoreboard=_timeline(fx.get("scoreboard")), segment_homographies=homographies,
    )
    exp = fx["expected"]["points"]
    assert len(match.points) == len(exp)
    for got, want in zip(match.points, exp):
        assert got.winner == want["winner"]
        assert got.outcome_source == want["outcome_source"]
        assert len(got.shots) == len(want["shots"]), [(s.frame, s.stroke) for s in got.shots]
        for g, w in zip(got.shots, want["shots"]):
            assert abs(g.frame - w["frame"]) <= 1
            assert g.player == w["player"] and g.stroke == w["stroke"]
            if w["speed_kmh"] is None:
                assert g.speed_kmh is None
            else:
                assert g.speed_kmh == pytest.approx(w["speed_kmh"], abs=1.0)


def test_candidate_path_matches_the_golden_fixture():
    fx = json.loads((FIXTURES / "sideview57s_candidates.json").read_text())
    f0 = fx["first_frame"]
    cands = [[BallDetection(f0 + i, c["x"], c["y"], c["confidence"]) for c in fr] for i, fr in enumerate(fx["candidates"])]
    fps = fx["fps"]
    path = select_ball_path(cands, fps, fx["max_speed_px"] * 30.0 / fps)
    traj = postprocess_trajectory(path, fps, max_speed_px=fx["max_speed_px"], homography=np.asarray(fx["homography"]), frame_shape=(720, 1280))
    real = {d.frame_index: (d.x, d.y) for d in traj.detections if not d.interpolated}
    exp = fx["expected_real_detections"]
    hits = sum(1 for e in exp if e["frame"] in real and abs(real[e["frame"]][0] - e["x"]) <= 2 and abs(real[e["frame"]][1] - e["y"]) <= 2)
    assert hits >= 0.95 * len(exp), f"{hits}/{len(exp)} expected detections reproduced"
    assert len(real) <= 1.05 * len(exp) + 10
