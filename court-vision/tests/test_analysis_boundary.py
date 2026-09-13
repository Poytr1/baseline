"""The analysis layer must import and run without any model or OpenCV
dependency — that is what makes it portable (a Swift/Core ML app reproduces
the per-frame JSON and re-implements exactly these modules)."""

import json
import subprocess
import sys
from pathlib import Path

ANALYSIS_MODULES = [
    "court_vision.types", "court_vision.trajectory", "court_vision.ball_speed", "court_vision.hit_detect",
    "court_vision.shot_classify", "court_vision.serialize", "court_vision.scene_filter", "court_vision.config",
]

SCRIPT = r"""
import sys, types
class _Blocked(types.ModuleType):
    def __getattr__(self, name):
        raise ImportError(f"{self.__name__} is off limits in the analysis layer (attribute {name})")
for name in ("cv2", "torch", "ultralytics", "rapidocr_onnxruntime", "onnxruntime", "PIL"):
    sys.modules[name] = _Blocked(name)
import importlib
for m in %s:
    importlib.import_module(m)
# and a smoke computation through the whole analysis chain
import numpy as np
from court_vision.trajectory import BallDetection, postprocess_trajectory, court_gate, convex_hull, point_in_polygon
from court_vision.types import FrameTrackingResult, PlayerDetection
from court_vision.shot_classify import build_match_data
from court_vision.scene_filter import GameplaySegment
from court_vision.config import PipelineSettings
H = np.array([[0.02, 0.0, -6.4], [0.0, -0.05, 30.0], [0.0, 0.0, 1.0]])
dets = [BallDetection(f, 300.0 + 10.0 * f, 400.0 - 5.0 * f, 0.9) for f in range(40)]
traj = postprocess_trajectory(dets, 30.0, homography=H, frame_shape=(720, 1280))
assert len(traj.detections) == 40
assert point_in_polygon((1.0, 1.0), np.array([(0, 0), (2, 0), (2, 2), (0, 2)], float))
assert len(convex_hull(np.array([(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)], float))) == 4
seg = GameplaySegment(0, 100, 0.0, 3.3, 101)
tracking = [FrameTrackingResult(f, dets[f] if f < 40 else None, [PlayerDetection(f, (250.0, 100.0, 350.0, 500.0), 0.9, (0.0, -12.0), "near_player")], []) for f in range(101)]
match = build_match_data("smoke", [seg], tracking, 30.0, court_homography=H, settings=PipelineSettings(contact_method="proximity"))
print("OK", len(match.points))
"""


def test_analysis_layer_imports_and_runs_without_cv2_or_torch():
    src = Path(__file__).resolve().parents[1] / "src"
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT % json.dumps(ANALYSIS_MODULES)],
        capture_output=True, text=True, env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().startswith("OK")
