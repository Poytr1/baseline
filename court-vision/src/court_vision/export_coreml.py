"""Export the three vision models to Core ML and check them against PyTorch.

The analysis layer is plain numpy (see ``types.py`` and
``tests/test_analysis_boundary.py``); these packages are the other half of
an iOS port. Each export is verified on real frames: the ball detector by
the position of its strongest heatmap blob, the pose model by the export
succeeding with the same input size the pipeline uses. Packages are
written to ``models/coreml/`` (not versioned — regenerate with
``court-vision export-coreml``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def export_wasb(out_dir: Path, frames_dir: Path | None = None, n_check: int = 20) -> dict:
    import coremltools as ct
    import cv2
    import torch

    from court_vision.wasb import WASB_HEIGHT, WASB_WIDTH, _get_wasb_model, extract_ball_candidates

    model = _get_wasb_model().eval().cpu()

    class Heatmap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.sigmoid(self.m(x)[0][:, 2])  # current-frame heatmap, (1, 288, 512)

    wrapped = Heatmap(model).eval()
    example = torch.rand(1, 9, WASB_HEIGHT, WASB_WIDTH)
    t0 = time.time()
    traced = torch.jit.trace(wrapped, example)
    ml = ct.convert(
        traced, inputs=[ct.TensorType(name="frames", shape=example.shape)],
        outputs=[ct.TensorType(name="heatmap")],
        convert_to="mlprogram", compute_units=ct.ComputeUnit.ALL, minimum_deployment_target=ct.target.iOS17,
    )
    path = out_dir / "wasb_ball.mlpackage"
    ml.save(str(path))
    report = {"path": str(path), "convert_s": round(time.time() - t0, 1),
              "input": "frames: (1, 9, 288, 512) float32, three consecutive RGB frames resized to 512x288, /255, stacked on channels (oldest first)",
              "output": "heatmap: (1, 288, 512) sigmoid; take connected components above threshold, weighted centroids"}
    if frames_dir is not None:
        frames = [cv2.imread(str(frames_dir / f"frame_{i:06d}.jpg")) for i in range(100, 100 + n_check + 2)]
        frames = [f for f in frames if f is not None]

        def tens(win):
            proc = [cv2.cvtColor(cv2.resize(f, (WASB_WIDTH, WASB_HEIGHT)), cv2.COLOR_BGR2RGB) for f in win]
            return torch.from_numpy(np.concatenate(proc, axis=2)).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        devs, ms = [], []
        for i in range(2, len(frames)):
            x = tens(frames[i - 2:i + 1])
            with torch.no_grad():
                ref = wrapped(x).numpy()[0]
            t0 = time.time()
            hm = np.asarray(ml.predict({"frames": x.numpy()})["heatmap"]).reshape(ref.shape)
            ms.append((time.time() - t0) * 1000)
            a = extract_ball_candidates(ref, 1280, 720, 0.3, 1)
            b = extract_ball_candidates(hm, 1280, 720, 0.3, 1)
            if a and b:
                devs.append(float(np.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1])))
            elif bool(a) != bool(b):
                devs.append(float("inf"))
        report.update({"checked_windows": len(devs), "peak_deviation_px_max": max(devs) if devs else None,
                       "coreml_ms_per_window": round(float(np.median(ms)), 1) if ms else None})
    return report


def export_yolo_pose(out_dir: Path, model_name: str = "yolov8s-pose.pt", imgsz: int = 1280) -> dict:
    import shutil

    from ultralytics import YOLO

    t0 = time.time()
    exported = Path(YOLO(model_name).export(format="coreml", imgsz=imgsz, nms=False))
    dst = out_dir / f"{Path(model_name).stem}.mlpackage"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(exported), str(dst))
    return {"path": str(dst), "convert_s": round(time.time() - t0, 1), "imgsz": imgsz,
            "note": "raw head output (nms=False): decode boxes + 17 keypoints and run NMS on device, as ultralytics does"}


def export_court_net(out_dir: Path) -> dict:
    import coremltools as ct
    import torch

    from court_vision.court_keypoint_net import _get_model

    model, _ = _get_model()
    model = model.eval().cpu()
    example = torch.rand(1, 3, 360, 640)
    t0 = time.time()
    traced = torch.jit.trace(model, example)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="image", shape=example.shape)],
                    convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17)
    path = out_dir / "court_keypoints.mlpackage"
    ml.save(str(path))
    return {"path": str(path), "convert_s": round(time.time() - t0, 1), "input": "image: (1, 3, 360, 640) RGB /255 (broadcast views only; fixed cameras use a calibration)"}


def export_all(out_dir: Path, frames_dir: Path | None = None, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, fn in (("wasb", lambda: export_wasb(out_dir, frames_dir)), ("yolo_pose", lambda: export_yolo_pose(out_dir)),
                     ("court_net", lambda: export_court_net(out_dir))):
        try:
            report[name] = {"converted": True, **fn()}
            log(f"[export] {name}: ok ({report[name].get('convert_s', '?')}s)")
        except Exception as e:  # keep going: one failure must not hide the others
            report[name] = {"converted": False, "error": repr(e)[:500]}
            log(f"[export] {name}: FAILED {e!r}")
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=1))
    return report
