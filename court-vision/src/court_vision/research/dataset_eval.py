"""Cross-validation of the vision stages on public datasets.

* **court**: run ``detect_court`` on TennisCourtDetector images and measure
  (a) success rate and (b) mean pixel error of the 14 canonical court
  keypoints projected back through the estimated homography, against the
  dataset's labelled keypoints. This is independent of our own clips, so it
  is the guard against over-fitting court detection to one broadcast.
* **ball**: run the ball detector (and optionally the full hit detector) on
  TrackNet clips; measure detection rate within a pixel radius on visible
  frames, false positives on non-visible frames, and hit-frame recall
  against the dataset's ``status == 1`` (hit) labels.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from court_vision.court_detect import detect_court
from court_vision.court_keypoint_net import KEYPOINT_COURT_COORDS


@dataclass
class CourtEval:
    method: str
    images: int = 0
    detected: int = 0
    success_rate: float = 0.0
    mean_kp_error_px: float = math.inf
    median_kp_error_px: float = math.inf
    within_5px: float = 0.0
    within_10px: float = 0.0
    per_image: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("per_image")
        return d


def _project_keypoints(H: np.ndarray) -> list[tuple[float, float]] | None:
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    pts = []
    for cx, cy in KEYPOINT_COURT_COORDS:
        p = H_inv @ np.array([cx, cy, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        pts.append((float(p[0] / p[2]), float(p[1] / p[2])))
    return pts


def evaluate_court(labels_path: Path, method: str = "auto", limit: int | None = None, log=print) -> CourtEval:
    fixture = json.loads(Path(labels_path).read_text())
    ev = CourtEval(method=method)
    errors: list[float] = []
    per_kp_hits5 = per_kp_hits10 = per_kp_total = 0
    for entry in fixture["court"][:limit] if limit else fixture["court"]:
        img = cv2.imread(entry["image_path"])
        if img is None:
            continue
        ev.images += 1
        res = detect_court(img, method=method)
        row = {"image": entry["image_path"], "success": bool(res.success)}
        if res.success and res.homography is not None and entry.get("keypoints"):
            proj = _project_keypoints(res.homography)
            if proj is not None:
                ev.detected += 1
                errs = [math.hypot(px - gx, py - gy) for (px, py), (gx, gy) in zip(proj, entry["keypoints"])]
                # ignore keypoints outside the image in the labels (dataset uses those too)
                errs = [e for e, (gx, gy) in zip(errs, entry["keypoints"]) if 0 <= gx <= 1280 and 0 <= gy <= 720]
                if errs:
                    m = float(np.mean(errs))
                    errors.append(m)
                    per_kp_total += len(errs)
                    per_kp_hits5 += sum(1 for e in errs if e <= 5)
                    per_kp_hits10 += sum(1 for e in errs if e <= 10)
                    row["mean_err"] = round(m, 2)
        ev.per_image.append(row)
        if ev.images % 50 == 0:
            log(f"[court-eval {method}] {ev.images} images, success {ev.detected}, mean err {np.mean(errors) if errors else float('nan'):.2f}px")
    ev.success_rate = ev.detected / ev.images if ev.images else 0.0
    if errors:
        ev.mean_kp_error_px = float(np.mean(errors))
        ev.median_kp_error_px = float(np.median(errors))
    if per_kp_total:
        ev.within_5px = per_kp_hits5 / per_kp_total
        ev.within_10px = per_kp_hits10 / per_kp_total
    return ev


@dataclass
class BallEval:
    method: str
    clips: int = 0
    frames_visible: int = 0
    frames_not_visible: int = 0
    detected_10px: int = 0
    detected_20px: int = 0
    false_positives: int = 0
    detect_rate_10px: float = 0.0
    detect_rate_20px: float = 0.0
    false_positive_rate: float = 0.0
    mean_error_px: float = math.inf
    hit_frames: int = 0
    hits_recalled: int = 0
    hit_recall: float = 0.0
    hits_predicted: int = 0
    hit_precision: float = 0.0
    per_clip: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _link_frames(clip_dir: Path, labels: list[dict]) -> Path:
    """Expose TrackNet ``0000.jpg`` frames under our ``frame_%06d.jpg`` naming
    via symlinks (no copying)."""
    linked = clip_dir / "frames"
    linked.mkdir(exist_ok=True)
    for lab in labels:
        src = clip_dir / lab["file"]
        dst = linked / f"frame_{lab['frame']:06d}.jpg"
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
    return linked


def evaluate_ball(
    manifest_path: Path,
    method: str = "wasb",
    confidence_threshold: float = 0.3,
    max_clips: int | None = None,
    with_hits: bool = True,
    log=print,
) -> BallEval:
    from court_vision.ball_tracker import detect_ball_sequence
    from court_vision.trajectory import finalize_ball_by_frame, postprocess_trajectory
    from court_vision.hit_detect import detect_hits
    from court_vision.player_detect import detect_players_segment
    from court_vision.research.datasets import load_ball_clip
    from court_vision.scene_filter import GameplaySegment

    manifest = json.loads(Path(manifest_path).read_text())
    ev = BallEval(method=method)
    errors: list[float] = []
    fps = 30.0
    for entry in manifest[:max_clips] if max_clips else manifest:
        clip = load_ball_clip(entry)
        frames_dir = _link_frames(clip.frames_dir, clip.labels)
        n = len(clip.labels)
        raw = detect_ball_sequence(frames_dir, 0, n - 1, method=method, confidence_threshold=confidence_threshold)
        seg = GameplaySegment(start_frame=0, end_frame=n - 1, start_time_s=0.0, end_time_s=(n - 1) / fps, frame_count=n)
        traj = postprocess_trajectory(raw, fps)
        ball = finalize_ball_by_frame(traj.detections, seg)
        c_vis = c_nv = d10 = d20 = fp = 0
        for lab in clip.labels:
            b = ball.get(lab["frame"])
            real = b is not None and not b.interpolated
            if lab["visible"]:
                c_vis += 1
                if real:
                    d = math.hypot(b.x - lab["x"], b.y - lab["y"])
                    if d <= 10:
                        d10 += 1
                    if d <= 20:
                        d20 += 1
                        errors.append(d)
            else:
                c_nv += 1
                if real:
                    fp += 1
        row = {"clip": f"{clip.game}/{clip.clip}", "frames": n, "visible": c_vis, "det10": d10, "det20": d20, "fp": fp}
        ev.frames_visible += c_vis
        ev.frames_not_visible += c_nv
        ev.detected_10px += d10
        ev.detected_20px += d20
        ev.false_positives += fp
        if with_hits:
            H = detect_court(cv2.imread(str(frames_dir / "frame_000000.jpg"))).homography
            tracking = detect_players_segment(frames_dir, seg, ball, homography=H, player_detect_stride=2)
            hits = detect_hits(tracking, fps, homography=H)
            gt_hits = [lab["frame"] for lab in clip.labels if lab["status"] == 1]
            tol = 8
            recalled = sum(1 for g in gt_hits if any(abs(h.frame - g) <= tol for h in hits))
            correct = sum(1 for h in hits if any(abs(h.frame - g) <= tol for g in gt_hits))
            ev.hit_frames += len(gt_hits)
            ev.hits_recalled += recalled
            ev.hits_predicted += len(hits)
            row.update({"gt_hits": len(gt_hits), "pred_hits": len(hits), "recalled": recalled, "correct": correct})
            ev.hit_precision = (ev.hit_precision * 0)  # placeholder, recomputed below
            ev.per_clip.append({**row, "_correct": correct})
        else:
            ev.per_clip.append(row)
        ev.clips += 1
        log(f"[ball-eval {method}] {row}")
    if ev.frames_visible:
        ev.detect_rate_10px = ev.detected_10px / ev.frames_visible
        ev.detect_rate_20px = ev.detected_20px / ev.frames_visible
    if ev.frames_not_visible:
        ev.false_positive_rate = ev.false_positives / ev.frames_not_visible
    if errors:
        ev.mean_error_px = float(np.mean(errors))
    if ev.hit_frames:
        ev.hit_recall = ev.hits_recalled / ev.hit_frames
    total_correct = sum(r.get("_correct", 0) for r in ev.per_clip)
    if ev.hits_predicted:
        ev.hit_precision = total_correct / ev.hits_predicted
    for r in ev.per_clip:
        r.pop("_correct", None)
    return ev
