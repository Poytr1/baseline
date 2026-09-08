"""External-dataset court-keypoint benchmark for auto-tuning.

The local fixture only covers one 7s clip. A public court-keypoint dataset
(e.g. yastrebksv/TennisCourtDetector, converted into our LabelSet format with
per-label absolute image_path) lets us measure how the court detector
GENERALIZES across many unrelated images — the single strongest defense against
overfitting one clip.

This runs the pipeline's own per-image court detector (court_detect.detect_court)
on each labelled image and scores reprojection error + success rate against the
labelled doubles corners. It is standalone: no shot/ball/tracking data needed,
so it works on any image set, not just the local clip.

CLI:
    python -m scripts.autotune.dataset_eval \
        --labels /path/to/dataset_labels.json \
        --limit 200 --out dataset_court_scorecard.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from court_vision.court_detect import COURT_KEYPOINTS, _compute_reprojection_error, detect_court

from scripts.autotune.labels import LabelSet, image_path_for, load_labels

_COURT_PTS = np.array(
    [
        COURT_KEYPOINTS["baseline_near_left_doubles"][:2],
        COURT_KEYPOINTS["baseline_near_right_doubles"][:2],
        COURT_KEYPOINTS["baseline_far_right_doubles"][:2],
        COURT_KEYPOINTS["baseline_far_left_doubles"][:2],
    ],
    dtype=np.float64,
)


@dataclass
class DatasetCourtScorecard:
    images_total: int
    images_detected: int          # detector produced a homography
    success_rate: float           # images_detected / images_total
    reproj_error_px: float        # mean reprojection error over detected images (lower better)
    score: float                  # blended court-generalization score in [0, 1]
    missing_images: int           # labelled entries whose image file was absent

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _px_score(err: float, scale: float = 2.0) -> float:
    if not math.isfinite(err):
        return 0.0
    return 1.0 / (1.0 + err / scale)


def benchmark_court(labels: LabelSet, limit: int | None = None) -> DatasetCourtScorecard:
    court_labels = labels.court[:limit] if limit else labels.court

    total = 0
    detected = 0
    missing = 0
    errors: list[float] = []

    for cl in court_labels:
        ordered = cl.ordered_pixels()
        if ordered is None:
            continue
        img_path = image_path_for(cl, labels.frames_dir)
        frame = cv2.imread(str(img_path))
        if frame is None:
            missing += 1
            continue
        total += 1

        result = detect_court(frame)
        if not result.success or result.homography is None:
            continue
        detected += 1
        pixel_pts = np.array(ordered, dtype=np.float64)
        errors.append(_compute_reprojection_error(pixel_pts, _COURT_PTS, result.homography))

    success_rate = detected / total if total else 0.0
    mean_err = float(np.mean(errors)) if errors else math.inf
    score = 0.5 * success_rate + 0.5 * _px_score(mean_err)

    return DatasetCourtScorecard(
        images_total=total,
        images_detected=detected,
        success_rate=success_rate,
        reproj_error_px=mean_err,
        score=score,
        missing_images=missing,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark court detection on an external labelled image set.")
    ap.add_argument("--labels", type=Path, required=True, help="Dataset label fixture JSON (with image_path per label).")
    ap.add_argument("--limit", type=int, default=None, help="Max images to score (keeps runs bounded).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    labels = load_labels(args.labels)
    card = benchmark_court(labels, limit=args.limit)
    out_json = card.to_json()
    if args.out:
        args.out.write_text(out_json)
    print(out_json)


if __name__ == "__main__":
    main()
