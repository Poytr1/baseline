"""Fetch + convert the TennisCourtDetector court-keypoint dataset.

Downloads the public TennisCourtDetector dataset (yastrebksv) — ~8,841 broadcast
frames, each with 14 annotated court keypoints — and converts the annotations
into our per-stage label fixture (scripts.autotune.labels schema) so the court
detector can be benchmarked on independent imagery via scripts.autotune.dataset_eval.

This is the reusable expander behind the auto-tune workflow's Dataset phase. It is
idempotent: an existing archive/extracted images are reused, so re-running only
re-converts annotations.

LICENSE NOTE: the dataset repo states no license (defaults to all-rights-reserved).
Recorded here for transparency — vet before any redistribution / production use.

Keypoint → doubles-corner mapping (verified against the workflow's converted
fixture): index 0 = far_left, 1 = far_right, 2 = near_left, 3 = near_right.
These are the four outermost (doubles) baseline corners.

CLI:
    python -m scripts.autotune.fetch_court_dataset           # full set
    python -m scripts.autotune.fetch_court_dataset --limit 300
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

# Google Drive file id for the full dataset archive (images + annotations).
# Source: github.com/yastrebksv/TennisCourtDetector README.
_DRIVE_ID = "1lhAaeQCmk2y440PmagA0KmIVBIysVMwu"
_DATASET_LICENSE = "Unstated (no LICENSE file in repo; defaults to all-rights-reserved)"

# Verified doubles-corner indices into the 14-keypoint annotation.
_CORNER_IDX = {"far_left": 0, "far_right": 1, "near_left": 2, "near_right": 3}


def _default_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "public_datasets"


def download_archive(dataset_dir: Path, force: bool = False, expected_min: int = 8000) -> None:
    """Download + extract the dataset archive.

    Skips the download only when a (near-)complete extraction already exists:
    at least `expected_min` images on disk. A partial set (e.g. a 60-image
    sample from a prior run) does NOT count as complete and will be completed by
    downloading the full archive. Pass force=True to always re-download.
    """
    images_dir = dataset_dir / "images"
    have = len(list(images_dir.glob("*.png"))) if images_dir.is_dir() else 0
    if not force and have >= expected_min:
        print(f"[fetch] {have} images already extracted at {images_dir} (>= {expected_min}) — skipping download.")
        return
    if have:
        print(f"[fetch] only {have} images on disk (< {expected_min}); downloading full archive to complete the set.")

    import gdown

    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive = dataset_dir / "tennis_court_dataset.zip"
    if not archive.exists():
        print(f"[fetch] downloading dataset archive (multi-GB) from Drive id {_DRIVE_ID} ...")
        gdown.download(id=_DRIVE_ID, output=str(archive), quiet=False)
    else:
        print(f"[fetch] archive already downloaded at {archive}")

    print(f"[fetch] extracting {archive} ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dataset_dir)
    # The archive may nest images under a subfolder; normalize so images_dir holds the .png files.
    if not any(images_dir.glob("*.png")):
        found = list(dataset_dir.rglob("*.png"))
        if found:
            images_dir.mkdir(exist_ok=True)
            for p in found:
                if p.parent != images_dir:
                    p.rename(images_dir / p.name)
    print(f"[fetch] extracted {len(list(images_dir.glob('*.png')))} images to {images_dir}")


def _load_annotations(dataset_dir: Path) -> list[dict]:
    anns: list[dict] = []
    for name in ("data_train.json", "data_val.json"):
        p = dataset_dir / name
        if p.exists():
            anns.extend(json.load(open(p)))
    if not anns:
        raise FileNotFoundError(
            f"No data_train.json / data_val.json found in {dataset_dir}. "
            "The archive may use a different layout — inspect it."
        )
    return anns


def build_fixture(dataset_dir: Path, limit: int | None = None, image_size=(1280, 720)) -> dict:
    """Convert dataset annotations into our court label fixture (dict, ready to JSON-dump).

    Only includes images whose .png is actually present on disk and whose four
    doubles-corner keypoints are valid.
    """
    images_dir = dataset_dir / "images"
    anns = _load_annotations(dataset_dir)

    court: list[dict] = []
    for a in anns:
        img = images_dir / f"{a['id']}.png"
        if not img.exists():
            continue
        kps = a.get("kps") or []
        if len(kps) < 4:
            continue
        try:
            corners = {
                name: [float(kps[idx][0]), float(kps[idx][1])]
                for name, idx in _CORNER_IDX.items()
            }
        except (IndexError, TypeError, ValueError):
            continue
        court.append({
            "frame": len(court),
            "image_path": str(img.resolve()),
            "corners": corners,
        })
        if limit and len(court) >= limit:
            break

    return {
        "frames_dir": str(images_dir.resolve()),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "ball": [],
        "players": [],
        "court": court,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + convert TennisCourtDetector into our court label fixture.")
    ap.add_argument("--dataset-dir", type=Path, default=None, help="Cache dir (default: court-vision/data/public_datasets).")
    ap.add_argument("--limit", type=int, default=None, help="Cap converted images (default: all available).")
    ap.add_argument("--no-download", action="store_true", help="Skip download; only (re)convert from existing files.")
    ap.add_argument("--force", action="store_true", help="Re-download the archive even if images already exist.")
    ap.add_argument("--out", type=Path, default=None, help="Fixture output path (default: <dataset-dir>/dataset_labels.json).")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir or _default_dir()
    if not args.no_download:
        download_archive(dataset_dir, force=args.force)

    fixture = build_fixture(dataset_dir, limit=args.limit)
    out = args.out or (dataset_dir / "dataset_labels.json")
    out.write_text(json.dumps(fixture, indent=2))
    print(f"[fetch] license: {_DATASET_LICENSE}")
    print(f"[fetch] wrote {len(fixture['court'])} court labels -> {out}")


if __name__ == "__main__":
    main()
