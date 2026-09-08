"""Diagnose WASB heatmap quality on the 28s video.

Samples frames across the video, runs WASB inference, and reports the
distribution of max heatmap values. This tells us whether mid-rally
losses are due to low-confidence peaks (threshold issue) or
essentially-zero peaks (preprocessing/model issue).
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from court_vision.wasb import WASB_HEIGHT, WASB_WIDTH, _get_wasb_model
from court_vision.device import get_device


def run() -> None:
    frames_dir = Path("output/frames")
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        print("No frames found at output/frames/")
        return

    # Sample ~200 frames evenly across the video
    n = len(frame_files)
    step = max(1, n // 200)
    sampled = frame_files[::step]
    print(f"Video has {n} frames; sampling {len(sampled)} (every {step}th)")

    device = get_device()
    model = _get_wasb_model()

    peaks: list[tuple[int, float, float, float]] = []  # (idx, ch0, ch1, ch2)
    for fpath in sampled:
        idx = int(fpath.stem.split("_")[1])
        # Build 3-frame buffer ending at idx
        buf = []
        for k in (idx - 2, idx - 1, idx):
            p = frames_dir / f"frame_{max(0, k):06d}.jpg"
            frame = cv2.imread(str(p))
            if frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            resized = cv2.resize(frame, (WASB_WIDTH, WASB_HEIGHT))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            buf.append(rgb)
        concat = np.concatenate(buf, axis=2)
        tensor = torch.from_numpy(concat).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
        logits = out[0]  # (1, 3, H, W)
        hms = torch.sigmoid(logits[0]).cpu().numpy()  # (3, H, W)
        peaks.append((idx, float(hms[0].max()), float(hms[1].max()), float(hms[2].max())))

    peaks_ch2 = np.array([p[3] for p in peaks])
    print("\n=== Channel 2 (current frame) peak distribution ===")
    print(f"  count: {len(peaks_ch2)}")
    print(f"  min:   {peaks_ch2.min():.3f}")
    print(f"  mean:  {peaks_ch2.mean():.3f}")
    print(f"  median:{np.median(peaks_ch2):.3f}")
    print(f"  max:   {peaks_ch2.max():.3f}")
    print(f"  std:   {peaks_ch2.std():.3f}")
    bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
    hist, _ = np.histogram(peaks_ch2, bins=bins)
    for i, h in enumerate(hist):
        print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: {h:4d} ({100*h/len(peaks_ch2):5.1f}%)")

    # Also compare all 3 channels on a handful of frames
    print("\n=== Sample frames, all 3 channel peaks ===")
    for idx, c0, c1, c2 in peaks[::max(1, len(peaks) // 12)][:12]:
        print(f"  frame {idx:5d}: ch0={c0:.3f}  ch1={c1:.3f}  ch2={c2:.3f}")


if __name__ == "__main__":
    run()
