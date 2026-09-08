# Court Vision

Automated shot-by-shot tennis data extraction from broadcast video.

## Setup

```bash
cd court-vision
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Process a YouTube match (prints gameplay segments to stdout)
court-vision process "https://youtube.com/watch?v=abc123"

# Process a local video file
court-vision process match.mp4
```

## Preview

Generate an annotated video with court, player, and ball overlays rendered on each frame:

```bash
# From a local video
court-vision preview match.mp4

# Specify output path
court-vision preview match.mp4 --output annotated.mp4

# With a custom config
court-vision preview match.mp4 --config config.yaml
```

This runs the full pipeline, renders overlays on every frame, and encodes the result to H.264 via ffmpeg. The output defaults to `<source>_preview.mp4` in the same directory as the input.

## Evaluate & Tune

The evaluate/tune workflow lets you measure pipeline accuracy against human-corrected ground truth and optimize contact detection parameters — without re-running the expensive neural inference stages.

### Prerequisites

You need two files:

- **`tracking_data.json`** — Cached per-frame tracking data (ball, players, poses) from a pipeline run.
- **Ground truth JSON** — A human-corrected `match_data.json` (e.g. produced via the `review` UI).

### Evaluate

Compare pipeline output against ground truth:

```bash
court-vision evaluate data/match_data.json data/ground_truth.json
```

Reports precision, recall, F1 for contact detection, plus stroke and player attribution accuracy. Use `--tolerance` to adjust the frame-matching window (default 15 frames = 0.5s at 30fps):

```bash
court-vision evaluate data/match_data.json data/ground_truth.json --tolerance 10
```

### Tune

Grid-search over `proximity_threshold` and `min_frames_between_contacts` to find the parameter combination that maximizes F1:

```bash
court-vision tune data/ground_truth.json data/tracking_data.json
```

This re-runs only shot classification (Stage 5) on cached tracking data for each parameter combination, so it completes in seconds. Options:

```bash
# Specify the source video (used for match ID)
court-vision tune data/ground_truth.json data/tracking_data.json --source match.mp4

# Show top 10 results instead of default 5
court-vision tune data/ground_truth.json data/tracking_data.json --top 10
```

Output shows the top parameter combinations ranked by F1, e.g.:

```
Top 5 parameter combinations:

  1. F1=0.80  P=0.85  R=0.75  Stroke=0.67  prox=100  min_frames=10
  2. F1=0.78  P=0.90  R=0.69  Stroke=0.64  prox=75   min_frames=8
  ...
```

## Review UI

The review UI is a Streamlit app for inspecting and correcting pipeline output shot-by-shot.

### Prerequisites

- **`match_data.json`** — Pipeline output from `court-vision process`.
- **Frames directory** — The extracted frames (e.g. `data/frames/`).
- **`tracking_data.json`** (optional) — Per-frame tracking data for ball/player overlays.

### Launch

```bash
court-vision review data/match_data.json --frames data/frames/ --tracking data/tracking_data.json
```

This opens the Streamlit app in your browser. From there you can:

- Browse points and navigate frame-by-frame with the slider
- Edit shot stroke types and player attribution
- Add or delete shots at specific frames
- Approve high-confidence points in bulk
- Jump to low-confidence points for manual review
- Save corrected data back to the JSON file

The `--frames` and `--tracking` options can also be set from within the UI sidebar after launch.

## Development

```bash
pytest
```
