# Court Vision

Automated shot-by-shot tennis data extraction from broadcast video.

Pipeline: frame extraction → gameplay segmentation → court homography
(neural keypoints, classical fallback) → ball tracking (WASB / TrackNet) →
players + pose (YOLO-pose, court-aware tracking) → scoreboard OCR → hits,
strokes, points and point winners.

## Setup

```bash
cd court-vision
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
brew install ffmpeg            # preview encoding
```

Model weights (WASB, TrackNet, court keypoints, YOLO-pose) download on first
use into `~/.cache/court-vision/models/`. Scoreboard OCR uses RapidOCR
(bundled ONNX models); `tesseract` is an optional fallback.

## Usage

```bash
# Process a YouTube match or a local file -> match_data.json next to the frames
court-vision process "https://youtube.com/watch?v=abc123"
court-vision process match.mp4 --config court-vision.yaml

# Annotated preview video (court, ball trail, players, pose)
court-vision preview match.mp4 --output annotated.mp4

# Compare a run with human-corrected labels
court-vision evaluate output/match_data.json corrected.json

# Streamlit review UI for correcting shots
court-vision review output/match_data.json --frames output/frames --tracking output/tracking_data.json
```

Every knob lives under `pipeline:` in `court-vision.yaml`; see
`src/court_vision/config.py` for the full list and defaults.

## Auto-research harness

`court-vision research …` runs the pipeline on registered example clips with
a per-stage cache, scores each run against corrected labels, renders the
keyframes that matter (every predicted hit, every missed hit, point
boundaries) into contact sheets for Claude Code / human review, and folds
review verdicts and feedback back into the labels.

```bash
court-vision research run houston28s --tag base
court-vision research run houston28s -s contact_min_gap_s=0.6 -s ball_confidence_threshold=0.25
court-vision research sweep research/sweeps/contacts.yaml
court-vision research leaderboard
court-vision research feedback add vienna7s --frame 182 --stroke forehand
court-vision research dataset fetch court && court-vision research dataset eval-court
court-vision research render runs/experiments/<dir> -o annotated.mp4   # strokes, speeds, winners on video
```

See [docs/research-harness.md](docs/research-harness.md) for the loop,
the scorecard, the review/feedback format and the public-dataset
cross-validation.

## Output format

`match_data.json`: `points[]` with `start_frame`, `end_frame`, `server`,
`winner`, `outcome` (`winner` / `error` by `outcome_player`),
`outcome_source` (`scoreboard` / `trajectory` / `unknown`) and `shots[]`
(`frame`, `player`, `stroke` ∈ forehand / backhand / serve / volley /
overhead / slice, `placement`, `confidence`, `speed_kmh` — average speed
from contact to the first bounce, empty when no bounce is visible). `metadata.hits` keeps the raw
hit-detector evidence for each shot.

## Development

```bash
pytest
```
