# Court Vision — Automated Shot-by-Shot Data Collection Pipeline

## Overview

Court Vision is a Python-based computer vision subproject within the Baseline tennis analytics platform. It automates the extraction of shot-by-shot match data from broadcast video (YouTube), producing structured data that feeds into Baseline's existing analytics pipeline and contributes to an open-source dataset.

## Goals

1. **Build an open-source shot-by-shot tennis dataset** from freely available broadcast video
2. **Integrate with Baseline** — output format compatible with Sackmann data and existing import scripts
3. **Human-in-the-loop** — built-in review UI for correcting ML predictions before data is finalized

## MVP Scope

### In scope
- Process one full match end-to-end from a YouTube URL or local video file
- Shot type classification: forehand, backhand, serve, volley, overhead
- Ball placement: direction (down-the-line, crosscourt, middle) and depth (short, deep)
- Point outcome: winner, error, unforced error
- Confidence scores on all predictions
- Streamlit review UI for human correction
- Export to Sackmann-compatible CSV and aggregated match stats

### Out of scope (future)
- Spin detection (topspin, flat, slice)
- Serve speed estimation
- Player identification by name (MVP tracks "near player" / "far player")
- Rally sequence encoding in Match Charting Project format
- Batch processing of multiple videos
- Multi-user review collaboration

## Architecture

### Approach: Monolithic Sequential Pipeline

A single Python package with sequential stages. Each stage is a module within the package. Chosen over microservices (overkill for MVP) and notebook-first (delays runnable pipeline).

### Project Structure

```
tennisconcrete/
├── app/                        # Existing Next.js app
├── scripts/                    # Existing data sync scripts
├── prisma/                     # Existing schema
├── court-vision/               # CV pipeline subproject
│   ├── src/
│   │   └── court_vision/       # Python package
│   │       ├── __init__.py
│   │       ├── pipeline.py     # Orchestrator — runs all stages in sequence
│   │       ├── scene_filter.py # CNN scene classifier
│   │       ├── court_detect.py # Court line detection + homography
│   │       ├── ball_tracker.py # TrackNet-based ball tracking
│   │       ├── player_detect.py# YOLO player detection + pose estimation
│   │       ├── shot_classify.py# Shot type from pose + ball trajectory
│   │       ├── export.py       # JSON/CSV output in Sackmann-compatible format
│   │       └── device.py       # Device selection (MPS/CUDA/CPU)
│   ├── review_ui/              # Streamlit review app
│   │   └── app.py
│   ├── models/                 # Pre-trained model weights (gitignored)
│   ├── data/                   # Sample videos + test fixtures (gitignored)
│   ├── output/                 # Pipeline output (gitignored)
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
└── ...
```

## Pipeline Stages

### Stage 1: Video Ingestion
- Accept YouTube URL or local video file path
- Download via `yt-dlp` if URL
- Extract frames at native FPS (typically 25-30fps for broadcast)
- Store as frame sequence for downstream processing

### Stage 2: Scene Filter
- Lightweight CNN classifier (ResNet-18 fine-tuned) categorizes each frame: **gameplay** / close-up / replay / crowd / transition
- Court line detection as secondary signal — detected court lines strongly indicate gameplay
- Outputs filtered frame sequence with only gameplay frames + timestamp ranges for each continuous gameplay segment
- Gaps between gameplay segments serve as point boundary candidates

### Stage 3: Court Detection & Homography
- Detect court lines using Hough transform or trained line detector
- Compute homography matrix mapping pixel coordinates → real court coordinates (meters)
- Re-computed per segment (camera may shift between points)
- Output: homography matrix per frame/segment

### Stage 4: Ball Tracking + Player Detection
- **Ball:** TrackNetV2 for frame-by-frame ball position, with interpolation through occlusion gaps
- **Players:** YOLOv8 for bounding boxes, mapped to court coordinates via homography to identify near/far player
- **Pose:** MediaPipe skeleton estimation on each player, feeding shot classification
- Output: ball trajectory + player positions + pose keypoints per frame

### Stage 5: Shot Classification
- **Stroke type** from pose keypoints at ball contact: forehand / backhand / serve / volley / overhead / slice
- **Placement** from ball landing position on court: direction (down-the-line, crosscourt, middle), depth (short, deep)
- **Point outcome** from whether ball lands in/out and opponent reachability: winner / error / unforced error
- Rule-based heuristics for MVP (e.g., dominant hand position relative to body determines FH vs BH). No ML model needed — rules get 80%+ accuracy.
- Output: structured shot-by-shot data per point with confidence scores

### Data Flow

```
YouTube URL / local file
  → yt-dlp download (if URL)
  → Frame extraction (25-30 fps)
  → Scene filter (keep gameplay only)
  → Court homography (pixel → court coords)
  → Ball tracking + Player detection + Pose estimation
  → Shot classification (stroke type + placement + outcome)
  → JSON output with confidence scores
  → Human review (Streamlit UI)
  → Export to Sackmann-compatible CSV
```

## Output Format

### Raw Output (per match)

Full frame-level detail in JSON:

```json
{
  "match_id": "youtube_abc123",
  "source_url": "https://youtube.com/watch?v=abc123",
  "metadata": {
    "tournament": null,
    "players": ["near_player", "far_player"],
    "surface": null,
    "date_processed": "2026-03-25"
  },
  "points": [
    {
      "point_number": 1,
      "start_frame": 1200,
      "end_frame": 1450,
      "start_time_s": 40.0,
      "end_time_s": 48.3,
      "server": "near_player",
      "shots": [
        {
          "shot_number": 1,
          "frame": 1205,
          "time_s": 40.17,
          "player": "near_player",
          "stroke": "serve",
          "placement": {"x": 4.2, "y": 1.1, "zone": "wide"},
          "confidence": 0.92
        },
        {
          "shot_number": 2,
          "frame": 1230,
          "time_s": 41.0,
          "player": "far_player",
          "stroke": "forehand",
          "placement": {"x": -2.1, "y": 8.5, "zone": "crosscourt_deep"},
          "confidence": 0.85
        }
      ],
      "outcome": "winner",
      "outcome_player": "far_player",
      "rally_length": 5,
      "review_status": "pending"
    }
  ]
}
```

### Aggregated Output (for Baseline integration)

CSV/JSON summary mapping to stats Baseline already understands: aces, winners, unforced errors, first serve percentage, etc. Compatible with the existing `Match` model fields in Prisma.

### Integration Path

1. Pipeline outputs raw JSON to `court-vision/output/`
2. Human review UI flags/corrects low-confidence predictions
3. Export script converts reviewed data to:
   - **Point-by-point CSV** — compatible with Sackmann `tennis_pointbypoint` format
   - **Aggregated match stats** — compatible with existing `Match` model fields
4. Existing Node.js import scripts (or a new one) load aggregated stats into PostgreSQL

The CV pipeline does not need to know about Prisma or PostgreSQL — it produces files that the existing Node.js layer consumes.

## Human Review UI

### Tech: Streamlit

Quick to build, Python-native, built-in video/image display. Good enough for a review tool without needing a production-grade frontend.

### Features

- **Video player with overlay** — original frame with ball position, player bounding boxes, and court lines drawn on top
- **Shot-by-shot timeline** — scrub through points, see each detected shot with classification
- **Edit controls** — correct stroke type (dropdown), placement zone, point outcome, or mark false positives
- **Confidence highlighting** — shots below configurable threshold (e.g., 0.7) highlighted for priority review
- **Bulk actions** — approve all high-confidence shots in a point, skip to next low-confidence prediction
- **Status tracking** — each point marked as `pending`, `approved`, or `corrected`

### Review Workflow

1. Run pipeline on a match → raw JSON with confidence scores
2. Open review UI → loads JSON + video
3. UI shows low-confidence predictions first
4. Reviewer corrects or approves each point
5. Save → updated JSON with `review_status` = `approved` / `corrected`
6. Export to Sackmann-compatible format

## Dependencies

| Category | Package | Purpose |
|---|---|---|
| Video | `yt-dlp` | YouTube download |
| Video | `opencv-python` | Frame extraction, image processing |
| ML Framework | `torch` + `torchvision` | Model inference |
| Ball Tracking | TrackNetV2 (custom/published weights) | Ball detection from frames |
| Player Detection | `ultralytics` (YOLOv8) | Player bounding boxes |
| Pose Estimation | `mediapipe` | Player skeleton/keypoints |
| Court Detection | `opencv-python` | Hough lines + homography |
| Scene Filter | `torchvision` (ResNet-18) | Pre-trained, fine-tuned |
| Review UI | `streamlit` | Human review interface |
| Data | `pandas` | Data manipulation and export |
| CLI | `typer` | Command-line interface |
| Testing | `pytest` | Unit and integration tests |

## Hardware & Device Strategy

Primary target: **Apple Silicon (M-series)** via PyTorch MPS backend.

| Component | Backend |
|---|---|
| PyTorch models (TrackNet, ResNet-18) | MPS (Metal Performance Shaders) |
| YOLOv8 | MPS (supported by `ultralytics`) |
| MediaPipe | CPU (lightweight, designed for on-device inference) |
| OpenCV | CPU (image processing) |

Device selection logic:
```python
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

Estimated processing time per match on M-series: ~45-90 minutes.

## CLI Interface

```bash
# Process a YouTube match
court-vision process "https://youtube.com/watch?v=abc123" --output output/match1.json

# Process a local video file
court-vision process match.mp4 --output output/match1.json

# Launch review UI
court-vision review output/match1.json --video match.mp4

# Export reviewed data to Sackmann format
court-vision export output/match1.json --format sackmann --output output/match1.csv
```

## Model Strategy

For MVP, lean on **pre-trained models** — avoid training from scratch:

- **Scene filter:** Fine-tune ResNet-18 on a small labeled dataset (~500 frames per category), labeled from a few YouTube matches
- **Ball tracking:** TrackNetV2 with published weights, fine-tune if accuracy is poor on YouTube quality
- **Player detection:** YOLOv8 out-of-the-box for person detection, no training needed
- **Pose estimation:** MediaPipe out-of-the-box, no training needed
- **Shot classification:** Rule-based heuristics from pose keypoints, no ML model for MVP
