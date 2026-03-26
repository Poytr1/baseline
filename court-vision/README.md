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

## Development

```bash
pytest
```
