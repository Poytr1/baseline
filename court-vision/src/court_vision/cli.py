"""Typer CLI for Court Vision."""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="court-vision", help="Automated shot-by-shot tennis data from broadcast video.")


@app.command()
def process(
    source: str = typer.Argument(help="YouTube URL or path to a local video file."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to court-vision.yaml config file."),
    scene_weights: Optional[Path] = typer.Option(None, "--scene-weights", help="Path to fine-tuned scene filter weights."),
) -> None:
    """Process a tennis match video through the CV pipeline."""
    from court_vision.pipeline import run_pipeline

    result = run_pipeline(
        source=source,
        config_path=config,
        scene_weights_path=scene_weights,
    )

    typer.echo(f"Processed {result.total_frames} frames at {result.fps:.1f} FPS")
    typer.echo(f"Found {len(result.gameplay_segments)} gameplay segments ({result.gameplay_frame_count} frames)")

    for i, seg in enumerate(result.gameplay_segments, 1):
        typer.echo(f"  Segment {i}: frames {seg.start_frame}-{seg.end_frame} ({seg.start_time_s:.1f}s - {seg.end_time_s:.1f}s)")

    if result.court_detections:
        successful = sum(1 for d in result.court_detections if d.success)
        typer.echo(f"Court detection: {successful}/{len(result.court_detections)} segments with homography")

    if result.tracking_results:
        ball_count = sum(1 for t in result.tracking_results if t.ball is not None)
        player_frames = sum(1 for t in result.tracking_results if len(t.players) > 0)
        typer.echo(f"Tracking: ball detected in {ball_count}/{len(result.tracking_results)} frames, "
                   f"players in {player_frames}/{len(result.tracking_results)} frames")

    if result.match_data:
        total_shots = sum(len(p.shots) for p in result.match_data.points)
        typer.echo(f"Shot classification: {len(result.match_data.points)} points, {total_shots} shots detected")


@app.command()
def preview(
    source: str = typer.Argument(help="YouTube URL or path to a local video file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output video path."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file."),
    scene_weights: Optional[Path] = typer.Option(None, "--scene-weights", help="Scene filter weights."),
) -> None:
    """Generate a preview video with overlay annotations."""
    import os
    import subprocess
    import tempfile

    import cv2
    import numpy as np

    from court_vision.overlay import render_overlay
    from court_vision.pipeline import run_pipeline
    from court_vision.player_detect import FrameTrackingResult

    result = run_pipeline(
        source=source,
        config_path=config,
        scene_weights_path=scene_weights,
    )

    # Build frame_index -> tracking lookup
    tracking_by_frame: dict[int, FrameTrackingResult] = {}
    if result.tracking_results:
        for t in result.tracking_results:
            tracking_by_frame[t.frame_index] = t

    # Extract best court homography (first successful detection)
    court_homography: np.ndarray | None = None
    if result.court_detections:
        for det in result.court_detections:
            if det.success and det.homography is not None:
                court_homography = det.homography
                break

    # Determine output path
    if output is None:
        source_path = Path(source)
        output = source_path.parent / f"{source_path.stem}_preview.mp4"

    # Collect and sort frame files
    frame_files = sorted(result.frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        typer.echo("No frames found to render.", err=True)
        raise typer.Exit(1)

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    h, w = first_frame.shape[:2]

    # Write intermediate video with mp4v codec
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, result.fps, (w, h))

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue

        # Extract frame index from filename (frame_NNNNNN.jpg)
        frame_index = int(frame_file.stem.split("_")[1])

        tracking = tracking_by_frame.get(
            frame_index,
            FrameTrackingResult(frame_index=frame_index, ball=None, players=[], poses=[]),
        )

        overlay_frame = render_overlay(frame, tracking, homography=court_homography)
        writer.write(overlay_frame)

    writer.release()

    # Re-encode with ffmpeg to H.264
    subprocess.run(
        [
            "ffmpeg", "-i", tmp_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-y", str(output),
        ],
        capture_output=True,
    )

    # Clean up intermediate file
    Path(tmp_path).unlink(missing_ok=True)

    typer.echo(f"Preview saved to {output}")


@app.command()
def export(
    match_json: Path = typer.Argument(help="Path to match data JSON file."),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or csv."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path."),
) -> None:
    """Export match data to JSON or CSV format."""
    import json as json_module

    from court_vision.export import export_csv, export_json
    from court_vision.shot_classify import MatchData

    with open(match_json) as f:
        raw = json_module.load(f)

    match = MatchData(
        match_id=raw["match_id"],
        source_url=raw["source_url"],
        metadata=raw.get("metadata", {}),
        points=[],
    )

    if output is None:
        stem = match_json.stem
        output = match_json.parent / f"{stem}_export.{format}"

    if format == "csv":
        export_csv(match, output)
    else:
        export_json(match, output)

    typer.echo(f"Exported to {output}")


@app.command()
def review(
    match_json: Path = typer.Argument(help="Path to match data JSON file."),
    frames_dir: Optional[Path] = typer.Option(None, "--frames", help="Path to extracted frames directory."),
    tracking_json: Optional[Path] = typer.Option(None, "--tracking", help="Path to tracking data JSON."),
) -> None:
    """Launch the Streamlit review UI for a match."""
    import subprocess
    import sys

    if not match_json.exists():
        typer.echo(f"Error: {match_json} not found.", err=True)
        raise typer.Exit(1)

    app_path = Path(__file__).parent / "review_app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--", str(match_json),
    ]

    if frames_dir:
        cmd.extend(["--frames-dir", str(frames_dir)])
    if tracking_json:
        cmd.extend(["--tracking", str(tracking_json)])

    typer.echo(f"Launching review UI for {match_json}...")
    subprocess.run(cmd)


@app.command()
def version() -> None:
    """Print the Court Vision version."""
    from court_vision import __version__

    typer.echo(f"court-vision {__version__}")


if __name__ == "__main__":
    app()
