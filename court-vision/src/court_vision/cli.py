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


@app.command()
def version() -> None:
    """Print the Court Vision version."""
    from court_vision import __version__

    typer.echo(f"court-vision {__version__}")


if __name__ == "__main__":
    app()
