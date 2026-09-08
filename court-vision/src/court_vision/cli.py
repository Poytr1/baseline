"""Typer CLI for Court Vision."""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="court-vision", help="Automated shot-by-shot tennis data from broadcast video.")


@app.command()
def process(
    source: str = typer.Argument(help="YouTube URL or path to a local video file."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to court-vision.yaml config file."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory for pipeline output."),
) -> None:
    """Process a tennis match video through the CV pipeline."""
    from court_vision.pipeline import run_pipeline

    result = run_pipeline(source=source, config_path=config, output_dir=output_dir)

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

        from court_vision.export import export_json
        match_json_path = Path(result.frames_dir).parent / "match_data.json"
        export_json(result.match_data, match_json_path)
        typer.echo(f"Match data saved to {match_json_path}")


@app.command()
def preview(
    source: str = typer.Argument(help="YouTube URL or path to a local video file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output video path."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file."),
) -> None:
    """Generate a preview video with overlay annotations."""
    import os
    import subprocess
    import tempfile

    import cv2
    import numpy as np

    from court_vision.overlay import compute_strong_ball_frames, render_overlay
    from court_vision.pipeline import run_pipeline
    from court_vision.player_detect import FrameTrackingResult

    result = run_pipeline(source=source, config_path=config)

    # Build frame_index -> tracking lookup
    tracking_by_frame: dict[int, FrameTrackingResult] = {}
    if result.tracking_results:
        for t in result.tracking_results:
            tracking_by_frame[t.frame_index] = t

    strong_ball_frames = compute_strong_ball_frames(result.tracking_results or [])

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

    # Collect gameplay frame indices from segments
    gameplay_frames: set[int] = set()
    for seg in result.gameplay_segments:
        for i in range(seg.start_frame, seg.end_frame + 1):
            gameplay_frames.add(i)

    # Collect and sort frame files, filtered to gameplay only
    frame_files = [
        f for f in sorted(result.frames_dir.glob("frame_*.jpg"))
        if int(f.stem.split("_")[1]) in gameplay_frames
    ]
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

    from court_vision.progress import create_progress

    with create_progress() as render_progress:
        render_task = render_progress.add_task("Rendering preview", total=len(frame_files))
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

            overlay_frame = render_overlay(
                frame, tracking, homography=court_homography,
                strong_ball_frames=strong_ball_frames,
            )
            writer.write(overlay_frame)
            render_progress.advance(render_task)

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
def evaluate(
    match_json: Path = typer.Argument(help="Path to pipeline-produced match data JSON."),
    ground_truth: Path = typer.Argument(help="Path to human-corrected ground truth JSON."),
    frame_tolerance: int = typer.Option(15, "--tolerance", "-t", help="Frame tolerance for shot matching."),
) -> None:
    """Evaluate pipeline output against corrected ground truth."""
    from court_vision.evaluate import match_shots
    from court_vision.review_data import load_match_json

    gt = load_match_json(ground_truth)
    pred = load_match_json(match_json)
    result = match_shots(gt, pred, frame_tolerance=frame_tolerance)

    typer.echo(f"Contact Detection:  P={result.precision:.2f}  R={result.recall:.2f}  F1={result.f1:.2f}")
    typer.echo(f"  TP={result.true_positives}  FP={result.false_positives}  FN={result.false_negatives}")
    typer.echo(f"Stroke Accuracy:    {result.stroke_correct}/{result.stroke_total} = {result.stroke_accuracy:.2f}")
    typer.echo(f"Player Accuracy:    {result.player_correct}/{result.player_total} = {result.player_accuracy:.2f}")


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


research_app = typer.Typer(name="research", help="Auto-research harness: run, sweep, review, feedback.")
app.add_typer(research_app, name="research")


@research_app.command("clips")
def research_clips(registry: Optional[Path] = typer.Option(None, "--registry", help="clips.yaml path.")) -> None:
    """List registered example clips."""
    from court_vision.research.clips import load_clips

    for c in load_clips(registry).values():
        gt = "GT" if c.has_ground_truth else "no GT"
        typer.echo(f"{c.name:14s} {c.video}  [{gt}]  {c.notes}")


@research_app.command("run")
def research_run(
    clip: str = typer.Argument(help="Clip name from research/clips.yaml."),
    set_: list[str] = typer.Option([], "--set", "-s", help="Config override key=value (repeatable)."),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Experiment tag."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Base config YAML."),
    root: Path = typer.Option(Path("runs"), "--root", help="Runs root (cache + experiments)."),
    registry: Optional[Path] = typer.Option(None, "--registry"),
    force: list[str] = typer.Option([], "--force", "-f", help="Recompute these stages even if cached."),
    no_keyframes: bool = typer.Option(False, "--no-keyframes", help="Skip keyframe rendering."),
) -> None:
    """Run one experiment: pipeline (cached per stage) -> scorecard -> review packet."""
    from court_vision.research.experiment import parse_override, run_experiment

    overrides = dict(parse_override(o) for o in set_)
    res = run_experiment(clip, overrides, tag=tag, root=root, registry=registry, force=set(force),
                         keyframes=not no_keyframes, config_path=config)
    typer.echo(f"\nscore={res.scorecard.score:.4f}  ->  {res.directory}/REVIEW.md")


@research_app.command("sweep")
def research_sweep(
    sweep_file: Path = typer.Argument(help="Sweep YAML (see research/sweeps/)."),
    clips: Optional[str] = typer.Option(None, "--clips", help="Comma-separated clip names (overrides file)."),
    root: Path = typer.Option(Path("runs"), "--root"),
    registry: Optional[Path] = typer.Option(None, "--registry"),
    top: int = typer.Option(10, "--top"),
) -> None:
    """Grid/random sweep over config knobs across clips; prints a ranked table."""
    from court_vision.research.sweep import load_sweep, run_sweep

    rows = run_sweep(load_sweep(sweep_file), root=root, registry=registry,
                     clips=clips.split(",") if clips else None)
    typer.echo("")
    for r in rows[:top]:
        typer.echo(f"{r['mean_score']:.4f}  {r['overrides']}  {r['per_clip']}")


@research_app.command("leaderboard")
def research_leaderboard(
    root: Path = typer.Option(Path("runs"), "--root"),
    clip: Optional[str] = typer.Option(None, "--clip"),
    top: int = typer.Option(15, "--top"),
) -> None:
    """Show the best experiments so far."""
    from court_vision.research.experiment import load_leaderboard

    rows = [r for r in load_leaderboard(root) if not clip or r["clip"] == clip]
    rows.sort(key=lambda r: r["score"], reverse=True)
    for r in rows[:top]:
        typer.echo(f"{r['score']:.4f} f1={r['shot_f1']:.2f} win={r['outcome']} pts={r['points']} side={r['side']} "
                   f"{r['clip']:11s} {r.get('tag') or ''} {r['overrides']}  {r['dir']}")


@research_app.command("render")
def research_render(
    experiment_dir: Path = typer.Argument(help="Experiment directory (has tracking_data.json + match_data.json)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output mp4 (default: <experiment>/annotated.mp4)."),
    registry: Optional[Path] = typer.Option(None, "--registry"),
    start: Optional[int] = typer.Option(None, "--start", help="First frame."),
    end: Optional[int] = typer.Option(None, "--end", help="Last frame."),
    all_frames: bool = typer.Option(False, "--all-frames", help="Include non-gameplay frames."),
) -> None:
    """Render an annotated video (court, ball trail, players, strokes, point winners) from a finished experiment."""
    from court_vision.research.render import render_experiment_video

    out = render_experiment_video(experiment_dir, output=output, registry=registry, start_frame=start, end_frame=end,
                                  gameplay_only=not all_frames)
    typer.echo(f"annotated video: {out}")


@research_app.command("apply-review")
def research_apply_review(
    experiment_dir: Path = typer.Argument(help="Experiment directory containing review.json."),
    registry: Optional[Path] = typer.Option(None, "--registry"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write corrected GT here instead of in place."),
) -> None:
    """Fold a review.json (Claude Code or human) into the clip's ground truth."""
    import yaml as _yaml

    from court_vision.research.clips import get_clip
    from court_vision.research.review import apply_review_to_ground_truth, load_review

    cfg = _yaml.safe_load((experiment_dir / "config.yaml").read_text())
    clip = get_clip(cfg["clip"], registry)
    if not clip.has_ground_truth:
        typer.echo("clip has no ground truth to correct; create one from match_data.json first", err=True)
        raise typer.Exit(1)
    path = apply_review_to_ground_truth(load_review(experiment_dir / "review.json"), clip.ground_truth, out_path=out)
    typer.echo(f"ground truth updated: {path}")


feedback_app = typer.Typer(name="feedback", help="Record and apply human feedback.")
research_app.add_typer(feedback_app, name="feedback")


@feedback_app.command("add")
def feedback_add(
    clip: str = typer.Argument(help="Clip name."),
    frame: Optional[int] = typer.Option(None, "--frame", help="Contact frame to correct/add/delete."),
    player: Optional[str] = typer.Option(None, "--player", help="near_player | far_player"),
    stroke: Optional[str] = typer.Option(None, "--stroke", help="forehand|backhand|serve|volley|overhead|slice"),
    add: bool = typer.Option(False, "--add", help="Add a missing shot at --frame."),
    delete: bool = typer.Option(False, "--delete", help="Remove the GT shot nearest --frame."),
    point: Optional[int] = typer.Option(None, "--point", help="Point number for winner/server feedback."),
    winner: Optional[str] = typer.Option(None, "--winner"),
    server: Optional[str] = typer.Option(None, "--server"),
    note: Optional[str] = typer.Option(None, "--note"),
    author: str = typer.Option("human", "--author"),
) -> None:
    """Append one feedback entry for a clip."""
    from court_vision.research.feedback import add_feedback

    entry = {k: v for k, v in dict(frame=frame, player=player, stroke=stroke, add=add or None, delete=delete or None,
                                    point=point, winner=winner, server=server, note=note).items() if v is not None}
    if not entry:
        typer.echo("nothing to record", err=True)
        raise typer.Exit(1)
    typer.echo(f"recorded -> {add_feedback(clip, entry, author=author)}")


@feedback_app.command("apply")
def feedback_apply(
    clip: str = typer.Argument(help="Clip name."),
    registry: Optional[Path] = typer.Option(None, "--registry"),
) -> None:
    """Apply all recorded feedback for a clip to its ground truth."""
    from court_vision.research.feedback import apply_feedback

    out = apply_feedback(clip, registry=registry)
    typer.echo(f"ground truth updated: {out}" if out else "no feedback or no ground truth")


@feedback_app.command("show")
def feedback_show(clip: str = typer.Argument(help="Clip name.")) -> None:
    """Print recorded feedback for a clip."""
    from court_vision.research.feedback import load_feedback

    for e in load_feedback(clip):
        typer.echo(e)


dataset_app = typer.Typer(name="dataset", help="Public datasets for cross-validation (subset fetch + eval).")
research_app.add_typer(dataset_app, name="dataset")


@dataset_app.command("fetch")
def dataset_fetch(
    which: str = typer.Argument(help="court | ball"),
    out_dir: Path = typer.Option(Path("data/public_datasets"), "--out"),
    n_images: int = typer.Option(300, "--n-images", help="court: number of labelled images."),
    games: str = typer.Option("game7", "--games", help="ball: comma-separated TrackNet game folders."),
    max_clips: Optional[int] = typer.Option(None, "--max-clips", help="ball: cap clips per game."),
) -> None:
    """Fetch a labelled subset over HTTP Range (no multi-GB download)."""
    from court_vision.research.datasets import fetch_ball_subset, fetch_court_subset

    if which == "court":
        typer.echo(fetch_court_subset(out_dir, n_images=n_images))
    elif which == "ball":
        typer.echo(fetch_ball_subset(out_dir, games=tuple(games.split(",")), max_clips=max_clips))
    else:
        raise typer.BadParameter("which must be court or ball")


@dataset_app.command("eval-court")
def dataset_eval_court(
    labels: Path = typer.Option(Path("data/public_datasets/court_labels.json"), "--labels"),
    method: str = typer.Option("auto", "--method", help="auto | neural | classical"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write JSON report here."),
) -> None:
    """Court detector accuracy on TennisCourtDetector images (14 keypoints)."""
    import json as _json

    from court_vision.research.dataset_eval import evaluate_court

    ev = evaluate_court(labels, method=method, limit=limit)
    typer.echo(_json.dumps(ev.to_dict(), indent=2))
    if out:
        out.write_text(_json.dumps({**ev.to_dict(), "per_image": ev.per_image}, indent=1))


@dataset_app.command("eval-ball")
def dataset_eval_ball(
    manifest: Path = typer.Option(Path("data/public_datasets/tracknet/ball_clips.json"), "--manifest"),
    method: str = typer.Option("wasb", "--method", help="wasb | tracknet"),
    threshold: float = typer.Option(0.3, "--threshold"),
    max_clips: Optional[int] = typer.Option(None, "--max-clips"),
    no_hits: bool = typer.Option(False, "--no-hits", help="Skip player/hit evaluation (ball only)."),
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Ball detector + hit detector accuracy on TrackNet clips (x/y + hit status labels)."""
    import json as _json

    from court_vision.research.dataset_eval import evaluate_ball

    ev = evaluate_ball(manifest, method=method, confidence_threshold=threshold, max_clips=max_clips, with_hits=not no_hits)
    typer.echo(_json.dumps(ev.to_dict(), indent=2))
    if out:
        out.write_text(_json.dumps(ev.to_dict(), indent=1))


@app.command()
def version() -> None:
    """Print the Court Vision version."""
    from court_vision import __version__

    typer.echo(f"court-vision {__version__}")


if __name__ == "__main__":
    app()
