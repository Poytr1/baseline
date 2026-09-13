"""Staged pipeline runner with an on-disk cache.

Each stage's cache key is a hash of (clip, the config knobs that feed the
stage, the source files that implement it, and the keys of its upstream
stages). Changing a Stage-5 knob therefore reuses cached scene / court /
ball / player outputs, while editing e.g. ``player_detect.py`` invalidates
exactly the player stage and everything downstream of it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import court_vision
from court_vision.trajectory import BallDetection, finalize_ball_by_frame, postprocess_trajectory, select_ball_path
from court_vision.config import STAGE_DEPS, STAGE_PARAMS, PipelineConfig
from court_vision.court_detect import CourtDetectionResult
from court_vision.ingest import FrameSequence, extract_frames
from court_vision.pipeline import (
    auto_stride,
    load_fixed_court,
    stage_court,
    stage_scene,
    stage_scoreboard,
    stage_shots,
    whole_clip_segment,
)
from court_vision.player_detect import FrameTrackingResult, detect_players_segment
from court_vision.research.clips import Clip
from court_vision.scene_filter import GameplaySegment
from court_vision.scoreboard import ScoreRow, ScoreSample, ScoreTimeline
from court_vision.serialize import (
    ball_from_dict,
    ball_to_dict,
    court_from_dict,
    court_to_dict,
    segment_from_dict,
    segment_to_dict,
    tracking_list_from_json,
    tracking_list_to_json,
)
from court_vision.shot_classify import MatchData

_PKG = Path(court_vision.__file__).parent
STAGE_SOURCES: dict[str, tuple[str, ...]] = {
    "ingest": ("ingest.py",),
    "scene": ("scene_filter.py", "heuristic_scene_filter.py", "court_detect.py"),
    "court": ("court_detect.py", "court_keypoint_net.py"),
    "ball": ("ball_tracker.py", "wasb.py", "tracknet.py"),
    "ball_post": ("trajectory.py",),
    "players": ("player_detect.py",),
    "scoreboard": ("scoreboard.py",),
    "shots": ("shot_classify.py", "hit_detect.py"),
}


def _source_hash(stage: str) -> str:
    h = hashlib.sha1()
    for name in STAGE_SOURCES.get(stage, ()):
        p = _PKG / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


def stage_params(config: PipelineConfig, stage: str) -> dict:
    p = config.pipeline.model_dump()
    return {k: p[k] for k in STAGE_PARAMS.get(stage, ()) if k in p}


@dataclass
class RunArtifacts:
    clip: Clip
    config: PipelineConfig
    frame_seq: FrameSequence
    segments: list[GameplaySegment]
    court: list[CourtDetectionResult]
    ball_raw: list[list[BallDetection]]  # per segment, before post-processing
    ball: list[dict[int, BallDetection | None]]  # per segment, after post-processing
    tracking: list[FrameTrackingResult]
    scoreboard: ScoreTimeline | None
    match_data: MatchData
    timings: dict[str, float] = field(default_factory=dict)
    cache_keys: dict[str, str] = field(default_factory=dict)
    cache_hits: dict[str, bool] = field(default_factory=dict)


class StageCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, stage: str, clip: str, config: PipelineConfig, upstream: dict[str, str]) -> str:
        payload = {
            "stage": stage,
            "clip": clip,
            "params": stage_params(config, stage),
            "src": _source_hash(stage),
            "deps": {d: upstream[d] for d in STAGE_DEPS.get(stage, ()) if d in upstream},
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def path(self, clip: str, stage: str, key: str) -> Path:
        d = self.root / clip
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{stage}-{key}.json"

    def load(self, clip: str, stage: str, key: str):
        p = self.path(clip, stage, key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def save(self, clip: str, stage: str, key: str, data) -> None:
        self.path(clip, stage, key).write_text(json.dumps(data, default=_json_default))


def _json_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _timeline_to_json(tl: ScoreTimeline | None) -> dict | None:
    if tl is None:
        return None
    return {
        "roi": list(tl.roi) if tl.roi else None,
        "samples": [
            {"frame": s.frame, "rows": [r.__dict__ for r in s.rows]} for s in tl.samples
        ],
    }


def _timeline_from_json(d: dict | None) -> ScoreTimeline | None:
    if not d:
        return None
    tl = ScoreTimeline(roi=tuple(d["roi"]) if d.get("roi") else None)
    for s in d.get("samples", []):
        tl.samples.append(ScoreSample(frame=int(s["frame"]), rows=[ScoreRow(**r) for r in s["rows"]]))
    return tl


def run_clip(
    clip: Clip,
    config: PipelineConfig,
    cache: StageCache | None = None,
    force: set[str] | None = None,
    log=print,
) -> RunArtifacts:
    """Run all stages for a clip, reusing cached stage outputs when possible."""
    force = force or set()
    keys: dict[str, str] = {}
    hits: dict[str, bool] = {}
    timings: dict[str, float] = {}

    def cached(stage: str, compute, to_json, from_json):
        t0 = time.time()
        if cache is None:
            out = compute()
            timings[stage] = time.time() - t0
            hits[stage] = False
            return out
        key = cache.key(stage, clip.name, config, keys)
        keys[stage] = key
        raw = None if stage in force else cache.load(clip.name, stage, key)
        if raw is not None:
            hits[stage] = True
            timings[stage] = time.time() - t0
            return from_json(raw)
        out = compute()
        cache.save(clip.name, stage, key, to_json(out))
        hits[stage] = False
        timings[stage] = time.time() - t0
        return out

    # ── ingest (frames on disk are their own cache) ──
    t0 = time.time()
    frames_dir = clip.frames_dir or (clip.video.parent / "frames")
    frame_seq = extract_frames(clip.video, target_resolution=tuple(config.pipeline.target_resolution), output_dir=frames_dir)
    # clip-level options that change what the stages see are part of the
    # ingest key, so they invalidate everything downstream
    calib_sig = hashlib.sha1(clip.calibration.read_bytes()).hexdigest()[:8] if clip.calibration and clip.calibration.exists() else ""
    keys["ingest"] = hashlib.sha1(
        f"{clip.video}:{frame_seq.total_frames}:{config.pipeline.target_resolution}:{clip.single_segment}:{calib_sig}".encode()
    ).hexdigest()[:12]
    timings["ingest"] = time.time() - t0
    log(f"[ingest] {frame_seq.total_frames} frames @ {frame_seq.fps:.2f} fps ({timings['ingest']:.1f}s)")

    segments = cached(
        "scene",
        lambda: [whole_clip_segment(frame_seq)] if clip.single_segment else stage_scene(frame_seq, config),
        lambda segs: [segment_to_dict(s) for s in segs],
        lambda raw: [segment_from_dict(d) for d in raw],
    )
    log(f"[scene] {len(segments)} segment(s) {'(cached)' if hits['scene'] else ''} ({timings['scene']:.1f}s)")

    res = frame_seq.resolution  # (width, height)
    court = cached(
        "court",
        lambda: (load_fixed_court(clip.calibration, len(segments), (res[1], res[0])) if clip.calibration
                 else stage_court(frame_seq, segments, config)),
        lambda cds: [court_to_dict(c) for c in cds],
        lambda raw: [court_from_dict(d) for d in raw],
    )
    log(f"[court] {sum(1 for c in court if c.success)}/{len(court)} homographies {'(cached)' if hits['court'] else ''} ({timings['court']:.1f}s)")

    multi = config.pipeline.ball_candidates > 1

    def compute_ball_raw():
        from court_vision.ball_tracker import detect_ball_candidates_sequence, detect_ball_sequence
        from court_vision.pipeline import ball_far_roi_for
        p = config.pipeline
        shape = (frame_seq.resolution[1], frame_seq.resolution[0])
        out = []
        for i, seg in enumerate(segments):
            common = dict(
                method=p.ball_detection_method,
                confidence_threshold=p.ball_confidence_threshold,
                frame_step=auto_stride(p.ball_frame_step, frame_seq.fps),
                far_roi=ball_far_roi_for(court, i, shape, p.ball_far_crop),
                detect_stride=auto_stride(p.ball_detect_stride, frame_seq.fps),
                far_gate=p.ball_far_crop_gate,
            )
            if multi:
                out.append(detect_ball_candidates_sequence(
                    frame_seq.frames_dir, seg.start_frame, seg.end_frame, max_candidates=p.ball_candidates, **common,
                ))
            else:
                out.append(detect_ball_sequence(frame_seq.frames_dir, seg.start_frame, seg.end_frame, **common))
        return out

    if multi:  # per segment: one candidate list per frame
        ball_raw = cached(
            "ball",
            compute_ball_raw,
            lambda segs: [[[ball_to_dict(b) for b in fr] for fr in seg] for seg in segs],
            lambda raw: [[[ball_from_dict(b) for b in fr] for fr in seg] for seg in raw],
        )
        n_raw = sum(1 for seg in ball_raw for fr in seg if fr)
        log(f"[ball] {n_raw} frames with candidates (top {config.pipeline.ball_candidates}) {'(cached)' if hits['ball'] else ''} ({timings['ball']:.1f}s)")
    else:
        ball_raw = cached(
            "ball",
            compute_ball_raw,
            lambda segs: [[ball_to_dict(b) for b in seg] for seg in segs],
            lambda raw: [[ball_from_dict(b) for b in seg] for seg in raw],
        )
        n_raw = sum(len(s) for s in ball_raw)
        log(f"[ball] {n_raw} raw detections {'(cached)' if hits['ball'] else ''} ({timings['ball']:.1f}s)")

    # ball post-processing is cheap: never cached, but still keyed for downstream
    keys["ball_post"] = cache.key("ball_post", clip.name, config, keys) if cache else "nocache"
    p = config.pipeline
    ball: list[dict[int, BallDetection | None]] = []
    for i, (seg, raw) in enumerate(zip(segments, ball_raw)):
        H_seg = court[i].homography if (i < len(court) and court[i].success) else None
        if multi:
            raw = select_ball_path(raw, frame_seq.fps, p.ball_max_speed_px * 30.0 / max(frame_seq.fps, 1.0))
        stride = auto_stride(p.ball_detect_stride, frame_seq.fps)
        traj = postprocess_trajectory(
            raw, frame_seq.fps, max_gap_s=p.ball_max_gap_s,
            max_speed_px=p.ball_max_speed_px, smooth_window=p.ball_smooth_window,
            homography=H_seg, frame_shape=(frame_seq.resolution[1], frame_seq.resolution[0]),
            min_run_s=p.ball_min_run_s, sample_stride=stride,
        )
        ball.append(finalize_ball_by_frame(
            traj.detections, seg,
            stationary_std_px=p.ball_stationary_std_px, strong_confidence=p.ball_strong_confidence,
            fps=frame_seq.fps, sample_stride=stride,
        ))

    def compute_players():
        out: list[FrameTrackingResult] = []
        for i, seg in enumerate(segments):
            H = court[i].homography if (i < len(court) and court[i].success) else None
            out.extend(detect_players_segment(
                frame_seq.frames_dir, seg, {},
                homography=H,
                player_detect_stride=auto_stride(p.player_detect_stride, frame_seq.fps),
                model_name=p.player_model, imgsz=p.player_imgsz, conf=p.player_conf,
                far_crop=p.player_far_crop, far_tiles=p.player_far_tiles, max_court_x=p.player_max_court_x, max_court_y=p.player_max_court_y,
            ))
        return out

    players_only = cached("players", compute_players, tracking_list_to_json, tracking_list_from_json)
    log(f"[players] {sum(1 for t in players_only if len(t.players) == 2)}/{len(players_only)} frames with both players {'(cached)' if hits['players'] else ''} ({timings['players']:.1f}s)")

    # merge ball into tracking
    ball_all: dict[int, BallDetection | None] = {}
    for d in ball:
        ball_all.update(d)
    tracking = [
        FrameTrackingResult(frame_index=t.frame_index, ball=ball_all.get(t.frame_index), players=t.players, poses=t.poses)
        for t in players_only
    ]

    scoreboard = None
    if p.outcome_method in ("auto", "scoreboard"):
        def compute_sb():
            try:
                return stage_scoreboard(frame_seq, config)
            except Exception as e:  # best effort
                log(f"[scoreboard] failed: {e}")
                return None
        scoreboard = cached("scoreboard", compute_sb, _timeline_to_json, _timeline_from_json)
        n_valid = sum(1 for s in scoreboard.samples if s.is_valid()) if scoreboard else 0
        log(f"[scoreboard] roi={scoreboard.roi if scoreboard else None} valid={n_valid} {'(cached)' if hits.get('scoreboard') else ''} ({timings.get('scoreboard', 0):.1f}s)")

    t0 = time.time()
    match_data = stage_shots(str(clip.video), frame_seq, segments, court, tracking, config, scoreboard=scoreboard)
    timings["shots"] = time.time() - t0
    keys["shots"] = cache.key("shots", clip.name, config, keys) if cache else "nocache"
    log(f"[shots] {len(match_data.points)} point(s), {sum(len(pt.shots) for pt in match_data.points)} shot(s) ({timings['shots']:.1f}s)")

    return RunArtifacts(
        clip=clip, config=config, frame_seq=frame_seq, segments=segments, court=court,
        ball_raw=ball_raw, ball=ball, tracking=tracking, scoreboard=scoreboard,
        match_data=match_data, timings=timings, cache_keys=keys, cache_hits=hits,
    )
