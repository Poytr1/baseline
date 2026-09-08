# Auto-research harness

`court-vision research …` is the loop we use to make the video pipeline usable:
run a clip, score it against human-corrected labels, look at the frames that
matter, feed corrections back, repeat. Every stage of the pipeline is cached
on disk, so tuning a late-stage knob (contact detection, stroke rules,
outcome inference) costs seconds, while a change to a vision stage re-runs
only that stage and what depends on it.

```
research/clips.yaml            clip registry: video, frames, ground truth, stage labels
research/sweeps/*.yaml         parameter grids
research/feedback/<clip>.jsonl human / Claude corrections (append-only)
runs/cache/<clip>/<stage>-<key>.json   cached stage outputs
runs/experiments/<stamp>_<clip>_<tag>/ one experiment (scorecard, outputs, keyframes, REVIEW.md)
runs/leaderboard.jsonl         one line per experiment
```

## The loop

```bash
cd court-vision
court-vision research clips                       # what example videos exist
court-vision research run houston28s --tag base   # pipeline -> scorecard -> review packet
court-vision research run houston28s --tag gap -s contact_min_gap_s=0.6   # try a knob
court-vision research sweep research/sweeps/contacts.yaml                 # grid over clips
court-vision research leaderboard                 # best experiments so far
```

Every run writes `runs/experiments/<stamp>_<clip>_<tag>/`:

| file | what it is |
|---|---|
| `scorecard.json` | points / winners / shots / strokes / ball / players metrics and a blended `score` |
| `match_data.json` | pipeline output (same schema as ground truth; `metadata.hits` has per-hit debug info) |
| `tracking_data.json` | per-frame ball, player boxes and poses |
| `keyframes/sheet_NN.jpg` | contact sheets: every predicted hit, every missed ground-truth hit, point boundaries |
| `keyframes/shot_f<frame>_zoom.jpg` | 1.6x crop around each predicted contact |
| `REVIEW.md` | what to look at + how to record verdicts |

### Claude Code review

Claude Code reviews an experiment by reading `REVIEW.md` and the contact
sheets (each frame carries the ball trail for the last 0.5 s, player boxes,
pose and the projected court), then writes `review.json` next to it:

```json
{"reviewer": "claude-code",
 "shots": [{"frame": 94, "verdict": "ok"},
           {"frame": 42, "verdict": "false_positive"},
           {"frame": 182, "verdict": "wrong", "stroke": "forehand", "fix_ground_truth": true}],
 "missing_shots": [{"frame": 129, "player": "far_player", "stroke": "slice"}],
 "points": [{"point_number": 1, "winner": "near_player"}],
 "notes": "..."}
```

`court-vision research apply-review <experiment_dir>` folds the entries
flagged `fix_ground_truth` (plus `missing_shots` / `points`) into the clip's
ground truth (a `.bak.json` copy is kept), so labels improve with every
review.

### Human feedback

The same vocabulary is available from the command line and is stored per
clip in `research/feedback/<clip>.jsonl`:

```bash
court-vision research feedback add vienna7s --frame 182 --player near_player --stroke forehand --note "clear forehand"
court-vision research feedback add houston28s --point 2 --winner far_player
court-vision research feedback add houston28s --frame 888 --delete         # not a hit
court-vision research feedback add houston28s --frame 1126 --add --player far_player --stroke forehand
court-vision research feedback apply houston28s                              # rewrite ground truth
```

Facts about a clip that the pipeline cannot know (e.g. a left-handed
player) go in `research/clips.yaml` under `overrides:` and apply to every
experiment on that clip.

## Scorecard

`score` blends, in this order of weight: shot F1 (0.25), point-winner
accuracy (0.20), point segmentation F1 (0.15), forehand/backhand side
accuracy (0.15), ball coverage (0.10), serve F1 (0.05), slice F1 (0.05),
both-players rate (0.05). Shots match within ±0.5 s; points match by frame
overlap (IoU ≥ 0.3). When the clip has a stage-label fixture
(`autotune_labels.json`: court corners, ball centres, player boxes) the card
also reports ball detect rate / localisation error, player IoU and court
reprojection error.

## Stage cache

`STAGE_PARAMS` in `config.py` lists which knobs feed each stage; the cache
key for a stage hashes those knobs, the source files that implement the
stage, and the keys of upstream stages. Force a recompute with
`--force players` etc.

## Cross-validation on public datasets

```bash
court-vision research dataset fetch court --n-images 300      # TennisCourtDetector val subset
court-vision research dataset fetch ball --games game7        # TrackNet tennis clips (+ hit/bounce labels)
court-vision research dataset eval-court --method auto
court-vision research dataset eval-ball --method wasb
```

Both archives are read over HTTP Range requests from their HuggingFace
mirrors, so only the sampled members are downloaded. Court evaluation
projects the 14 canonical court keypoints through the estimated homography
and compares them with the dataset's labels; ball evaluation reports
detection rate within 10/20 px, false positives on ball-less frames and,
because TrackNet labels frame status (flying / hit / bounce), recall and
precision of our hit detector on an independent broadcast source.

## Pipeline knobs worth knowing

| knob | default | effect |
|---|---|---|
| `ball_detection_method` | `wasb` | WASB (HRNet) or TrackNet v2 heatmaps |
| `ball_confidence_threshold` | 0.3 | heatmap peak threshold; outlier rejection cleans up the rest |
| `ball_frame_step` | auto | 3-frame window spacing (2 at 60 fps) |
| `player_model` / `player_imgsz` | yolov8s-pose / 1280 | one YOLO-pose pass gives boxes + 17 keypoints; far court is also cropped and upscaled |
| `contact_method` | trajectory | hits from ball direction change + player proximity + wrist-speed peaks, with an "away from hitter" test that rejects bounces |
| `contact_min_gap_s` | 0.45 | minimum time between hits; same-player hits within 1.5 s collapse |
| `near_player_hand` / `far_player_hand` | auto | handedness (auto = inferred from the serve pose) |
| `slice_drop_ratio` | 0.3 | racket wrist drop (torso lengths) over the look-back that marks a slice |
| `outcome_method` | auto | scoreboard OCR first, then ball landing (in/out/net), else unknown |
| `point_split_gap_s` | 4.0 | gap between hits that starts a new point inside one camera segment |

## Results (2026-09-08)

Scorecards after the rework and the `contacts` sweep (`runs/leaderboard.jsonl`; shot tolerance ±0.5 s):

| clip | points | winner | shots P / R / F1 | FH-BH side | serve P/R | slice P/R | ball coverage |
|---|---|---|---|---|---|---|---|
| houston28s (59.94 fps, 2 points, 13 shots) | 2/2 | 2/2 | 1.00 / 1.00 / 1.00 | 9/9 | 1.0 / 1.0 | 1.0 / 0.5 | 0.87 |
| vienna7s (30 fps, 1 point, 5 shots) | 1/1 | unknown* | 1.00 / 1.00 / 1.00 | 3/3 | – | 0 / 0 | 0.93 |

\* the 7 s clip ends with the ball still in flight after the last hit and the
scoreboard never updates, so the pipeline reports `winner: null` /
`outcome_source: unknown` rather than guessing. The two GT "slice" labels in
vienna7s are forehand-side contacts the wrist-drop rule does not flag.

Starting point before the rework (same GT): houston28s shot F1 0.69, stroke
accuracy 0.30, side 4/7, winners 1/2; vienna7s F1 0.77, side 2/3.

Court detector on 300 TennisCourtDetector validation images
(`court-vision research dataset eval-court`):

| method | success | mean kp error | median | ≤5 px | ≤10 px |
|---|---|---|---|---|---|
| neural (default) | 300/300 | 3.4 px | 2.6 px | 87% | 97% |
| classical Hough fallback | 286/300 | 67 px | 64 px | 7% | 12% |
