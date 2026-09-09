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
| `ball_min_run_s` | 0.12 | a ball tracklet shorter than this must link (speed-consistently) to a longer one or is dropped; kills shoes, second balls and crowd blips |
| `player_model` / `player_imgsz` | yolov8s-pose / 1280 | one YOLO-pose pass gives boxes + 17 keypoints; far court is also cropped and upscaled |
| `contact_method` | trajectory | hits from ball direction change + player proximity + wrist-speed peaks, with an "away from hitter" test that rejects bounces |
| `contact_min_gap_s` | 0.6 | minimum time between hits; same-player hits within 1.5 s collapse to the best-evidenced one |
| `close_up_ratio` | 0.55 | a "player" taller than this fraction of the frame is a close-up, never a hitter |
| (built in) | – | players alternate: a missed return between two same-player hits < 4 s apart is filled at the ball's closest approach to the opponent |
| `near_player_hand` / `far_player_hand` | auto | handedness (auto = image side of the racket arm on serves; override per clip in `clips.yaml`) |
| `slice_drop_ratio` | 0.3 | racket wrist drop (torso lengths) over the look-back that marks a slice |
| `outcome_method` | auto | scoreboard OCR first, then ball landing (in/out/net), else unknown |
| `point_split_gap_s` | 4.0 | gap between hits that starts a new point inside one camera segment |

## Results (2026-09-08)

Scorecards after the rework and the `contacts` sweep (`runs/leaderboard.jsonl`; shot tolerance ±0.5 s):

| clip | points | winner | shots P / R / F1 | FH-BH side | serve P/R | slice P/R | ball coverage (rally frames) | shots with speed |
|---|---|---|---|---|---|---|---|---|
| houston28s (59.94 fps, 2 points, 13 shots) | 2/2 | 2/2† | 1.00 / 1.00 / 1.00 | 9/9 | 1.0 / 1.0 | 1.0 / 0.5 | 0.75 | 13/13 |
| vienna7s (30 fps, 1 point, 5 shots) | 1/1 | 1/1* | 1.00 / 1.00 / 1.00 | 3/3 | – | 0 / 0 | 0.92 | 5/5 |

\* the 7 s clip ends before the scoreboard updates; the winner comes from the
landing of the last shot (`outcome_source: trajectory`), which the bounce
detector now finds in the last second of the clip. Before that it was
reported as `unknown` rather than guessed.  
† point 2 of the 28 s clip likewise ends before the score graphic updates and
is decided from the landing. The full 134 s reel shows the graphic going from
"AD Zhang" to "40-40", i.e. the near player (Shelton) won — the original
human label said the far player; it was corrected through `research feedback`,
as was a forehand/backhand label in vienna7s. Two label errors in 18 shots /
3 points is the level of noise reviews have to expect. The two GT "slice" labels in
vienna7s are forehand-side contacts the wrist-drop rule does not flag.

On the unlabelled 134 s reel the same settings give 9 points / 47 shots with a
speed on 38 of them; one far-player hit is timed at the apex of a high ball
over the player instead of the contact a second later (an apex test on the
reversal candidates rejected real contacts too and was dropped).

Starting point before the rework (same GT): houston28s shot F1 0.69, stroke
accuracy 0.30, side 4/7, winners 1/2; vienna7s F1 0.77, side 2/3.

Court detector on 300 TennisCourtDetector validation images
(`court-vision research dataset eval-court`):

| method | success | mean kp error | median | ≤5 px | ≤10 px |
|---|---|---|---|---|---|
| neural (default) | 300/300 | 3.4 px | 2.6 px | 87% | 97% |
| classical Hough fallback | 286/300 | 67 px | 64 px | 7% | 12% |

Ball and hit detector on TrackNet tennis `game7` (Federer–Kyrgios, Laver Cup
2017; 9 clips, 1,881 labelled frames; `court-vision research dataset
eval-ball --method wasb`):

| metric | value |
|---|---|
| ball detected within 10 px / 20 px (visible frames) | 81% / 85% |
| mean localisation error (matched frames) | 3.5 px |
| false positives on ball-less frames | 16% |
| hit-frame recall (TrackNet `status == 1`, ±8 frames) | 37/48 = 0.77 |
| hit precision | 37/40 = 0.93 |

One clip (`game7/Clip7`, 124 visible frames) produced no detections at all
and accounts for most of the missed frames and hits.

### Full highlights reel (no labels)

`court-vision research run houston134s` on the 134 s Shelton–Zhang reel
(8,030 frames, 9 camera segments) takes about 25 minutes cold on an M2 Pro
(ball 2 × WASB ≈ 20 min, players ≈ 5 min, scoreboard OCR ≈ 2 min) and seconds
warm. It yields 10 points / 45 shots with players strictly alternating in
every rally, serves found at the start of 7 of the 8 full points, and winners
for 8 of the 10 points (7 from the score graphic, 1 from the landing; the
two `unknown` points are 1–2 s fragments). Point 2 reproduces the corrected
28 s labels shot for shot. Reviewing its contact sheets is how the close-up
guard, the missed-return fill and the second label error were found — the
packet under `runs/experiments/…houston134s…/REVIEW.md` is the starting
point for labelling the reel.

## Shot speed

`Shot.speed_kmh` is the average speed from contact to the first bounce
(`ball_speed.bounce_speed`): the hitter's feet at contact (court metres from
the player box) to the landing spot found by `estimate_bounce`, divided by
the flight time. Both endpoints are on the ground, so the homography is
accurate there; an airborne ball projected through the ground plane is not
(a ball 3 m up on the far side lands 20–40 m "behind" the baseline), which
is why the flight is not integrated sample by sample and why the earlier
launch-speed and net-crossing estimates were biased by 50–100 %.

The bounce itself is found from that projection error: a ball landing on
the far side has its projected depth run ahead while it rises, fall back
as it descends and run ahead again after the bounce, so the depth rate
jumps up at the bounce; a ball landing on the near side is pulled toward
the net while airborne, so its depth rate drops there. `estimate_bounce`
picks the kink with the sign expected for the landing side, falling back
to a prominent depth minimum, and ignores the 2 m net band (stuck
detections) and anything that would land behind the back fence.

The value is a mean over the flight (drag makes it a few percent under
launch speed, like broadcast graphics). The landing is the *first clear*
kink — a local peak at least half as strong as the strongest one: the
strongest kink is often a winner's second bounce (a 33 km/h forehand),
while the first kink over a fixed threshold fires on noise (a 251 km/h
serve). On the labelled clips every shot gets a speed: serves 105–146 km/h,
groundstrokes 61–125 km/h, the one slice 72 km/h. It is left empty when no
bounce is visible before the next hit. `research render` prints it with
each stroke label and in the rally strip, and draws the two ground points
it is measured between on a top-down court in the bottom-right corner:
the hitter's feet at contact (numbered dot, matching the rally strip) and
the first bounce (cross), joined for the latest shot. The live player
positions are the ringed dots. The airborne ball is deliberately not
drawn on that map — its ground projection is metres off, which is the
whole reason the speed is taken between contact and bounce.

## Trajectory cleaning

Raw detections go through `trajectory.py` before anything uses them:

1. court gate — an image-space polygon: the court widened by 9.5 m each
   side and 6 m toward the camera, plus a *prism* over the far half of the
   court (2.5 m each side, lifted 1.2x the far half's pixel height, so a
   lob over the far court projects 40 m past the baseline on the ground
   but still passes). The prism follows the court's perspective; a plain
   box reaches into the crowd beside the far baseline;
2. runs — a step faster than the speed cap or a gap over ~0.5 s starts a
   new tracklet; a step *across* a gap must also stay within 4x the ball's
   speed on either side of it (the cap averages over the gap, so a blip six
   frames before the ball re-appears looked legal);
3. blips — a run shorter than 1 s that moves less than 3 % of the frame
   diagonal, or stands still for 40 % of its frames, is dropped: a ball
   never hangs still, a spectator's head or a ball on the ground does, and
   so does the detector hopping between heads;
4. linking — tracklets shorter than `ball_min_run_s` survive only by
   linking to a longer one with a bridge no faster than the cap and no
   more than 5x the track's own speed (floor 3 px/frame); a run of fewer
   than three detections may only bridge 0.25 s;
5. gap interpolation (speed-consistent), smoothing over consecutive
   frames, a 1 s stationary filter and weak-edge trimming.

On the 134 s reel this took the short junk tracklets in the final output
from 40 (crowd, ball kids, the bottom of the frame) to 4 isolated
single-frame blips, and fixed three serve contact frames that had been
pulled back to the toss by junk; the rendered trail also refuses to draw
a line across an implausible jump. `ball cov` on the scorecard is
measured on rally frames only (inside predicted points), so dropping a
ball in a player's hand between points no longer costs coverage.
