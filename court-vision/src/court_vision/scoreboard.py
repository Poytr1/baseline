"""Broadcast scoreboard reading (OCR) for point outcomes and boundaries.

The on-screen score graphic is the most reliable signal for *who won a point*:
compare the score before the point with the score after it. This module

1. locates the scoreboard as a temporally static, high-contrast rectangle in
   the lower part of the frame,
2. OCRs its two rows with tesseract (called as a subprocess; no Python
   binding needed), and
3. parses each row into (name, game counts, point score, serving flag).

Everything is best-effort: any failure yields an empty timeline and the
pipeline falls back to trajectory-based outcome inference.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_POINT_TOKENS = {"0", "15", "30", "40", "AD", "A", "ADV"}
_POINT_RANK = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4, "ADV": 4}


@dataclass
class ScoreRow:
    name: str
    games: list[int]
    points: str  # "0","15","30","40","AD", tie-break digits, or "" when unknown
    serving: bool = False
    raw: str = ""

    def point_rank(self) -> int | None:
        if self.points in _POINT_RANK:
            return _POINT_RANK[self.points]
        if self.points.isdigit():  # tie-break
            return 10 + int(self.points)
        return None


@dataclass
class ScoreSample:
    frame: int
    rows: list[ScoreRow]

    def is_valid(self) -> bool:
        """Two rows, each with at least a game count or a point score."""
        return len(self.rows) == 2 and all((r.games or r.points) for r in self.rows)

    def has_points(self) -> bool:
        return len(self.rows) == 2 and all(r.point_rank() is not None for r in self.rows)


@dataclass
class ScoreTimeline:
    samples: list[ScoreSample] = field(default_factory=list)
    roi: tuple[int, int, int, int] | None = None

    def sample_at(self, frame: int, direction: str = "before") -> ScoreSample | None:
        """Nearest *valid* sample at/before (``direction='before'``) or
        at/after (``'after'``) ``frame``."""
        valid = [s for s in self.samples if s.is_valid()]
        if direction == "before":
            cands = [s for s in valid if s.frame <= frame]
            return cands[-1] if cands else None
        cands = [s for s in valid if s.frame >= frame]
        return cands[0] if cands else None

    def stable_state(self, start: int, end: int) -> ScoreSample | None:
        """Most common valid state in [start, end] (majority vote on the score
        tuple), or None."""
        valid = [s for s in self.samples if s.is_valid() and start <= s.frame <= end]
        if not valid:
            return None
        counts: dict[tuple, list[ScoreSample]] = {}
        for s in valid:
            key = tuple((tuple(r.games), r.points) for r in s.rows)
            counts.setdefault(key, []).append(s)
        best = max(counts.values(), key=len)
        return best[0]


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


@lru_cache(maxsize=1)
def _get_rapidocr():
    """RapidOCR (ONNX PaddleOCR) engine, or None if not installed."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return None
    try:
        return RapidOCR()
    except Exception:
        return None


def ocr_available() -> bool:
    return _get_rapidocr() is not None or tesseract_available()


@dataclass
class TextBox:
    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    conf: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def h(self) -> float:
        return self.y2 - self.y1


def ocr_text_boxes(img: np.ndarray, scale: float = 2.0) -> list[TextBox]:
    """Run RapidOCR on an image; boxes are returned in the *input* image's
    pixel coordinates."""
    engine = _get_rapidocr()
    if engine is None or img.size == 0:
        return []
    src = img
    if scale != 1.0:
        src = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    try:
        result, _ = engine(src)
    except Exception:
        return []
    boxes: list[TextBox] = []
    for quad, text, conf in (result or []):
        xs = [p[0] / scale for p in quad]
        ys = [p[1] / scale for p in quad]
        boxes.append(TextBox(min(xs), min(ys), max(xs), max(ys), str(text).strip(), float(conf)))
    return boxes


def _cluster_rows(boxes: list[TextBox]) -> list[list[TextBox]]:
    """Group boxes into rows by vertical overlap; rows top-to-bottom, boxes left-to-right."""
    rows: list[list[TextBox]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        for row in rows:
            ry = sum(x.cy for x in row) / len(row)
            rh = sum(x.h for x in row) / len(row)
            if abs(b.cy - ry) < 0.6 * max(rh, b.h):
                row.append(b)
                break
        else:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda b: b.x1)
    rows.sort(key=lambda r: sum(b.cy for b in r) / len(r))
    return rows


_ALPHA = re.compile(r"[A-Za-z][A-Za-z.\-']{1,}")
_NUM = re.compile(r"AD|\d+")


def _row_tokens(row: list[TextBox]) -> tuple[str, list[str]]:
    name_parts: list[str] = []
    nums: list[str] = []
    for b in row:
        t = b.text.upper().replace("O", "0") if not _ALPHA.fullmatch(b.text) else b.text
        toks = _NUM.findall(t.upper())
        alpha = _ALPHA.findall(b.text)
        if alpha and not toks:
            name_parts.extend(alpha)
        elif toks:
            nums.extend(toks)
    return " ".join(name_parts), nums


def _parse_numbers(text: str) -> tuple[list[int], str]:
    toks = _NUM.findall(text.upper().replace("O", "0"))
    return _parse_tokens(toks)


def _parse_tokens(toks: list[str]) -> tuple[list[int], str]:
    if not toks:
        return [], ""
    last = toks[-1]
    if last in _POINT_TOKENS or (last.isdigit() and int(last) > 7):
        games = [int(t) for t in toks[:-1] if t.isdigit()]
        return games, last
    return [int(t) for t in toks if t.isdigit()], ""


# ── Locate ───────────────────────────────────────────────────────────────────

def locate_scoreboard(frames: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    """Find the score graphic: text that stays put across sampled frames.

    With RapidOCR: OCR the lower 45% of each probe frame, keep boxes whose
    text+position recur in most probes, cluster them into rows, and take
    the block of >=2 adjacent rows that each carry a name and a number.
    Falls back to a pixel-variance heuristic without RapidOCR.
    """
    if len(frames) < 3:
        return None
    h, w = frames[0].shape[:2]
    if _get_rapidocr() is not None:
        y0 = int(h * 0.55)
        per_frame: list[list[TextBox]] = []
        for f in frames:
            boxes = ocr_text_boxes(f[y0:, :], scale=1.5)
            per_frame.append([TextBox(b.x1, b.y1 + y0, b.x2, b.y2 + y0, b.text, b.conf)
                              for b in boxes if b.conf > 0.5])
        # Names stay put across the clip; scores change, so only alphabetic
        # boxes are required to be static.
        static: list[TextBox] = []
        for b in per_frame[0]:
            if not _ALPHA.search(b.text) or _NUM.fullmatch(b.text.upper()):
                continue
            hits = 1
            for other in per_frame[1:]:
                if any(abs(o.cx - b.cx) < 12 and abs(o.cy - b.cy) < 8 and _ALPHA.search(o.text) for o in other):
                    hits += 1
            if hits >= max(2, int(0.6 * len(per_frame))):
                static.append(b)
        rows = _cluster_rows(static)
        if len(rows) < 2:
            return None
        # Two vertically closest name rows form the scoreboard.
        best = None
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            ay = sum(x.cy for x in a) / len(a)
            by = sum(x.cy for x in b) / len(b)
            rh = max(sum(x.h for x in a) / len(a), sum(x.h for x in b) / len(b))
            if abs(by - ay) < 3 * rh and (best is None or abs(by - ay) < best[0]):
                best = (abs(by - ay), a + b, ay, by, rh)
        if best is None:
            return None
        _, name_boxes, ay, by, rh = best
        # Union every box (numbers included, whatever they read) that sits in
        # either row band in any probe frame.
        boxes = list(name_boxes)
        for fb in per_frame:
            for o in fb:
                if min(abs(o.cy - ay), abs(o.cy - by)) < 0.7 * rh:
                    boxes.append(o)
        x1 = int(min(b.x1 for b in boxes)); x2 = int(max(b.x2 for b in boxes))
        y1 = int(min(b.y1 for b in boxes)); y2 = int(max(b.y2 for b in boxes))
        pad_x, pad_y = int(rh * 1.5), int(rh * 0.6)
        return (max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y))

    # Fallback: temporally static, high-contrast rectangle.
    y0 = int(h * 0.55)
    grays = np.stack([cv2.cvtColor(f[y0:, :], cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    std = grays.std(axis=0)
    mean = grays.mean(axis=0)
    static = ((std < 10.0) & ((mean < 100) | (mean > 160))).astype(np.uint8)
    static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    static = cv2.morphologyEx(static, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(static, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < w * 0.12 or bh < h * 0.04 or bh > h * 0.3 or bw / max(bh, 1) < 1.5:
            continue
        if area / float(bw * bh) < 0.35:
            continue
        if best is None or area > best[0]:
            best = (area, x, y + y0, x + bw, y + y0 + bh)
    if best is None:
        return None
    _, x1, y1, x2, y2 = best
    return (max(0, x1 - 4), max(0, y1 - 4), min(w, x2 + 4), min(h, y2 + 4))


# ── OCR ──────────────────────────────────────────────────────────────────────

def _tesseract(img: np.ndarray, psm: int, whitelist: str | None = None) -> str:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "row.png"
        cv2.imwrite(str(p), img)
        cmd = ["tesseract", str(p), "-", "--psm", str(psm)]
        if whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except (subprocess.TimeoutExpired, OSError):
            return ""
        return r.stdout.strip()


def _prep(img: np.ndarray, scale: float = 3.0) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if g.mean() < 128:
        g = 255 - g
    return cv2.GaussianBlur(g, (3, 3), 0)


def _serving_flags(crop: np.ndarray, row_bands: list[tuple[int, int]], text_x0: float) -> list[bool]:
    """Detect the serve indicator (dot/arrow) left of the names.

    Looks at the strip between the crop's left edge and where the name text
    starts; the serving row has a bright/saturated blob there.
    """
    h, w = crop.shape[:2]
    x_end = max(3, min(int(text_x0) - 2, w))
    strip = crop[:, :x_end]
    if strip.size == 0:
        return [False] * len(row_bands)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    bright = ((hsv[:, :, 2] > 140) & ((hsv[:, :, 1] > 90) | (hsv[:, :, 2] > 200))).astype(np.float32)
    vals = []
    for y1, y2 in row_bands:
        band = bright[max(0, y1):min(h, y2)]
        vals.append(float(band.mean()) if band.size else 0.0)
    if not vals:
        return []
    top = max(vals)
    flags = [v == top and top > 0.02 and (top - min(vals)) > 0.02 for v in vals]
    return flags if sum(flags) == 1 else [False] * len(row_bands)


def _normalize_points(rows: list[ScoreRow]) -> list[ScoreRow]:
    """Fill a blank point cell from context: many graphics show only "AD"
    on the leading side (blank = 40), and hide a 0 while the other side has
    points."""
    if len(rows) == 2:
        a, b = rows
        for me, other in ((a, b), (b, a)):
            if me.points == "" and other.points in ("AD", "A", "ADV"):
                me.points = "40"
            elif me.points == "" and other.points in ("15", "30", "40"):
                me.points = "0"
    return rows


def ocr_scoreboard(crop: np.ndarray, n_rows: int = 2) -> list[ScoreRow]:
    """OCR a scoreboard crop into rows (RapidOCR first, tesseract fallback)."""
    rows = _ocr_scoreboard_raw(crop, n_rows)
    return _normalize_points(rows)


def _ocr_scoreboard_raw(crop: np.ndarray, n_rows: int = 2) -> list[ScoreRow]:
    if crop.size == 0:
        return []
    h, w = crop.shape[:2]
    rows: list[ScoreRow] = []
    boxes = ocr_text_boxes(crop, scale=2.0) if _get_rapidocr() is not None else []
    if boxes:
        clusters = [r for r in _cluster_rows(boxes) if _row_tokens(r)[0] or _row_tokens(r)[1]]
        clusters = [r for r in clusters if _row_tokens(r)[0]][:n_rows] or clusters[:n_rows]
        bands = [(int(min(b.y1 for b in r)), int(max(b.y2 for b in r))) for r in clusters]
        text_x0 = min((b.x1 for r in clusters for b in r if _ALPHA.fullmatch(b.text)), default=w * 0.1)
        flags = _serving_flags(crop, bands, text_x0)
        for i, r in enumerate(clusters):
            name, nums = _row_tokens(r)
            games, points = _parse_tokens(nums)
            rows.append(ScoreRow(name=name, games=games, points=points,
                                 serving=flags[i] if i < len(flags) else False,
                                 raw=" ".join(b.text for b in r)))
        return rows

    if not tesseract_available():
        return []
    bands = [(int(r * h / n_rows), int((r + 1) * h / n_rows)) for r in range(n_rows)]
    flags = _serving_flags(crop, bands, w * 0.12)
    for r, (y1, y2) in enumerate(bands):
        band = crop[y1:y2]
        full = _tesseract(_prep(band), psm=7)
        nums = _tesseract(_prep(band[:, int(w * 0.45):]), psm=7, whitelist="0123456789AD ")
        name = " ".join(_ALPHA.findall(full)).strip()
        games, points = _parse_numbers(nums)
        if not games and not points:
            games, points = _parse_numbers(full)
        rows.append(ScoreRow(name=name, games=games, points=points,
                             serving=flags[r] if r < len(flags) else False,
                             raw=f"{full} | {nums}"))
    return rows


# ── Timeline ─────────────────────────────────────────────────────────────────

def read_scoreboard_timeline(
    frames_dir: Path,
    total_frames: int,
    fps: float,
    sample_s: float = 0.5,
    roi: tuple[int, int, int, int] | None = None,
) -> ScoreTimeline:
    """Sample frames every ``sample_s`` seconds and OCR the scoreboard."""
    timeline = ScoreTimeline()
    if not ocr_available() or total_frames <= 0:
        return timeline
    step = max(1, int(round(sample_s * fps)))
    sample_frames = list(range(0, total_frames, step))
    if roi is None:
        probe_idx = sample_frames[:: max(1, len(sample_frames) // 12)][:12]
        probe = [cv2.imread(str(frames_dir / f"frame_{i:06d}.jpg")) for i in probe_idx]
        probe = [f for f in probe if f is not None]
        roi = locate_scoreboard(probe)
    if roi is None:
        return timeline
    timeline.roi = roi
    x1, y1, x2, y2 = roi
    for i in sample_frames:
        frame = cv2.imread(str(frames_dir / f"frame_{i:06d}.jpg"))
        if frame is None:
            continue
        rows = ocr_scoreboard(frame[y1:y2, x1:x2])
        timeline.samples.append(ScoreSample(frame=i, rows=rows))
    return timeline


# ── Outcome inference ────────────────────────────────────────────────────────

def compare_states(before: ScoreSample, after: ScoreSample) -> int | None:
    """Return the index (0/1) of the row that won a point between two states,
    or None if the change is not a single-point advance."""
    if not (before.is_valid() and after.is_valid()):
        return None
    b0, b1 = before.rows
    a0, a1 = after.rows
    # Game count change wins outright.
    for idx, (b, a) in enumerate(((b0, a0), (b1, a1))):
        if b.games and a.games and len(a.games) == len(b.games) and sum(a.games) == sum(b.games) + 1:
            other_b, other_a = (b1, a1) if idx == 0 else (b0, a0)
            if other_b.games == other_a.games or sum(other_a.games) == sum(other_b.games):
                return idx
    r0b, r0a = b0.point_rank(), a0.point_rank()
    r1b, r1a = b1.point_rank(), a1.point_rank()
    if None in (r0b, r0a, r1b, r1a):
        # Points unreadable: fall back to game counts of the last set column.
        try:
            g0 = a0.games[-1] - b0.games[-1]
            g1 = a1.games[-1] - b1.games[-1]
        except (IndexError, TypeError):
            return None
        if g0 == 1 and g1 == 0:
            return 0
        if g1 == 1 and g0 == 0:
            return 1
        return None
    d0, d1 = r0a - r0b, r1a - r1b
    # AD lost: AD-40 -> 40-40 means the *other* row won.
    if r0b == 4 and r0a == 3 and r1b == 3 and r1a == 3:
        return 1
    if r1b == 4 and r1a == 3 and r0b == 3 and r0a == 3:
        return 0
    if d0 > 0 and d1 <= 0:
        return 0
    if d1 > 0 and d0 <= 0:
        return 1
    return None


def infer_point_winner_row(
    timeline: ScoreTimeline,
    point_start: int,
    point_end: int,
    next_point_start: int | None,
    fps: float,
) -> int | None:
    """Winner row index for the point spanning [point_start, point_end]."""
    before = timeline.stable_state(point_start, max(point_start, min(point_end, point_start + int(3 * fps))))
    if before is None:
        before = timeline.sample_at(point_start, "before")
    if before is None:
        return None
    window_end = next_point_start if next_point_start is not None else point_end + int(20 * fps)
    after_samples = [
        s for s in timeline.samples
        if s.is_valid() and point_end < s.frame <= window_end + int(3 * fps)
    ]
    for s in after_samples:
        winner = compare_states(before, s)
        if winner is not None:
            return winner
    return None
