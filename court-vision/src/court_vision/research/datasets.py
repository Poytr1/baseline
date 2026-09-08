"""Public dataset access for cross-validation (subset fetch over HTTP Range).

Both reference datasets live in multi-GB zip archives on HuggingFace; the
servers support HTTP Range requests, so we read the zip central directory
remotely and extract only the members we need (a few hundred court images,
one TrackNet game folder) instead of downloading everything.

* Court keypoints — yastrebksv/TennisCourtDetector (8,841 broadcast frames,
  14 keypoints each). HF mirror: Gholamreza/tennis_court_keypoints_dataset.
* Ball positions — TrackNet tennis dataset (Huang et al. 2019; game1..game10,
  Label.csv with visibility / x / y / status(0 flying, 1 hit, 2 bounce)).
  HF mirror: owen1233/tracknet_preprocessd_data (tennis.zip).

Licenses: neither original dataset states a license; the mirrors are tagged
MIT by their uploaders. They are used here for evaluation only.
"""

from __future__ import annotations

import csv
import io
import json
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

COURT_ZIP_URL = "https://huggingface.co/datasets/Gholamreza/tennis_court_keypoints_dataset/resolve/main/tennis_court_det_dataset.zip"
BALL_ZIP_URL = "https://huggingface.co/datasets/owen1233/tracknet_preprocessd_data/resolve/main/tennis.zip"

# TennisCourtDetector keypoint index -> our doubles corner names
COURT_CORNER_IDX = {"far_left": 0, "far_right": 1, "near_left": 2, "near_right": 3}
DEFAULT_DIR = Path("data/public_datasets")


class RemoteZip:
    """Minimal read-only zip reader over HTTP Range requests."""

    def __init__(self, url: str, timeout: int = 120) -> None:
        self.url = url
        self.timeout = timeout
        self.size = self._content_length()
        self.entries: dict[str, tuple[int, int, int, int]] = {}  # name -> (local_header_offset, comp_size, uncomp_size, method)
        self._read_central_directory()

    def _request(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "court-vision/0.1"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def _content_length(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD", headers={"User-Agent": "court-vision/0.1"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return int(r.headers["Content-Length"])

    def _read_central_directory(self) -> None:
        tail = self._request(max(0, self.size - 66000), self.size - 1)
        # ZIP64 end of central directory locator
        loc = tail.rfind(b"PK\x06\x07")
        eocd = tail.rfind(b"PK\x05\x06")
        if loc != -1:
            z64_eocd_offset = struct.unpack("<Q", tail[loc + 8: loc + 16])[0]
            z64 = self._request(z64_eocd_offset, z64_eocd_offset + 56 - 1)
            cd_size = struct.unpack("<Q", z64[40:48])[0]
            cd_offset = struct.unpack("<Q", z64[48:56])[0]
        else:
            cd_size = struct.unpack("<I", tail[eocd + 12: eocd + 16])[0]
            cd_offset = struct.unpack("<I", tail[eocd + 16: eocd + 20])[0]
        cd = self._request(cd_offset, cd_offset + cd_size - 1)
        pos = 0
        while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
            method = struct.unpack("<H", cd[pos + 10: pos + 12])[0]
            comp = struct.unpack("<I", cd[pos + 20: pos + 24])[0]
            uncomp = struct.unpack("<I", cd[pos + 24: pos + 28])[0]
            n_len, e_len, c_len = struct.unpack("<HHH", cd[pos + 28: pos + 34])
            lho = struct.unpack("<I", cd[pos + 42: pos + 46])[0]
            name = cd[pos + 46: pos + 46 + n_len].decode("utf-8", "replace")
            extra = cd[pos + 46 + n_len: pos + 46 + n_len + e_len]
            # zip64 extra field
            if 0xFFFFFFFF in (comp, uncomp, lho):
                epos = 0
                while epos + 4 <= len(extra):
                    hid, hsize = struct.unpack("<HH", extra[epos: epos + 4])
                    if hid == 0x0001:
                        vals = extra[epos + 4: epos + 4 + hsize]
                        vpos = 0
                        if uncomp == 0xFFFFFFFF:
                            uncomp = struct.unpack("<Q", vals[vpos: vpos + 8])[0]; vpos += 8
                        if comp == 0xFFFFFFFF:
                            comp = struct.unpack("<Q", vals[vpos: vpos + 8])[0]; vpos += 8
                        if lho == 0xFFFFFFFF:
                            lho = struct.unpack("<Q", vals[vpos: vpos + 8])[0]; vpos += 8
                        break
                    epos += 4 + hsize
            self.entries[name] = (lho, comp, uncomp, method)
            pos += 46 + n_len + e_len + c_len

    def read(self, name: str) -> bytes:
        lho, comp, uncomp, method = self.entries[name]
        header = self._request(lho, lho + 30 - 1)
        n_len, e_len = struct.unpack("<HH", header[26:30])
        start = lho + 30 + n_len + e_len
        data = self._request(start, start + comp - 1)
        if method == 0:
            return data
        if method == 8:
            return zlib.decompress(data, -15)
        raise ValueError(f"unsupported compression method {method} for {name}")

    def names(self, prefix: str = "") -> list[str]:
        return [n for n in self.entries if n.startswith(prefix)]


# ── Court dataset ────────────────────────────────────────────────────────────

def fetch_court_subset(
    out_dir: Path = DEFAULT_DIR,
    n_images: int = 300,
    split: str = "val",
    seed: int = 0,
    log=print,
) -> Path:
    """Fetch ``n_images`` labelled images from the TennisCourtDetector val
    split and write ``court_labels.json`` in our label fixture format."""
    import random

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rz = RemoteZip(COURT_ZIP_URL)
    names = rz.names()
    json_name = next(n for n in names if n.endswith(f"data_{split}.json"))
    local_json = out_dir / f"data_{split}.json"
    if not local_json.exists():
        local_json.write_bytes(rz.read(json_name))
        log(f"[court] wrote {local_json}")
    anns = json.loads(local_json.read_text())
    rng = random.Random(seed)
    rng.shuffle(anns)
    img_prefix = next((n for n in names if n.endswith("/images/") or "/images/" in n), "")
    img_dir_in_zip = img_prefix[: img_prefix.index("/images/") + len("/images/")] if "/images/" in img_prefix else "images/"
    chosen = []
    for a in anns:
        member = f"{img_dir_in_zip}{a['id']}.png"
        if member in rz.entries and len(a.get("kps") or []) >= 4:
            chosen.append((a, member))
        if len(chosen) >= n_images:
            break
    _download_many(rz, [(m, images_dir / f"{a['id']}.png") for a, m in chosen
                        if not (images_dir / f"{a['id']}.png").exists()])
    court = []
    for a, member in chosen:
        local = images_dir / f"{a['id']}.png"
        if not local.exists():
            continue
        kps = a.get("kps") or []
        if len(kps) < 4:
            continue
        court.append({
            "frame": len(court),
            "image_path": str(local.resolve()),
            "corners": {k: [float(kps[i][0]), float(kps[i][1])] for k, i in COURT_CORNER_IDX.items()},
            "keypoints": [[float(x), float(y)] for x, y in kps],
        })
        if len(court) % 50 == 0:
            log(f"[court] {len(court)} images")
    fixture = {"frames_dir": str(images_dir.resolve()), "image_size": [1280, 720], "ball": [], "players": [], "court": court,
               "source": COURT_ZIP_URL, "split": split}
    out = out_dir / "court_labels.json"
    out.write_text(json.dumps(fixture, indent=1))
    log(f"[court] {len(court)} labelled images -> {out}")
    return out


# ── Ball dataset ─────────────────────────────────────────────────────────────

@dataclass
class BallClip:
    game: str
    clip: str
    frames_dir: Path
    labels: list[dict]  # {"frame": int, "file": str, "visible": bool, "x": float|None, "y": float|None, "status": int}


def _download_many(rz: "RemoteZip", todo: list[tuple[str, Path]], workers: int = 8) -> int:
    """Fetch zip members concurrently (each is an independent Range request)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(item: tuple[str, Path]) -> int:
        name, local = item
        try:
            local.write_bytes(rz.read(name))
            return 1
        except Exception:
            return 0

    if not todo:
        return 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(one, todo))


def fetch_ball_subset(
    out_dir: Path = DEFAULT_DIR,
    games: tuple[str, ...] = ("game7",),
    max_clips: int | None = None,
    log=print,
    workers: int = 8,
) -> Path:
    """Fetch whole TrackNet game folders (frames + Label.csv) and write a
    manifest ``ball_clips.json``."""
    out_dir = Path(out_dir) / "tracknet"
    out_dir.mkdir(parents=True, exist_ok=True)
    rz = RemoteZip(BALL_ZIP_URL)
    manifest: list[dict] = []
    for game in games:
        members = [n for n in rz.entries if f"/{game}/" in n and not n.endswith("/")]
        clips = sorted({n.split(f"/{game}/")[1].split("/")[0] for n in members})
        if max_clips:
            clips = clips[:max_clips]
        for clip in clips:
            cdir = out_dir / game / clip
            cdir.mkdir(parents=True, exist_ok=True)
            label_member = next((n for n in members if n.endswith(f"/{game}/{clip}/Label.csv")), None)
            if label_member is None:
                continue
            label_local = cdir / "Label.csv"
            if not label_local.exists():
                label_local.write_bytes(rz.read(label_member))
            todo = [
                (n, cdir / Path(n).name) for n in members
                if f"/{game}/{clip}/" in n and n.lower().endswith(".jpg") and not (cdir / Path(n).name).exists()
            ]
            n_written = _download_many(rz, todo, workers=workers)
            rows = list(csv.DictReader(io.StringIO(label_local.read_text())))
            manifest.append({"game": game, "clip": clip, "dir": str(cdir.resolve()), "frames": len(rows)})
            log(f"[ball] {game}/{clip}: {len(rows)} labelled frames ({n_written} downloaded)")
    out = out_dir / "ball_clips.json"
    out.write_text(json.dumps(manifest, indent=1))
    log(f"[ball] manifest -> {out}")
    return out


def load_ball_clip(entry: dict) -> BallClip:
    cdir = Path(entry["dir"])
    rows = list(csv.DictReader(io.StringIO((cdir / "Label.csv").read_text())))
    labels = []
    for i, r in enumerate(rows):
        vis = int(float(r.get("visibility", 0) or 0))
        x = r.get("x-coordinate") or r.get("x")
        y = r.get("y-coordinate") or r.get("y")
        labels.append({
            "frame": i, "file": r.get("file name") or r.get("file_name"),
            "visible": vis in (1, 2) and x not in (None, "", "0") and y not in (None, "", "0"),
            "x": float(x) if x not in (None, "") else None,
            "y": float(y) if y not in (None, "") else None,
            "status": int(float(r.get("status", 0) or 0)),
            "visibility": vis,
        })
    return BallClip(game=entry["game"], clip=entry["clip"], frames_dir=cdir, labels=labels)
