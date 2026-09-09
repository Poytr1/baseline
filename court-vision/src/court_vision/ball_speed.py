"""Shot speed from contact to the first bounce.

A single broadcast camera cannot measure a ball in the air: the ground
homography projects an airborne ball metres beyond where it is. Two places
are reliable, though — the hitter's feet at contact and the spot where the
ball lands (``shot_classify.estimate_bounce``). Their distance divided by
the flight time is the average speed over the flight, a little under the
launch speed a radar gun reports but free of the height bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from court_vision.player_detect import FrameTrackingResult
from court_vision.trajectory import map_ball_to_court

_MIN_KMH = 20.0
_MAX_KMH = 260.0


@dataclass
class SpeedEstimate:
    kmh: float
    distance_m: float
    seconds: float
    bounce_frame: int
    method: str = "bounce"


def bounce_speed(
    hitter_feet: tuple[float, float] | None,
    landing: tuple[float, float, int],
    hit_frame: int,
    fps: float,
    by_frame: dict[int, FrameTrackingResult] | None = None,
    homography: np.ndarray | None = None,
) -> SpeedEstimate | None:
    """Average speed from contact to the first bounce.

    Both endpoints are on the ground — the hitter's feet at contact and the
    landing spot — so the homography is trusted at both, and the flight
    time is a frame count. This is the mean speed over the whole flight:
    about 10-15% under the launch speed a radar gun reports, but free of
    the height bias that makes projecting an airborne ball meaningless.
    """
    x1, y1, bounce_frame = landing
    if hitter_feet is None:
        if by_frame is None or homography is None:
            return None
        t0 = by_frame.get(hit_frame)
        if t0 is None or t0.ball is None:
            return None
        hitter_feet = map_ball_to_court(t0.ball, homography)
        if hitter_feet is None:
            return None
    x0, y0 = hitter_feet
    seconds = (bounce_frame - hit_frame) / fps
    if seconds < 0.15:
        return None
    distance = float(np.hypot(x1 - x0, y1 - y0))
    kmh = distance / seconds * 3.6
    if not (_MIN_KMH <= kmh <= _MAX_KMH):
        return None
    return SpeedEstimate(kmh=round(kmh, 1), distance_m=round(distance, 2), seconds=round(seconds, 3),
                         bounce_frame=int(bounce_frame), method="bounce")
