"""Court detection — line detection, keypoint extraction, homography."""

# Standard tennis court dimensions in meters.
# Origin (0, 0) = center of net.
# X-axis: parallel to net, positive = right when facing far end.
# Y-axis: perpendicular to net, positive = far side.

_SINGLES_WIDTH_HALF = 4.115  # Singles sideline to center
_DOUBLES_WIDTH_HALF = 5.485  # Doubles sideline to center
_BASELINE_DIST = 11.885  # Baseline to net
_SERVICE_LINE_DIST = 6.4  # Service line to net

COURT_KEYPOINTS: dict[str, tuple[float, float]] = {
    # Far baseline (y = +11.885)
    "baseline_far_left_doubles": (-_DOUBLES_WIDTH_HALF, _BASELINE_DIST),
    "baseline_far_left_singles": (-_SINGLES_WIDTH_HALF, _BASELINE_DIST),
    "baseline_far_center": (0.0, _BASELINE_DIST),
    "baseline_far_right_singles": (_SINGLES_WIDTH_HALF, _BASELINE_DIST),
    "baseline_far_right_doubles": (_DOUBLES_WIDTH_HALF, _BASELINE_DIST),
    # Far service line (y = +6.4)
    "service_far_left": (-_SINGLES_WIDTH_HALF, _SERVICE_LINE_DIST),
    "service_far_center": (0.0, _SERVICE_LINE_DIST),
    "service_far_right": (_SINGLES_WIDTH_HALF, _SERVICE_LINE_DIST),
    # Net (y = 0)
    "net_left_singles": (-_SINGLES_WIDTH_HALF, 0.0),
    "net_center": (0.0, 0.0),
    "net_right_singles": (_SINGLES_WIDTH_HALF, 0.0),
    "net_left_doubles": (-_DOUBLES_WIDTH_HALF, 0.0),
    "net_right_doubles": (_DOUBLES_WIDTH_HALF, 0.0),
    # Near service line (y = -6.4)
    "service_near_left": (-_SINGLES_WIDTH_HALF, -_SERVICE_LINE_DIST),
    "service_near_center": (0.0, -_SERVICE_LINE_DIST),
    "service_near_right": (_SINGLES_WIDTH_HALF, -_SERVICE_LINE_DIST),
    # Near baseline (y = -11.885)
    "baseline_near_left_doubles": (-_DOUBLES_WIDTH_HALF, -_BASELINE_DIST),
    "baseline_near_left_singles": (-_SINGLES_WIDTH_HALF, -_BASELINE_DIST),
    "baseline_near_center": (0.0, -_BASELINE_DIST),
    "baseline_near_right_singles": (_SINGLES_WIDTH_HALF, -_BASELINE_DIST),
    "baseline_near_right_doubles": (_DOUBLES_WIDTH_HALF, -_BASELINE_DIST),
}
