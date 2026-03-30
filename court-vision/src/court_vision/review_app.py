"""Streamlit Review UI for Court Vision match data."""

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from court_vision.overlay import render_overlay
from court_vision.player_detect import FrameTrackingResult
from court_vision.review_data import load_match_json, save_match_json
from court_vision.shot_classify import MatchData

CONFIDENCE_THRESHOLD = 0.7
STROKE_OPTIONS = ["forehand", "backhand", "serve", "volley", "overhead", "slice"]
OUTCOME_OPTIONS = ["winner", "error", "unforced_error"]
STATUS_OPTIONS = ["pending", "approved", "corrected"]
STATUS_COLORS = {
    "pending": "#FFA500",
    "approved": "#00CC00",
    "corrected": "#3399FF",
}


def load_tracking_data(tracking_path: Path) -> dict[int, FrameTrackingResult]:
    """Load per-frame tracking data from JSON if available."""
    if not tracking_path.exists():
        return {}
    with open(tracking_path) as f:
        raw = json.load(f)
    # Tracking data is stored as a list of frame results
    from court_vision.ball_tracker import BallDetection
    from court_vision.player_detect import PlayerDetection, PoseKeypoints

    results: dict[int, FrameTrackingResult] = {}
    for entry in raw:
        ball = None
        if entry.get("ball"):
            b = entry["ball"]
            ball = BallDetection(
                frame_index=b["frame_index"], x=b["x"], y=b["y"],
                confidence=b["confidence"],
                interpolated=b.get("interpolated", False),
            )
        players = [
            PlayerDetection(
                frame_index=p["frame_index"], bbox=tuple(p["bbox"]),
                confidence=p["confidence"],
                court_position=tuple(p["court_position"]) if p.get("court_position") else None,
                role=p.get("role"),
            )
            for p in entry.get("players", [])
        ]
        poses = [
            PoseKeypoints(
                frame_index=pk["frame_index"], role=pk["role"],
                keypoints={k: tuple(v) for k, v in pk["keypoints"].items()},
            )
            for pk in entry.get("poses", [])
        ]
        frame_idx = entry["frame_index"]
        results[frame_idx] = FrameTrackingResult(
            frame_index=frame_idx, ball=ball, players=players, poses=poses,
        )
    return results


def get_frame_image(frames_dir: Path, frame_index: int) -> np.ndarray | None:
    """Load a frame image from the frames directory."""
    frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
    if not frame_path.exists():
        return None
    return cv2.imread(str(frame_path))


def point_confidence(match: MatchData, point_idx: int) -> float:
    """Get the minimum shot confidence for a point."""
    point = match.points[point_idx]
    if not point.shots:
        return 1.0
    return min(s.confidence for s in point.shots)


def run_app() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="Court Vision Review", layout="wide")
    st.title("Court Vision — Match Review")

    # --- Sidebar: File Loading ---
    with st.sidebar:
        st.header("Match Data")
        match_path = st.text_input("Match JSON path", value=st.session_state.get("match_path", ""))
        frames_dir = st.text_input("Frames directory", value=st.session_state.get("frames_dir", ""))
        tracking_path = st.text_input("Tracking JSON (optional)", value=st.session_state.get("tracking_path", ""))

        if st.button("Load Match"):
            if match_path and Path(match_path).exists():
                st.session_state["match_path"] = match_path
                st.session_state["match"] = load_match_json(Path(match_path))
                st.session_state["frames_dir"] = frames_dir
                st.session_state["tracking_path"] = tracking_path
                if tracking_path and Path(tracking_path).exists():
                    st.session_state["tracking"] = load_tracking_data(Path(tracking_path))
                else:
                    st.session_state["tracking"] = {}
                st.session_state["selected_point"] = 0
                st.rerun()
            else:
                st.error("Match JSON file not found.")

    match: MatchData | None = st.session_state.get("match")
    if match is None:
        st.info("Load a match JSON file to begin review.")
        return

    # --- Sidebar: Point List ---
    with st.sidebar:
        st.header("Points")

        # Bulk actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve High-Conf"):
                for point in match.points:
                    if point.review_status == "pending":
                        min_conf = min((s.confidence for s in point.shots), default=1.0)
                        if min_conf >= CONFIDENCE_THRESHOLD:
                            point.review_status = "approved"
                st.rerun()
        with col2:
            if st.button("Save"):
                save_match_json(match, Path(st.session_state["match_path"]))
                st.success("Saved!")

        # Skip to next low-confidence
        if st.button("Next Low-Conf"):
            current = st.session_state.get("selected_point", 0)
            for i in range(current + 1, len(match.points)):
                if point_confidence(match, i) < CONFIDENCE_THRESHOLD:
                    st.session_state["selected_point"] = i
                    st.rerun()
            st.warning("No more low-confidence points.")

        # Status summary
        pending = sum(1 for p in match.points if p.review_status == "pending")
        approved = sum(1 for p in match.points if p.review_status == "approved")
        corrected = sum(1 for p in match.points if p.review_status == "corrected")
        st.caption(f"Pending: {pending} | Approved: {approved} | Corrected: {corrected}")

        # Point list
        for i, point in enumerate(match.points):
            min_conf = point_confidence(match, i)
            color = STATUS_COLORS.get(point.review_status, "#999")
            conf_marker = " ⚠" if min_conf < CONFIDENCE_THRESHOLD else ""
            label = f"Point {point.point_number} [{point.review_status}]{conf_marker}"
            if st.button(label, key=f"point_{i}", use_container_width=True):
                st.session_state["selected_point"] = i
                st.rerun()

    # --- Main Area ---
    selected_idx = st.session_state.get("selected_point", 0)
    if selected_idx >= len(match.points):
        selected_idx = 0
    point = match.points[selected_idx]

    st.subheader(f"Point {point.point_number}")
    st.caption(f"Frames {point.start_frame}–{point.end_frame} | "
               f"{point.start_time_s:.1f}s–{point.end_time_s:.1f}s | "
               f"Rally: {point.rally_length} shots")

    # Frame viewer
    f_dir = st.session_state.get("frames_dir", "")
    tracking_data: dict[int, FrameTrackingResult] = st.session_state.get("tracking", {})

    if f_dir and Path(f_dir).exists():
        frame_num = st.slider(
            "Frame", min_value=point.start_frame, max_value=point.end_frame,
            value=point.start_frame, key=f"frame_slider_{selected_idx}",
        )

        frame_img = get_frame_image(Path(f_dir), frame_num)
        if frame_img is not None:
            # Apply overlay if tracking data exists
            if frame_num in tracking_data:
                frame_img = render_overlay(frame_img, tracking_data[frame_num])
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)
        else:
            st.warning(f"Frame {frame_num} not found in {f_dir}")
    else:
        st.info("Set frames directory to view video frames.")

    # --- Shot Editor ---
    st.subheader("Shots")

    changed = False
    for j, shot in enumerate(point.shots):
        with st.expander(f"Shot {shot.shot_number}: {shot.stroke} by {shot.player} "
                         f"(conf: {shot.confidence:.2f})", expanded=True):
            col_stroke, col_outcome = st.columns(2)
            with col_stroke:
                new_stroke = st.selectbox(
                    "Stroke", STROKE_OPTIONS,
                    index=STROKE_OPTIONS.index(shot.stroke) if shot.stroke in STROKE_OPTIONS else 0,
                    key=f"stroke_{selected_idx}_{j}",
                )
                if new_stroke != shot.stroke:
                    shot.stroke = new_stroke
                    changed = True

            with col_outcome:
                if j == len(point.shots) - 1:  # Last shot determines outcome
                    current_outcome = point.outcome or "winner"
                    new_outcome = st.selectbox(
                        "Point Outcome", OUTCOME_OPTIONS,
                        index=OUTCOME_OPTIONS.index(current_outcome) if current_outcome in OUTCOME_OPTIONS else 0,
                        key=f"outcome_{selected_idx}_{j}",
                    )
                    if new_outcome != point.outcome:
                        point.outcome = new_outcome
                        changed = True

            if shot.placement:
                st.caption(f"Placement: ({shot.placement.x:.1f}, {shot.placement.y:.1f}) — {shot.placement.zone}")

            # Jump to contact frame
            if st.button(f"Go to frame {shot.frame}", key=f"goto_{selected_idx}_{j}"):
                st.session_state[f"frame_slider_{selected_idx}"] = shot.frame
                st.rerun()

    # Review status
    st.subheader("Review Status")
    new_status = st.selectbox(
        "Status", STATUS_OPTIONS,
        index=STATUS_OPTIONS.index(point.review_status) if point.review_status in STATUS_OPTIONS else 0,
        key=f"status_{selected_idx}",
    )
    if new_status != point.review_status:
        point.review_status = new_status
        changed = True

    if changed:
        if point.review_status == "pending":
            point.review_status = "corrected"

    # Navigation
    col_prev, col_next = st.columns(2)
    with col_prev:
        if selected_idx > 0:
            if st.button("← Previous Point"):
                st.session_state["selected_point"] = selected_idx - 1
                st.rerun()
    with col_next:
        if selected_idx < len(match.points) - 1:
            if st.button("Next Point →"):
                st.session_state["selected_point"] = selected_idx + 1
                st.rerun()


if __name__ == "__main__":
    run_app()
