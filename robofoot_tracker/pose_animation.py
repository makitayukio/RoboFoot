# =============================================================================
# Grupo 3 - RoboFoot Tracker
# MCZA018 - Processamento Digital de Imagens - 2026.1
#
# Integrantes:
#   - Igor Ladeia de Freitas         (RA: 11201922180)
#   - Gustavo Fernandes do Nascimento (RA: 11202021700)
#   - Ryan Lucas da Silva            (RA: 11202522362)
#   - Eduardo Yukio Makita            (RA: 11202020221)
#
# Data: 2026-04-25
# Programa: robofoot_tracker
# Exemplo de execução:
#   $ python -c "from robofoot_tracker import Tracker; Tracker(camera=0).run_live()"
# =============================================================================
"""Render a top-down synthetic field MP4 from a poses CSV."""

from __future__ import annotations

import csv
import math
from collections import defaultdict

import cv2
import numpy as np

from .models import FieldDimensions, RobotDetection, BallDetection
from .viz import draw_detections
from .calibration import calibrate_from_points

_TEAM_BGR = {"blue": (255, 150, 0), "yellow": (0, 255, 255)}
_BALL_BGR = (0, 165, 255)
_ROBOT_RADIUS = 18
_BALL_RADIUS = 6


def _render_field_frame(
    rows: list[dict],
    dims: FieldDimensions,
    scale: float,
    width_px: int,
    height_px: int,
    bg: np.ndarray,
) -> np.ndarray:
    """Render a single synthetic top-down field frame from CSV rows.

    Args:
        rows: CSV rows (dicts) for this frame.
        dims: Field dimensions (unused directly, kept for future use).
        scale: Pixels per cm.
        width_px: Field image width in pixels.
        height_px: Field image height in pixels.
        bg: Background image to copy from.

    Returns:
        BGR frame with robots and ball drawn.
    """
    frame = bg.copy()
    for row in rows:
        try:
            x_cm = float(row["x_cm"])
            y_cm = float(row["y_cm"])
        except (ValueError, KeyError):
            continue
        px, py = int(x_cm * scale), int(y_cm * scale)

        kind = row.get("kind", "")
        if kind == "ball":
            cv2.circle(frame, (px, py), _BALL_RADIUS, _BALL_BGR, -1)
        elif kind == "robot":
            team = row.get("team", "")
            if team not in _TEAM_BGR:
                continue
            color = _TEAM_BGR[team]
            cv2.circle(frame, (px, py), _ROBOT_RADIUS, color, -1)

            # Orientation arrow (same style as viz.draw_detections)
            try:
                angle = float(row.get("angle_deg", ""))
            except (ValueError, TypeError):
                angle = None
            if angle is not None:
                rad = math.radians(angle)
                bx = px + _ROBOT_RADIUS * math.cos(rad)
                by = py - _ROBOT_RADIUS * math.sin(rad)
                arrow_len = 20
                tip_x = int(bx + arrow_len * math.cos(rad))
                tip_y = int(by - arrow_len * math.sin(rad))
                shaft_end_x = int(bx + (arrow_len - 16) * math.cos(rad))
                shaft_end_y = int(by - (arrow_len - 16) * math.sin(rad))
                cv2.line(frame, (int(bx), int(by)), (shaft_end_x, shaft_end_y), color, 3)
                head_len, head_width = 16, 12
                back_x = tip_x - int(head_len * math.cos(rad))
                back_y = tip_y + int(head_len * math.sin(rad))
                perp = rad + math.pi / 2
                hw = head_width / 2
                lx = back_x + int(hw * math.cos(perp))
                ly = back_y - int(hw * math.sin(perp))
                rx = back_x - int(hw * math.cos(perp))
                ry = back_y + int(hw * math.sin(perp))
                tri = np.array([[tip_x, tip_y], [lx, ly], [rx, ry]], dtype=np.int32)
                cv2.fillPoly(frame, [tri], color)

            # Label
            rid = row.get("robot_id", "?")
            label = f"{'B' if team == 'blue' else 'Y'}-{rid}"
            cv2.putText(frame, label, (px + _ROBOT_RADIUS + 4, py - _ROBOT_RADIUS),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def _build_field_bg(width_px: int, height_px: int) -> np.ndarray:
    """Create the static field background image."""
    bg = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    cv2.rectangle(bg, (0, 0), (width_px - 1, height_px - 1), (30, 100, 30), -1)
    cx, cy = width_px // 2, height_px // 2
    cv2.line(bg, (cx, 0), (cx, height_px), (60, 140, 60), 1)
    r = int(0.1 * min(width_px, height_px))
    cv2.circle(bg, (cx, cy), r, (60, 140, 60), 1)
    return bg


def _read_csv_frames(csv_path: str) -> dict[int, list[dict]]:
    """Read CSV and group rows by frame_index."""
    frames: dict[int, list[dict]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                fi = int(row["frame_index"])
            except (ValueError, KeyError):
                continue
            frames[fi].append(row)
    return frames


def render_pose_animation(
    csv_path: str,
    out_path: str,
    field_dims: FieldDimensions | None = None,
    fps: int = 30,
    width_px: int = 800,
) -> None:
    """Render a top-down synthetic field MP4 from a poses CSV.

    Reads the CSV, groups rows by frame_index, and writes one MP4 frame per
    group. Field is rendered as a dark green rectangle with center line and
    center circle. Robots drawn as filled circles (blue/yellow) with orientation
    arrows matching viz.draw_detections style. Ball rendered as an orange filled
    circle.

    Output resolution: width_px × (width_px * field_height / field_width).
    """
    dims = field_dims or FieldDimensions()
    scale = width_px / dims.width_cm
    height_px = int(dims.height_cm * scale)

    bg = _build_field_bg(width_px, height_px)
    frames = _read_csv_frames(csv_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width_px, height_px))

    max_fi = max(frames) if frames else -1
    for fi in range(max_fi + 1):
        frame = _render_field_frame(frames.get(fi, []), dims, scale, width_px, height_px, bg)
        writer.write(frame)

    writer.release()


def _rows_to_detections(rows: list[dict]) -> tuple[list[RobotDetection], BallDetection | None]:
    """Reconstruct RobotDetection/BallDetection objects from CSV rows."""
    dets: list[RobotDetection] = []
    ball: BallDetection | None = None
    for row in rows:
        try:
            kind = row.get("kind", "")
            if kind == "robot":
                dets.append(RobotDetection(
                    team=row["team"],
                    robot_id=int(row["robot_id"]),
                    position=(float(row["x_cm"]), float(row["y_cm"])),
                    angle_deg=float(row["angle_deg"]),
                ))
            elif kind == "ball":
                ball = BallDetection(position=(float(row["x_cm"]), float(row["y_cm"])))
        except (ValueError, KeyError, TypeError):
            continue
    return dets, ball


def render_side_by_side_video(
    source_video_path: str,
    csv_path: str,
    out_path: str,
    field_dims: FieldDimensions | None = None,
    calibration_points: list[tuple[float, float]] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    fps: int | None = None,
    panel_height: int = 480,
) -> None:
    """Render a composite side-by-side MP4 combining the annotated source video
    with the synthetic top-down pose animation for the same frame range.

    Layout: |source video (annotated)| synthetic animation|
    Left panel: source video with detections drawn from CSV rows for each frame.
    Right panel: same top-down synthetic field as render_pose_animation.
    Both panels are scaled to the same height (panel_height px) preserving aspect ratio.

    If calibration_points is None, the annotated source will be drawn without
    annotations (the calibration matrix is required to project cm→pixel).

    FPS defaults to the source video's FPS.
    """
    dims = field_dims or FieldDimensions()

    # Calibration
    calibration = None
    if calibration_points is not None:
        calibration = calibrate_from_points(calibration_points, dims)

    # Open source video
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Read CSV
    frames_data = _read_csv_frames(csv_path)

    # Frame range
    start = max(0, start_frame)
    end = min(total_frames - 1, end_frame - 1) if end_frame is not None else total_frames - 1
    end = max(start, end)

    # Panel sizes (both panels share panel_height)
    source_panel_w = int(src_w * panel_height / src_h) if src_h > 0 else panel_height

    field_w_px = 800  # internal field rendering width
    field_scale = field_w_px / dims.width_cm
    field_h_px = int(dims.height_cm * field_scale)
    field_panel_w = int(field_w_px * panel_height / field_h_px) if field_h_px > 0 else panel_height

    composite_w = source_panel_w + field_panel_w

    # Field background
    field_bg = _build_field_bg(field_w_px, field_h_px)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (composite_w, panel_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for frame_idx in range(start, end + 1):
        ok, src_frame = cap.read()
        if not ok:
            break

        # Annotate source frame
        rows = frames_data.get(frame_idx, [])
        if calibration is not None and rows:
            dets, ball = _rows_to_detections(rows)
            src_frame = draw_detections(src_frame, dets, calibration, ball=ball)

        # Resize source panel
        left = cv2.resize(src_frame, (source_panel_w, panel_height), interpolation=cv2.INTER_AREA)

        # Synthetic field panel
        field_frame = _render_field_frame(rows, dims, field_scale, field_w_px, field_h_px, field_bg)
        right = cv2.resize(field_frame, (field_panel_w, panel_height), interpolation=cv2.INTER_AREA)

        # Composite
        composite = cv2.hconcat([left, right])
        writer.write(composite)

    cap.release()
    writer.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render pose animation or side-by-side comparison from CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("out_path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--source", type=str, default=None, help="If given, render side-by-side with this source video")
    parser.add_argument("--calibration-points", type=str, default=None, help='4 comma-separated x,y pixel pairs: "x1,y1,x2,y2,x3,y3,x4,y4"')
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--panel-height", type=int, default=480)
    args = parser.parse_args()

    if args.source:
        points = None
        if args.calibration_points:
            vals = [float(x) for x in args.calibration_points.split(',')]
            points = [(vals[i], vals[i+1]) for i in range(0, 8, 2)]
        render_side_by_side_video(
            args.source, args.csv_path, args.out_path,
            calibration_points=points,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            fps=args.fps,
            panel_height=args.panel_height,
        )
    else:
        render_pose_animation(args.csv_path, args.out_path, fps=args.fps, width_px=args.width)
