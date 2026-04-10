"""Robot detection via color segmentation and tag geometry analysis."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .calibration import transform_point
from .color_config import ColorConfig, ID_COLORS, TEAM_COLORS
from .models import CalibrationData, COLOR_PAIR_TO_ID, RobotDetection, BallDetection


def _make_mask(
    hsv: np.ndarray,
    ranges: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Union of cv2.inRange masks for multiple HSV range tuples."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def _contour_centroid(contour: np.ndarray) -> tuple[float, float] | None:
    """Return (cx, cy) of a contour via moments, or None if degenerate."""
    m = cv2.moments(contour)
    if m["m00"] < 1e-6:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def _find_blobs(
    hsv: np.ndarray,
    ranges: list[tuple[np.ndarray, np.ndarray]],
    min_area: float,
    max_area: float,
) -> list[tuple[float, float, float]]:
    """Return list of (cx, cy, area) for blobs matching *ranges* within area bounds."""
    mask = _make_mask(hsv, ranges)
    # Light morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            cen = _contour_centroid(c)
            if cen is not None:
                blobs.append((cen[0], cen[1], area))
    return blobs


def detect_robots(
    frame: np.ndarray,
    calibration: CalibrationData,
    color_config: ColorConfig,
    min_team_area: float = 50,
    max_team_area: float = 5000,
    min_id_area: float = 10,
    max_id_area: float = 2000,
    id_search_radius: float = 80.0,
    teams: list[str] | None = None,
) -> list[RobotDetection]:
    """Detect robots in a single BGR *frame*.

    Args:
        frame: BGR image (numpy array).
        calibration: Homography calibration data.
        color_config: HSV color ranges.
        min_team_area / max_team_area: Contour area bounds for team-color blobs.
        min_id_area / max_id_area: Contour area bounds for ID-color blobs.
        id_search_radius: Max pixel distance from team blob to search for ID blobs.
        teams: Optional list of team color strings to search (e.g. ``['blue']``).
            When *None*, all ``TEAM_COLORS`` are searched. Invalid names are
            silently skipped.

    Returns:
        List of RobotDetection for each robot found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Pre-compute all ID blobs keyed by color name
    id_blobs: dict[str, list[tuple[float, float, float]]] = {}
    for color in ID_COLORS:
        id_blobs[color] = _find_blobs(
            hsv, color_config.get_ranges(color), min_id_area, max_id_area
        )

    detections: list[RobotDetection] = []

    # Resolve which team colors to iterate
    search_teams = TEAM_COLORS if teams is None else [t for t in teams if t in TEAM_COLORS]

    for team_color in search_teams:
        team_blobs = _find_blobs(
            hsv, color_config.get_ranges(team_color), min_team_area, max_team_area
        )

        for tx, ty, _ in team_blobs:
            # Gather nearby ID blobs across all ID colors
            nearby: list[tuple[str, float, float, float]] = []  # (color, x, y, dist)
            for color, blobs in id_blobs.items():
                for bx, by, _ in blobs:
                    dist = math.hypot(bx - tx, by - ty)
                    if dist <= id_search_radius:
                        nearby.append((color, bx, by, dist))

            if len(nearby) < 2:
                continue

            # Take the 2 closest ID blobs
            nearby.sort(key=lambda n: n[3])
            id1_color, id1_x, id1_y, _ = nearby[0]
            id2_color, id2_x, id2_y, _ = nearby[1]

            # Robot ID from sorted color pair
            pair = tuple(sorted([id1_color, id2_color]))
            robot_id = COLOR_PAIR_TO_ID.get(pair)  # type: ignore[arg-type]
            if robot_id is None:
                continue

            # Position: midpoint of team blob and ID blobs centroid
            id_cx = (id1_x + id2_x) / 2
            id_cy = (id1_y + id2_y) / 2
            mid_x = (tx + id_cx) / 2
            mid_y = (ty + id_cy) / 2

            # Angle: from ID-blobs centroid toward team-color centroid
            # Negate y because OpenCV y-axis is inverted
            angle_rad = math.atan2(-(ty - id_cy), tx - id_cx)
            angle_deg = (math.degrees(angle_rad) + 180) % 360

            # Transform to field coordinates
            pos = transform_point((mid_x, mid_y), calibration.homography_matrix)

            detections.append(
                RobotDetection(
                    team=team_color,
                    robot_id=robot_id,
                    position=pos,
                    angle_deg=round(angle_deg, 2),
                )
            )

    return detections


def detect_ball(
    frame: np.ndarray,
    calibration: CalibrationData,
    color_config: ColorConfig,
    min_area: float = 10,
    max_area: float = 2000,
) -> BallDetection | None:
    """Detect the ball in a single BGR *frame*.

    Uses the 'orange' colour range and picks the largest blob.

    Returns:
        BallDetection or None if no ball found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blobs = _find_blobs(hsv, color_config.get_ranges("orange"), min_area, max_area)
    if not blobs:
        return None
    # Pick largest blob by area
    best = max(blobs, key=lambda b: b[2])
    pos = transform_point((best[0], best[1]), calibration.homography_matrix)
    return BallDetection(position=pos)
