"""Visualization utilities for drawing robot detections on frames."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import bezier_curve_points as _bezier_curve_points  # re-export
from .models import CalibrationData, RobotDetection, BallDetection

_TEAM_BGR = {
    "blue": (255, 150, 0),
    "yellow": (0, 255, 255),
}
_ROBOT_RADIUS = 18


def _field_to_pixel(
    pos: tuple[float, float],
    inv_homography,
) -> tuple[int, int]:
    """Inverse-homography: field coords → pixel coords.

    *inv_homography* may be a precomputed 3×3 inverse matrix **or** a
    ``CalibrationData`` instance (for backward compatibility).
    """
    if isinstance(inv_homography, CalibrationData):
        inv_homography = np.linalg.inv(inv_homography.homography_matrix)
    src = np.array([[[pos[0], pos[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, inv_homography)
    return (int(dst[0, 0, 0]), int(dst[0, 0, 1]))


def draw_detections(
    frame: np.ndarray,
    detections: list[RobotDetection],
    calibration: CalibrationData | None = None,
    ball: BallDetection | None = None,
) -> np.ndarray:
    """Draw robot annotations on a copy of *frame*.

    If *calibration* is provided, field-coordinate positions are mapped back
    to pixel space for drawing. Otherwise positions are treated as pixel coords.

    If *ball* is provided and *calibration* is set, an orange filled circle is
    drawn at the ball position.

    Returns:
        Annotated BGR frame (original is not modified).
    """
    out = frame.copy()

    inv_h = np.linalg.inv(calibration.homography_matrix) if calibration is not None else None

    if calibration is not None:
        if calibration.all_points is not None and calibration.all_points.shape[0] == 8:
            ap = calibration.all_points
            edges = [(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 0)]
            for i0, i1, i2 in edges:
                pts = _bezier_curve_points(ap[i0], ap[i1], ap[i2])
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=False, color=(0, 255, 0), thickness=1)
        else:
            border = calibration.src_points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(out, [border], isClosed=True, color=(0, 255, 0), thickness=1)

    for det in detections:
        if inv_h is not None:
            px, py = _field_to_pixel(det.position, inv_h)
        else:
            px, py = int(det.position[0]), int(det.position[1])

        color = _TEAM_BGR.get(det.team, (255, 255, 255))
        label = f"{'B' if det.team == 'blue' else 'Y'}-{det.robot_id}"

        # Circle at position
        cv2.circle(out, (px, py), _ROBOT_RADIUS, color, 2)

        # Arrowhead at circle edge showing orientation
        rad = math.radians(det.angle_deg)
        tip_x = int(px + _ROBOT_RADIUS * math.cos(rad))
        tip_y = int(py - _ROBOT_RADIUS * math.sin(rad))
        head_len, head_width = 12, 8
        base_x = tip_x - int(head_len * math.cos(rad))
        base_y = tip_y + int(head_len * math.sin(rad))
        perp_rad = rad + math.pi / 2
        hw = head_width / 2
        lx = base_x + int(hw * math.cos(perp_rad))
        ly = base_y - int(hw * math.sin(perp_rad))
        rx = base_x - int(hw * math.cos(perp_rad))
        ry = base_y + int(hw * math.sin(perp_rad))
        triangle = np.array([[tip_x, tip_y], [lx, ly], [rx, ry]], dtype=np.int32)
        cv2.fillPoly(out, [triangle], color)

        # Text label
        cv2.putText(out, label, (px + 12, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw ball
    if ball is not None and inv_h is not None:
        bx, by = _field_to_pixel(ball.position, inv_h)
        cv2.circle(out, (bx, by), 6, (0, 165, 255), -1)

    return out
