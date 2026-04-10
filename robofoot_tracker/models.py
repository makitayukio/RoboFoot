"""Shared data models for robofoot_tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FieldDimensions:
    """Physical field size in centimeters."""
    width_cm: float = 150.0
    height_cm: float = 130.0


@dataclass
class CalibrationData:
    """Homography calibration result."""
    homography_matrix: np.ndarray
    src_points: np.ndarray  # 4 pixel-space corners
    dst_points: np.ndarray  # 4 field-space corners
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    all_points: Optional[np.ndarray] = None


@dataclass
class RobotDetection:
    """Single robot detection in one frame."""
    team: str              # 'blue' or 'yellow'
    robot_id: int          # 1-10, from COLOR_PAIR_TO_ID
    position: tuple[float, float]  # (x_cm, y_cm) in field coords
    angle_deg: float       # [0, 360), 0 = +x axis


@dataclass
class BallDetection:
    """Single ball detection in one frame."""
    position: tuple[float, float]  # (x_cm, y_cm) in field coords


@dataclass
class FrameResult:
    """Detection results for a single video frame."""
    frame_index: int
    detections: list[RobotDetection] = field(default_factory=list)
    ball: Optional[BallDetection] = None


@dataclass
class TrackerMetrics:
    """Performance metrics collected during tracking iteration."""
    fps: float = 0.0
    total_frames: int = 0
    skipped_frames: int = 0
    detection_rate: float = 0.0
    team_detection_counts: dict[str, int] = field(default_factory=dict)
    ball_detection_rate: float = 0.0
    total_processing_time: float = 0.0

    def summary(self) -> str:
        lines = [
            f"FPS: {self.fps:.2f}",
            f"Total frames: {self.total_frames}",
            f"Skipped frames: {self.skipped_frames}",
            f"Detection rate: {self.detection_rate:.2%}",
            f"Ball detection rate: {self.ball_detection_rate:.2%}",
            f"Team detection counts: {self.team_detection_counts}",
            f"Total processing time: {self.total_processing_time:.2f}s",
        ]
        return "\n".join(lines)


# Sorted color-pair -> numeric robot ID lookup.
# 4 ID colors taken 2 at a time = C(4,2) = 6 combinations, plus 4 same-color pairs = 10.
# Colors are always sorted alphabetically in the tuple key.
COLOR_PAIR_TO_ID: dict[tuple[str, str], int] = {
    ("cyan", "cyan"):     1,
    ("cyan", "green"):    2,
    ("cyan", "purple"):   3,
    ("cyan", "red"):      4,
    ("green", "green"):   5,
    ("green", "purple"):  6,
    ("green", "red"):     7,
    ("purple", "purple"): 8,
    ("purple", "red"):    9,
    ("red", "red"):       10,
}
