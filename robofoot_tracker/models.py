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
    robot_id: int          # 0-9, from COLOR_PAIR_TO_ID
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
    count_error_rate: float | None = None
    id_recall: float | None = None
    id_precision: float | None = None
    angle_jitter: float | None = None

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
        if self.count_error_rate is not None:
            lines.append(f"Count error rate: {self.count_error_rate:.2%}")
        if self.id_recall is not None:
            lines.append(f"ID recall: {self.id_recall:.2%}")
        if self.id_precision is not None:
            lines.append(f"ID precision: {self.id_precision:.2%}")
        if self.angle_jitter is not None:
            lines.append(f"Angle jitter: {self.angle_jitter:.2f} deg")
        return "\n".join(lines)


# Robot-centric color-pair -> numeric robot ID lookup.
# 10 entries: ordered pairs based on robot-centric left/right convention.
# "First" color = LEFT of robot's forward axis.
COLOR_PAIR_TO_ID: dict[tuple[str, str], int] = {
    # Ordered different-color pairs (10). First color = LEFT of robot's forward axis.
    ("green", "red"):     0,
    ("cyan", "red"):      1,
    ("red", "green"):     2,
    ("cyan", "green"):    3,
    ("purple", "green"):  4,
    ("red", "cyan"):      5,
    ("green", "cyan"):    6,
    ("purple", "cyan"):   7,
    ("green", "purple"):  8,
    ("cyan", "purple"):   9,
}
