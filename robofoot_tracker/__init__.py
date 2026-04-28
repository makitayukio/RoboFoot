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
"""robofoot_tracker — Robot football position and orientation tracker."""

from .tracker import Tracker
from .models import (
    CalibrationData,
    FieldDimensions,
    FrameResult,
    RobotDetection,
    BallDetection,
    TrackerMetrics,
    COLOR_PAIR_TO_ID,
)
from .color_config import ColorConfig, TEAM_COLORS, ID_COLORS
from .calibration import calibrate_from_points, calibrate_interactive, calibrate_colors_interactive, estimate_distortion, transform_point
from .detector import detect_robots, detect_ball
from .preprocessing import preprocess_frame
from .viz import draw_detections
from .pose_recorder import PoseRecorder
from .pose_animation import render_pose_animation, render_side_by_side_video

__all__ = [
    "Tracker",
    "CalibrationData",
    "FieldDimensions",
    "FrameResult",
    "RobotDetection",
    "BallDetection",
    "TrackerMetrics",
    "COLOR_PAIR_TO_ID",
    "ColorConfig",
    "TEAM_COLORS",
    "ID_COLORS",
    "calibrate_from_points",
    "calibrate_interactive",
    "calibrate_colors_interactive",
    "estimate_distortion",
    "transform_point",
    "detect_robots",
    "detect_ball",
    "preprocess_frame",
    "draw_detections",
    "PoseRecorder",
    "render_pose_animation",
    "render_side_by_side_video",
]
