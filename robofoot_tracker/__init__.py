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
]
