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
"""Main Tracker API — iterates video frames and yields RobotDetection results."""

from __future__ import annotations

import logging
import time
from typing import Iterator

import cv2
import numpy as np

from .calibration import calibrate_from_points, calibrate_interactive, calibrate_colors_interactive, CalibrationData
from .color_config import ColorConfig
from .detector import detect_robots, detect_ball
from .models import FieldDimensions, FrameResult, TrackerMetrics
from .preprocessing import preprocess_frame
from .viz import draw_detections

logger = logging.getLogger(__name__)

WINDOW_SIZE = 15
POSITION_STD_THRESHOLD = 1.0  # cm


class Tracker:
    """Public API for robot tracking on a video file or live camera.

    Usage::

        tracker = Tracker("video.mp4")
        for result in tracker:
            print(result.frame_index, result.detections)

        # Live camera mode:
        tracker = Tracker(camera=0)
        tracker.run_live()
    """

    def __init__(
        self,
        video_path: str | None = None,
        field_dims: FieldDimensions | None = None,
        calibration_points: list[tuple[float, float]] | None = None,
        blank_threshold: float = 5.0,
        teams: str | list[str] | None = "both",
        preprocessing: bool = False,
        clahe_clip_limit: float = 1.5,
        clahe_grid_size: tuple[int, int] = (8, 8),
        gaussian_ksize: int = 3,
        camera: int | None = None,
        color_calibration: bool = False,
        expected_count: int | None = None,
        expected_ids: list[int] | None = None,
        watershed: bool = False,
        record_poses: str | None = None,
    ) -> None:
        if video_path is None and camera is None:
            raise ValueError("Exactly one of video_path or camera must be provided")
        if video_path is not None and camera is not None:
            raise ValueError("Exactly one of video_path or camera must be provided")

        self.video_path = video_path
        self._camera = camera
        self.field_dims = field_dims or FieldDimensions()
        self.color_config = ColorConfig()
        self.calibration: CalibrationData | None = None
        self.blank_threshold = blank_threshold
        self._calibration_points = calibration_points
        self._preprocessing = preprocessing
        self._clahe_clip_limit = clahe_clip_limit
        self._clahe_grid_size = clahe_grid_size
        self._gaussian_ksize = gaussian_ksize
        self._color_calibration = color_calibration
        self._expected_count = expected_count
        self._expected_ids = expected_ids
        self._watershed = watershed
        self._record_poses = record_poses
        self._pose_recorder: PoseRecorder | None = None

        # Normalize teams: 'both'/None -> None (all), str -> [str], list -> as-is
        if teams is None or teams == "both":
            self._teams: list[str] | None = None
        elif isinstance(teams, str):
            self._teams = [teams]
        else:
            self._teams = list(teams)

        self._metrics = TrackerMetrics()

    @property
    def metrics(self) -> TrackerMetrics:
        return self._metrics

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing if enabled, otherwise return frame as-is."""
        if self._preprocessing:
            return preprocess_frame(frame, self._clahe_clip_limit, self._clahe_grid_size, self._gaussian_ksize)
        return frame

    def _calibrate_first_frame(self, cap: cv2.VideoCapture) -> None:
        """Read first frame and calibrate if not already calibrated."""
        if self.calibration is not None:
            return
        ok, first = cap.read()
        if not ok:
            raise RuntimeError("Cannot read first frame for calibration")
        if self._calibration_points:
            self.calibration = calibrate_from_points(self._calibration_points, self.field_dims)
        else:
            if self.video_path is not None:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ok, mid = cap.read()
                if ok and np.mean(mid) >= self.blank_threshold:
                    first = mid
            while np.mean(first) < self.blank_threshold:
                ok, first = cap.read()
                if not ok:
                    raise RuntimeError("All frames are dark — cannot calibrate interactively")
            self.calibration = calibrate_interactive(first, self.field_dims, cap=cap if self._camera is not None else None)

        if self._color_calibration:
            ok, color_frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, color_frame = cap.read()
            if ok:
                self.color_config = calibrate_colors_interactive(color_frame, cap=cap if self._camera is not None else None)

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply lens undistortion if calibration has distortion data."""
        if (self.calibration is not None
                and self.calibration.camera_matrix is not None
                and self.calibration.dist_coeffs is not None):
            return cv2.undistort(frame, self.calibration.camera_matrix, self.calibration.dist_coeffs)
        return frame

    def _detect(self, frame: np.ndarray) -> tuple[list, object]:
        """Run detection pipeline on a single frame."""
        f = self._preprocess(frame)
        dets = detect_robots(f, self.calibration, self.color_config, teams=self._teams, watershed=self._watershed)
        ball = detect_ball(f, self.calibration, self.color_config)
        return dets, ball

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Detect robots and ball on a single BGR frame. Requires calibration to be set."""
        if self.calibration is None:
            raise RuntimeError("Calibration not set — iterate the tracker or call calibrate first")
        dets, ball = self._detect(frame)
        return FrameResult(frame_index=-1, detections=dets, ball=ball)

    def _open_camera(self, index: int) -> cv2.VideoCapture:
        """Open a camera with the default backend."""
        return cv2.VideoCapture(index)

    def _build_window_title(self) -> str:
        """Build the cv2 window title from team configuration."""
        if self._teams is None:
            team_info = "All Teams"
        elif len(self._teams) == 1:
            team_info = self._teams[0].capitalize()
        else:
            team_info = " & ".join(t.capitalize() for t in sorted(self._teams))
        return f"Grupo 3 - RoboFoot Tracker - {team_info}"

    def run_live(self) -> None:
        """Run real-time detection loop with cv2 display. Camera mode only."""
        if self._camera is None:
            raise RuntimeError("run_live() requires camera mode — create Tracker with camera=<index>")

        self._metrics = TrackerMetrics()
        self._frames_with_detections = 0
        self._frames_with_ball = 0
        self._count_error_sum = 0.0
        self._id_recall_sum = 0.0
        self._id_precision_sum = 0.0
        self._accuracy_frames = 0
        self._angle_observations: dict[int, list[tuple[int, float, float, float]]] = {}
        if self._record_poses:
            from .pose_recorder import PoseRecorder
            self._pose_recorder = PoseRecorder(self._record_poses)
        cap = self._open_camera(self._camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._camera}. Check device index and permissions.")
        title = self._build_window_title()
        start = time.perf_counter()
        try:
            self._calibrate_first_frame(cap)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self._metrics.total_frames += 1
                frame = self.undistort_frame(frame)
                # Show raw frame immediately to keep window responsive
                cv2.imshow(title, frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
                # Now run detection (slow) and update with annotations
                dets, ball = self._detect(frame)
                self._accumulate(dets, ball, self._metrics.total_frames - 1, time_s=time.perf_counter() - start)
                annotated = draw_detections(frame, dets, self.calibration, ball)
                cv2.imshow(title, annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
        finally:
            self._finalize_metrics(start)
            cap.release()
            cv2.destroyAllWindows()

    def _accumulate(self, dets: list, ball: object, frame_index: int = 0, time_s: float = 0.0) -> None:
        """Accumulate detection stats into _metrics."""
        if dets:
            self._frames_with_detections += 1
            for d in dets:
                self._metrics.team_detection_counts[d.team] = (
                    self._metrics.team_detection_counts.get(d.team, 0) + 1
                )
        if ball is not None:
            self._frames_with_ball += 1

        if self._expected_count is not None and self._expected_ids is not None:
            self._accuracy_frames += 1
            self._count_error_sum += abs(len(dets) - self._expected_count) / self._expected_count
            detected_ids = {d.robot_id for d in dets}
            expected_set = set(self._expected_ids)
            intersection = len(detected_ids & expected_set)
            self._id_recall_sum += intersection / len(expected_set)
            self._id_precision_sum += intersection / max(len(detected_ids), 1)

        for d in dets:
            self._angle_observations.setdefault(d.robot_id, []).append(
                (frame_index, d.position[0], d.position[1], d.angle_deg)
            )

        if getattr(self, "_pose_recorder", None) is not None:
            self._pose_recorder.record(frame_index, time_s, dets, ball)

    def _finalize_metrics(self, start: float) -> None:
        """Compute derived metrics from accumulated counters."""
        if getattr(self, "_pose_recorder", None) is not None:
            self._pose_recorder.close()
            self._pose_recorder = None
        elapsed = time.perf_counter() - start
        self._metrics.total_processing_time = elapsed
        processed = self._metrics.total_frames - self._metrics.skipped_frames
        self._metrics.fps = processed / elapsed if elapsed > 0 else 0.0
        self._metrics.detection_rate = (
            self._frames_with_detections / processed if processed > 0 else 0.0
        )
        self._metrics.ball_detection_rate = (
            self._frames_with_ball / processed if processed > 0 else 0.0
        )
        if self._accuracy_frames > 0:
            self._metrics.count_error_rate = self._count_error_sum / self._accuracy_frames
            self._metrics.id_recall = self._id_recall_sum / self._accuracy_frames
            self._metrics.id_precision = self._id_precision_sum / self._accuracy_frames

        # Stationary-window angle jitter (circular std-dev, Mardia formula)
        per_window_jitters: list[float] = []
        for obs in self._angle_observations.values():
            if len(obs) < WINDOW_SIZE:
                continue
            obs_sorted = sorted(obs, key=lambda t: t[0])
            for i in range(len(obs_sorted) - WINDOW_SIZE + 1):
                window = obs_sorted[i:i + WINDOW_SIZE]
                f0 = window[0][0]
                if window[-1][0] - f0 != WINDOW_SIZE - 1:
                    continue
                xs = np.array([w[1] for w in window])
                ys = np.array([w[2] for w in window])
                if max(np.std(xs), np.std(ys)) >= POSITION_STD_THRESHOLD:
                    continue
                angles = np.radians([w[3] for w in window])
                mean_cos = np.mean(np.cos(angles))
                mean_sin = np.mean(np.sin(angles))
                R = np.sqrt(mean_cos**2 + mean_sin**2)
                if R >= 1.0:
                    per_window_jitters.append(0.0)
                elif R > 0.0:
                    sigma_rad = np.sqrt(-2.0 * np.log(R))
                    per_window_jitters.append(np.degrees(sigma_rad))
        if per_window_jitters:
            self._metrics.angle_jitter = float(np.mean(per_window_jitters))

    def __iter__(self) -> Iterator[FrameResult]:
        self._metrics = TrackerMetrics()
        self._frames_with_detections = 0
        self._frames_with_ball = 0
        self._count_error_sum = 0.0
        self._id_recall_sum = 0.0
        self._id_precision_sum = 0.0
        self._accuracy_frames = 0
        self._angle_observations: dict[int, list[tuple[int, float, float, float]]] = {}
        if self._record_poses:
            from .pose_recorder import PoseRecorder
            self._pose_recorder = PoseRecorder(self._record_poses)
        if self._camera is not None:
            yield from self._iter_camera()
        else:
            yield from self._iter_video()

    def _iter_camera(self) -> Iterator[FrameResult]:
        cap = self._open_camera(self._camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._camera}. Check device index and permissions.")
        start = time.perf_counter()
        try:
            self._calibrate_first_frame(cap)

            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self._metrics.total_frames += 1
                frame = self.undistort_frame(frame)
                dets, ball = self._detect(frame)
                self._accumulate(dets, ball, idx, time_s=time.perf_counter() - start)
                yield FrameResult(frame_index=idx, detections=dets, ball=ball)
                idx += 1
        finally:
            self._finalize_metrics(start)
            cap.release()

    def _iter_video(self) -> Iterator[FrameResult]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Opened video %s — %dx%d, %d frames", self.video_path, w, h, total)

        start = time.perf_counter()
        try:
            self._calibrate_first_frame(cap)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            idx = 0
            skipped = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self._metrics.total_frames += 1
                if np.mean(frame) < self.blank_threshold:
                    skipped += 1
                    self._metrics.skipped_frames += 1
                    idx += 1
                    continue
                frame = self.undistort_frame(frame)
                dets, ball = self._detect(frame)
                self._accumulate(dets, ball, idx, time_s=time.perf_counter() - start)
                yield FrameResult(frame_index=idx, detections=dets, ball=ball)
                idx += 1

            if skipped:
                logger.info("Skipped %d blank/dark frames", skipped)
        finally:
            self._finalize_metrics(start)
            cap.release()
