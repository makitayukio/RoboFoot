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
            self.calibration = calibrate_interactive(first, self.field_dims)

        if self._color_calibration:
            ok, color_frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, color_frame = cap.read()
            if ok:
                self.color_config = calibrate_colors_interactive(color_frame)

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
        dets = detect_robots(f, self.calibration, self.color_config, teams=self._teams)
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
        return f"RoboFoot Tracker - {team_info}"

    def run_live(self) -> None:
        """Run real-time detection loop with cv2 display. Camera mode only."""
        if self._camera is None:
            raise RuntimeError("run_live() requires camera mode — create Tracker with camera=<index>")

        self._metrics = TrackerMetrics()
        self._frames_with_detections = 0
        self._frames_with_ball = 0
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
                self._accumulate(dets, ball)
                annotated = draw_detections(frame, dets, self.calibration, ball)
                cv2.imshow(title, annotated)

                poses = []

                for det in dets:
                    pose = [det.position[0], det.position[1], det.angle_deg]
                    poses.append(pose)

                print(poses)

                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
        finally:
            self._finalize_metrics(start)
            cap.release()
            cv2.destroyAllWindows()

    def _accumulate(self, dets: list, ball: object) -> None:
        """Accumulate detection stats into _metrics."""
        if dets:
            self._frames_with_detections += 1
            for d in dets:
                self._metrics.team_detection_counts[d.team] = (
                    self._metrics.team_detection_counts.get(d.team, 0) + 1
                )
        if ball is not None:
            self._frames_with_ball += 1

    def _finalize_metrics(self, start: float) -> None:
        """Compute derived metrics from accumulated counters."""
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

    def __iter__(self) -> Iterator[FrameResult]:
        self._metrics = TrackerMetrics()
        self._frames_with_detections = 0
        self._frames_with_ball = 0
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
                self._accumulate(dets, ball)
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
                self._accumulate(dets, ball)
                yield FrameResult(frame_index=idx, detections=dets, ball=ball)
                idx += 1

            if skipped:
                logger.info("Skipped %d blank/dark frames", skipped)
        finally:
            self._finalize_metrics(start)
            cap.release()
