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
"""CSV pose recorder for robot and ball detections."""

from __future__ import annotations

import csv
from typing import TextIO

_HEADER = ["frame_index", "time_s", "kind", "team", "robot_id", "x_cm", "y_cm", "angle_deg"]


class PoseRecorder:
    """Write per-frame robot/ball poses to a CSV file."""

    def __init__(self, path: str) -> None:
        self._file: TextIO | None = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_HEADER)
        self._file.flush()

    def record(self, frame_index: int, time_s: float, detections: list, ball) -> None:
        if self._file is None:
            return
        for d in detections:
            self._writer.writerow([
                frame_index, f"{time_s:.4f}", "robot",
                d.team, d.robot_id,
                f"{d.position[0]:.2f}", f"{d.position[1]:.2f}",
                f"{d.angle_deg:.2f}",
            ])
        if ball is not None:
            self._writer.writerow([
                frame_index, f"{time_s:.4f}", "ball",
                "", -1,
                f"{ball.position[0]:.2f}", f"{ball.position[1]:.2f}",
                "",
            ])
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
