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
"""Frame preprocessing with CLAHE and Gaussian blur."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_frame(
    frame: np.ndarray,
    clahe_clip_limit: float = 1.5,
    clahe_grid_size: tuple[int, int] = (8, 8),
    gaussian_ksize: int = 3,
) -> np.ndarray:
    """Apply CLAHE on the V channel and Gaussian blur to a BGR frame.

    Args:
        frame: BGR image (H×W×3, uint8).
        clahe_clip_limit: CLAHE clip limit (default 1.5, conservative for orange/red safety).
        clahe_grid_size: CLAHE tile grid size.
        gaussian_ksize: Gaussian blur kernel size (must be odd and positive).

    Returns:
        Preprocessed BGR frame, same shape and dtype as input.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.GaussianBlur(bgr, (gaussian_ksize, gaussian_ksize), 0)
