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
"""Shared geometry utilities."""

import numpy as np


def bezier_curve_points(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, num_points: int = 20,
) -> np.ndarray:
    """Quadratic Bézier where *p1* is a pass-through point (on-curve at t=0.5).

    The actual Bézier control point is derived so the curve interpolates p1:
        control = 2·p1 − 0.5·(p0 + p2)
    """
    control = 2 * p1 - 0.5 * (p0 + p2)
    t = np.linspace(0, 1, num_points).reshape(-1, 1)
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t ** 2 * p2
    return pts.astype(np.int32)
