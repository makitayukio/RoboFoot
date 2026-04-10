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
