"""HSV color configuration for robot tag detection."""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np

TEAM_COLORS: list[str] = ["blue", "yellow"]
ID_COLORS: list[str] = ["red", "green", "cyan", "purple"]

# Each color maps to a list of (lower_hsv, upper_hsv) tuples.
# Red needs two ranges because hue wraps around 0/180 in OpenCV.
_DEFAULT_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "blue": [
        (np.array([100, 80, 80]), np.array([130, 255, 255])),
    ],
    "yellow": [
        (np.array([20, 80, 80]), np.array([55, 255, 255])),
    ],
    "red": [
        (np.array([0, 80, 80]), np.array([10, 255, 255])),
        (np.array([170, 80, 80]), np.array([180, 255, 255])),
    ],
    "green": [
        (np.array([36, 80, 80]), np.array([85, 255, 255])),
    ],
    "cyan": [
        (np.array([85, 80, 80]), np.array([100, 255, 255])),
    ],
    "purple": [
        (np.array([130, 80, 80]), np.array([170, 255, 255])),
    ],
    "orange": [
        (np.array([0, 200, 200]), np.array([10, 255, 255])),
    ],
}


class ColorConfig:
    """Tunable HSV color ranges for tag detection."""

    def __init__(self) -> None:
        self._ranges: dict[str, list[tuple[np.ndarray, np.ndarray]]] = deepcopy(
            _DEFAULT_RANGES
        )

    def get_ranges(self, color_name: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (lower, upper) HSV range tuples for *color_name*."""
        if color_name not in self._ranges:
            raise KeyError(f"Unknown color: {color_name}")
        return self._ranges[color_name]

    def set_range(
        self,
        color_name: str,
        ranges: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Override HSV ranges for *color_name*."""
        self._ranges[color_name] = list(ranges)
