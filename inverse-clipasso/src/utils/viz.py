"""
Visualization utilities for qualitative inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def save_before_after_grid(
    sketches: Iterable[np.ndarray],
    labels: Iterable[str],
    path: Path,
    *,
    cols: int = 4,
) -> None:
    """
    Save a simple grid comparing input and optimized sketches.
    """

    raise NotImplementedError("Implement matplotlib grid export.")


def edge_difference(before: np.ndarray, after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return edge maps and their absolute difference for debugging.
    """

    raise NotImplementedError("Implement edge comparison utility.")
