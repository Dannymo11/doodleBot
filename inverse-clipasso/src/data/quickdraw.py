"""
QuickDraw loaders for raster and vector sketches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def load_raster_samples(root: Path, labels: Iterable[str]) -> np.ndarray:
    """
    Load rasterized sketches given a root directory and label filter.
    """

    raise NotImplementedError("Implement QuickDraw raster loading.")


def load_vector_samples(root: Path, labels: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load vector stroke sequences along with timing metadata.
    """

    raise NotImplementedError("Implement QuickDraw vector loading.")
