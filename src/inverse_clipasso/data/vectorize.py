"""
Tools for converting raster sketches into vector primitives.
"""

from __future__ import annotations

from typing import Any, Tuple


def fit_splines(bitmap: Any, *, tolerance: float = 2.0) -> Tuple[Any, Any]:
    """
    Fit cubic Bézier splines to a binary bitmap using potrace-like logic.
    """

    raise NotImplementedError("Implement spline fitting over binarized input.")
