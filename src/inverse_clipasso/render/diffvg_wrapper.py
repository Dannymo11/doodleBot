"""
Thin wrapper around diffvg for converting parametric curves to images.
"""

from __future__ import annotations

from typing import Any, Tuple


def render_svg(paths: Any, *, canvas_size: Tuple[int, int], samples: int = 64) -> Any:
    """
    Render control points / stroke parameters into an image tensor.
    """

    raise NotImplementedError("Hook diffvg rendering here.")
