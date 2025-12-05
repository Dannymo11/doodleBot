"""Rendering backends."""

from .diffvg_wrapper import render_strokes_soft, render_strokes_rgb, render_svg
from .bezier import (
    BezierStroke,
    BezierSketch,
    render_bezier_strokes,
    initialize_bezier_from_edges,
)
