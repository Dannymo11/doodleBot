"""
Experiment utilities for sketch optimization research.

This module contains:
- Data loading utilities for QuickDraw and photos
- Visualization helpers for optimization results
- Experiment runners for long optimization sessions
"""

from .data_loaders import (
    load_quickdraw_exemplars,
    load_reference_image,
    load_photo_exemplars,
    quickdraw_to_stroke_params,
)
from .visualization import (
    render_to_pil,
    create_progression_grid,
    create_gif,
    visualize_full_results,
)
from .runners import (
    run_long_optimization,
    get_best_device,
)

__all__ = [
    # Data loaders
    "load_quickdraw_exemplars",
    "load_reference_image",
    "load_photo_exemplars",
    "quickdraw_to_stroke_params",
    # Visualization
    "render_to_pil",
    "create_progression_grid",
    "create_gif",
    "visualize_full_results",
    # Runners
    "run_long_optimization",
    "get_best_device",
]


