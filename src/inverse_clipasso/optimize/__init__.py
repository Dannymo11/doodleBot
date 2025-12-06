"""
Optimization routines for CLIP-guided sketch refinement.

This module provides:
- InverseClipassoOptimizer: Main optimizer for sketch refinement
- CLIPassoOptimizer: CLIPasso-style optimizer with Bézier curves
- Loss functions for semantic, perceptual, and geometric optimization
- Augmentation utilities for robust CLIP guidance
- Common utilities for learning rate scheduling, normalization, etc.
"""

from .inverse_clipasso import InverseClipassoOptimizer, OptimizationConfig, OptimizationResult
from .clipasso_optimizer import CLIPassoOptimizer, CLIPassoConfig, CLIPassoResult
from .losses import (
    semantic_loss, 
    style_loss, 
    stroke_regularizer, 
    total_variation, 
    aggregate_losses,
    original_strokes_loss,
    exemplar_loss,
    reference_image_loss,
)
from .augmentations import (
    CLIPAugmentations,
    DifferentiableAugmentations,
    get_augmenter,
)
from .perceptual_loss import (
    CLIPFeatureExtractor,
    MultiScalePerceptualLoss,
    CLIPassoLoss,
    compute_clip_loss_with_augmentations,
)
from .style_loss import (
    VGGStyleLoss,
    VGGFeatureExtractor,
)
from .common import (
    # Constants
    CLIP_MEAN,
    CLIP_STD,
    STROKE_RADIUS_MIN,
    STROKE_RADIUS_MAX,
    SINGLE_POINT_OFFSET,
    STROKE_INIT_PADDING,
    # Type definitions
    OptimizationCallback,
    # Classes
    CLIPNormalizer,
    # Learning rate utilities
    get_lr_multiplier,
    get_scheduled_weight,
    # Bézier utilities
    sample_bezier_curve,
    polyline_to_bezier,
    polyline_to_bezier_segments,
    # Stroke initialization
    create_random_stroke,
    # Loss utilities
    compute_weighted_loss,
    # Gradient utilities
    add_gradient_noise,
    clip_gradients,
    # Device utilities
    get_device,
    ensure_device,
)

__all__ = [
    # Optimizers
    "InverseClipassoOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "CLIPassoOptimizer",
    "CLIPassoConfig",
    "CLIPassoResult",
    # Loss functions
    "semantic_loss",
    "style_loss",
    "stroke_regularizer",
    "total_variation",
    "aggregate_losses",
    "original_strokes_loss",
    "exemplar_loss",
    "reference_image_loss",
    # Augmentations
    "CLIPAugmentations",
    "DifferentiableAugmentations",
    "get_augmenter",
    # Perceptual loss
    "CLIPFeatureExtractor",
    "MultiScalePerceptualLoss",
    "CLIPassoLoss",
    "compute_clip_loss_with_augmentations",
    # Style loss (VGG-based)
    "VGGStyleLoss",
    "VGGFeatureExtractor",
    # Common utilities
    "CLIP_MEAN",
    "CLIP_STD",
    "STROKE_RADIUS_MIN",
    "STROKE_RADIUS_MAX",
    "SINGLE_POINT_OFFSET",
    "STROKE_INIT_PADDING",
    "OptimizationCallback",
    "CLIPNormalizer",
    "get_lr_multiplier",
    "get_scheduled_weight",
    "sample_bezier_curve",
    "polyline_to_bezier",
    "polyline_to_bezier_segments",
    "create_random_stroke",
    "compute_weighted_loss",
    "add_gradient_noise",
    "clip_gradients",
    "get_device",
    "ensure_device",
]
