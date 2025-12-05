"""Optimization routines."""

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
