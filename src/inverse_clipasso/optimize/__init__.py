"""Optimization routines."""

from .inverse_clipasso import InverseClipassoOptimizer, OptimizationConfig, OptimizationResult
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
