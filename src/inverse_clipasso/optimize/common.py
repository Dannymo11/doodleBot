"""
Common utilities shared between optimization algorithms.

This module extracts duplicated code from InverseClipassoOptimizer and CLIPassoOptimizer
to reduce code duplication and ensure consistent behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

import torch
import torch.nn.functional as F


# ============================================================================
# CLIP Constants
# ============================================================================

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


# ============================================================================
# Stroke Initialization Constants
# ============================================================================

# Default radius range for randomly initialized strokes
STROKE_RADIUS_MIN = 15.0
STROKE_RADIUS_MAX = 35.0

# Default offset for single-point to curve conversion
SINGLE_POINT_OFFSET = 10.0

# Canvas padding for stroke initialization
STROKE_INIT_PADDING = 20


# ============================================================================
# Type Definitions
# ============================================================================

class OptimizationCallback(Protocol):
    """Protocol for optimization callback functions."""
    
    def __call__(
        self,
        step: int,
        losses: Dict[str, float],
        strokes: List[torch.Tensor],
    ) -> None:
        """
        Called at each optimization step.
        
        Args:
            step: Current optimization step.
            losses: Dictionary of loss values.
            strokes: Current stroke parameters.
        """
        ...


T = TypeVar('T', bound=torch.Tensor)


# ============================================================================
# CLIP Normalization
# ============================================================================

class CLIPNormalizer:
    """
    Handles CLIP-specific image normalization.
    
    Caches normalization tensors on the correct device for efficiency.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
    
    def _ensure_tensors(self, device: torch.device) -> None:
        """Ensure normalization tensors are on the correct device."""
        if self._mean is None or self._mean.device != device:
            self._mean = CLIP_MEAN.view(1, 3, 1, 1).to(device)
            self._std = CLIP_STD.view(1, 3, 1, 1).to(device)
    
    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply CLIP normalization to an image.
        
        Args:
            image: Image tensor [B, 3, H, W] or [3, H, W].
        
        Returns:
            Normalized image tensor.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        self._ensure_tensors(image.device)
        return (image - self._mean) / self._std
    
    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Remove CLIP normalization from an image.
        
        Args:
            image: Normalized image tensor.
        
        Returns:
            Denormalized image tensor [0, 1] range.
        """
        self._ensure_tensors(image.device)
        return image * self._std + self._mean


# ============================================================================
# Learning Rate Schedules
# ============================================================================

def get_lr_multiplier(
    step: int,
    total_steps: int,
    schedule: str = "cosine",
    warmup_steps: int = 0,
    min_lr_factor: float = 0.01,
) -> float:
    """
    Compute learning rate multiplier for the current step.
    
    Args:
        step: Current optimization step.
        total_steps: Total number of steps.
        schedule: Schedule type ('constant', 'linear', 'cosine', 'warmup_cosine').
        warmup_steps: Number of warmup steps (for warmup_cosine).
        min_lr_factor: Minimum LR as fraction of base LR.
    
    Returns:
        Learning rate multiplier in range [min_lr_factor, 1.0].
    """
    if schedule == "constant":
        return 1.0
    
    elif schedule == "linear":
        progress = step / max(total_steps - 1, 1)
        return 1.0 + (min_lr_factor - 1.0) * progress
    
    elif schedule == "cosine":
        progress = step / max(total_steps - 1, 1)
        return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress))
    
    elif schedule == "warmup_cosine":
        if step < warmup_steps:
            # Linear warmup
            return min_lr_factor + (1 - min_lr_factor) * (step / warmup_steps)
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
            return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress))
    
    else:
        raise ValueError(f"Unknown lr_schedule: {schedule}")


def get_scheduled_weight(
    step: int,
    total_steps: int,
    start_weight: float,
    end_weight: float,
    schedule: str = "constant",
) -> float:
    """
    Compute a scheduled weight value (e.g., for original_weight).
    
    Args:
        step: Current step.
        total_steps: Total steps.
        start_weight: Starting weight value.
        end_weight: Ending weight value.
        schedule: 'constant', 'linear_up', 'cosine_up'.
    
    Returns:
        Weight value for current step.
    """
    if schedule == "constant":
        return end_weight
    
    progress = step / max(total_steps - 1, 1)
    
    if schedule == "linear_up":
        return start_weight + (end_weight - start_weight) * progress
    
    elif schedule == "cosine_up":
        # Cosine from 0 to 1
        cosine_progress = (1 - math.cos(math.pi * progress)) / 2
        return start_weight + (end_weight - start_weight) * cosine_progress
    
    else:
        return end_weight


# ============================================================================
# Bézier Curve Utilities
# ============================================================================

def sample_bezier_curve(
    control_points: torch.Tensor,
    num_samples: int = 50,
) -> torch.Tensor:
    """
    Sample points along a cubic Bézier curve.
    
    Args:
        control_points: [4, 2] tensor of control points P0, P1, P2, P3.
        num_samples: Number of points to sample.
    
    Returns:
        [num_samples, 2] tensor of sampled points.
    """
    device = control_points.device
    t = torch.linspace(0, 1, num_samples, device=device).unsqueeze(1)  # [N, 1]
    
    P0, P1, P2, P3 = control_points[0], control_points[1], control_points[2], control_points[3]
    
    # Cubic Bézier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    points = (
        (1 - t) ** 3 * P0 +
        3 * (1 - t) ** 2 * t * P1 +
        3 * (1 - t) * t ** 2 * P2 +
        t ** 3 * P3
    )
    
    return points


def polyline_to_bezier(polyline: torch.Tensor) -> torch.Tensor:
    """
    Convert a polyline to cubic Bézier control points.
    
    Args:
        polyline: [N, 2] tensor of polyline points.
    
    Returns:
        [4, 2] tensor of Bézier control points.
    """
    n_points = polyline.shape[0]
    
    if n_points >= 4:
        # Sample 4 evenly-spaced points
        indices = torch.linspace(0, n_points - 1, 4).long()
        return polyline[indices]
    
    elif n_points == 3:
        # Duplicate middle point
        return torch.stack([
            polyline[0],
            polyline[1],
            polyline[1],
            polyline[2],
        ])
    
    elif n_points == 2:
        # Create control points between endpoints
        p0, p1 = polyline[0], polyline[1]
        return torch.stack([
            p0,
            p0 + (p1 - p0) * 0.33,
            p0 + (p1 - p0) * 0.67,
            p1,
        ])
    
    else:
        # Single point - create small curve
        p = polyline[0]
        offset = torch.tensor(
            [SINGLE_POINT_OFFSET, SINGLE_POINT_OFFSET], 
            device=polyline.device
        )
        return torch.stack([
            p - offset,
            p - offset * 0.5,
            p + offset * 0.5,
            p + offset,
        ])


def polyline_to_bezier_segments(
    polyline: torch.Tensor,
    max_points_per_segment: int = 6,
) -> List[torch.Tensor]:
    """
    Convert a polyline to multiple cubic Bézier curves for better fidelity.
    
    Instead of compressing an entire stroke into a single 4-point Bézier,
    this splits long strokes into multiple segments, preserving more detail.
    
    Args:
        polyline: [N, 2] tensor of polyline points.
        max_points_per_segment: Maximum points to use for fitting each Bézier.
            Smaller values = more segments = higher fidelity but more parameters.
    
    Returns:
        List of [4, 2] Bézier control point tensors.
    """
    n_points = polyline.shape[0]
    
    # Short polylines can use a single Bézier
    if n_points <= 4:
        return [polyline_to_bezier(polyline)]
    
    # Split into overlapping segments for continuity
    # Each segment shares its last point with the next segment's first point
    step = max(1, max_points_per_segment - 1)
    segments = []
    
    start = 0
    while start < n_points - 1:
        end = min(start + max_points_per_segment, n_points)
        segment_points = polyline[start:end]
        
        if segment_points.shape[0] >= 2:
            bezier = polyline_to_bezier(segment_points)
            segments.append(bezier)
        
        # Move to next segment, overlapping by 1 point for continuity
        start = end - 1
        
        # Avoid infinite loop for edge cases
        if start >= n_points - 2:
            break
    
    return segments if segments else [polyline_to_bezier(polyline)]


# ============================================================================
# Stroke Initialization
# ============================================================================

def create_random_stroke(
    canvas_size: int,
    num_points: int = 4,
    mode: str = "random",
    device: str = "cpu",
    padding: int = STROKE_INIT_PADDING,
) -> torch.Tensor:
    """
    Create a new stroke with random initialization.
    
    Args:
        canvas_size: Size of the canvas.
        num_points: Number of control points.
        mode: Initialization mode ('random', 'center', 'edge').
        device: Device for the tensor.
        padding: Padding from canvas edges.
    
    Returns:
        [num_points, 2] tensor of stroke points.
    """
    if mode == "random":
        # Random position and shape
        center_x = torch.rand(1).item() * (canvas_size - 2 * padding) + padding
        center_y = torch.rand(1).item() * (canvas_size - 2 * padding) + padding
        
        # Create a small stroke around the center
        angles = torch.linspace(0, 2 * math.pi, num_points + 1)[:-1]
        radius = STROKE_RADIUS_MIN + torch.rand(1).item() * (STROKE_RADIUS_MAX - STROKE_RADIUS_MIN)
        
        points = torch.zeros(num_points, 2)
        for i, angle in enumerate(angles):
            r = radius * (0.8 + 0.4 * torch.rand(1).item())
            points[i, 0] = center_x + r * torch.cos(angle)
            points[i, 1] = center_y + r * torch.sin(angle)
    
    elif mode == "center":
        # Start from center of canvas
        center_x = canvas_size / 2
        center_y = canvas_size / 2
        
        points = torch.zeros(num_points, 2)
        for i in range(num_points):
            points[i, 0] = center_x + (i - num_points / 2) * 10 + torch.rand(1).item() * 5
            points[i, 1] = center_y + torch.rand(1).item() * 20 - 10
    
    elif mode == "edge":
        # Start from a random edge
        edge = torch.randint(0, 4, (1,)).item()
        
        if edge == 0:  # Top
            start_x = torch.rand(1).item() * canvas_size
            start_y = padding
        elif edge == 1:  # Right
            start_x = canvas_size - padding
            start_y = torch.rand(1).item() * canvas_size
        elif edge == 2:  # Bottom
            start_x = torch.rand(1).item() * canvas_size
            start_y = canvas_size - padding
        else:  # Left
            start_x = padding
            start_y = torch.rand(1).item() * canvas_size
        
        # Create stroke moving inward
        points = torch.zeros(num_points, 2)
        direction = torch.rand(2) - 0.5
        direction = direction / direction.norm() * 15
        
        for i in range(num_points):
            points[i, 0] = start_x + direction[0] * i
            points[i, 1] = start_y + direction[1] * i
    
    else:
        # Default to random
        points = torch.rand(num_points, 2) * (canvas_size - 2 * padding) + padding
    
    # Ensure points are within bounds
    points = points.clamp(padding, canvas_size - padding)
    
    return points.to(device).requires_grad_(True)


# ============================================================================
# Loss Utilities
# ============================================================================

def compute_weighted_loss(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    """
    Compute weighted sum of loss terms.
    
    Args:
        losses: Dictionary of loss name → loss tensor.
        weights: Dictionary of loss name → weight.
    
    Returns:
        Total weighted loss.
    """
    device = next(iter(losses.values())).device
    total = torch.tensor(0.0, device=device)
    
    for name, value in losses.items():
        weight = weights.get(name, 0.0)
        if weight > 0:
            total = total + weight * value
    
    return total


# ============================================================================
# Gradient Utilities
# ============================================================================

def add_gradient_noise(
    parameters: List[torch.Tensor],
    noise_scale: float,
) -> None:
    """
    Add Gaussian noise to gradients (in-place).
    
    Args:
        parameters: List of parameter tensors.
        noise_scale: Standard deviation of noise.
    """
    for param in parameters:
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_scale
            param.grad.add_(noise)


def clip_gradients(
    parameters: List[torch.Tensor],
    max_norm: float,
) -> float:
    """
    Clip gradient norms.
    
    Args:
        parameters: List of parameter tensors.
        max_norm: Maximum gradient norm.
    
    Returns:
        Total gradient norm before clipping.
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()


# ============================================================================
# Device Utilities
# ============================================================================

def get_device(tensors: List[torch.Tensor], default: str = "cpu") -> torch.device:
    """
    Get device from a list of tensors.
    
    Args:
        tensors: List of tensors to check.
        default: Default device if list is empty.
    
    Returns:
        Device of the first tensor, or default.
    """
    if tensors:
        return tensors[0].device
    return torch.device(default)


def ensure_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Ensure tensor is on the specified device.
    
    Args:
        tensor: Input tensor.
        device: Target device.
    
    Returns:
        Tensor on the target device.
    """
    if tensor.device != device:
        return tensor.to(device)
    return tensor


