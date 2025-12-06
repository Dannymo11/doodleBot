"""
Bézier curve representation for CLIPasso-style stroke optimization.

Implements cubic Bézier curves which provide:
- Smooth, natural-looking strokes
- Fewer parameters than polylines for similar quality
- Better gradient flow during optimization
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BezierStroke(nn.Module):
    """
    A single stroke represented as a cubic Bézier curve.
    
    Cubic Bézier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    
    Where:
    - P₀, P₃ are the start/end points
    - P₁, P₂ are the control points that define curvature
    
    Attributes:
        control_points: [4, 2] tensor of (x, y) control points
        stroke_width: Learnable stroke thickness
        opacity: Learnable opacity (0-1)
    """
    
    def __init__(
        self,
        control_points: Optional[torch.Tensor] = None,
        stroke_width: float = 3.0,
        opacity: float = 1.0,
        canvas_size: int = 224,
        device: str = 'cpu',
    ):
        super().__init__()
        
        if control_points is not None:
            self.control_points = nn.Parameter(control_points.to(device))
        else:
            # Initialize with random points, biased toward center
            padding = canvas_size * 0.1
            points = torch.rand(4, 2, device=device) * (canvas_size - 2 * padding) + padding
            # Sort by x to create a more natural left-to-right stroke
            sorted_indices = points[:, 0].argsort()
            self.control_points = nn.Parameter(points[sorted_indices])
        
        self.stroke_width = nn.Parameter(torch.tensor(stroke_width, device=device))
        self.opacity = nn.Parameter(torch.tensor(opacity, device=device))
        self.canvas_size = canvas_size
    
    def sample_curve(self, num_samples: int = 50) -> torch.Tensor:
        """
        Sample points along the Bézier curve.
        
        Args:
            num_samples: Number of points to sample along the curve.
        
        Returns:
            Tensor of shape [num_samples, 2] with (x, y) coordinates.
        """
        device = self.control_points.device
        t = torch.linspace(0, 1, num_samples, device=device).unsqueeze(1)  # [N, 1]
        
        P0, P1, P2, P3 = self.control_points[0], self.control_points[1], \
                          self.control_points[2], self.control_points[3]
        
        # Cubic Bézier formula
        points = (
            (1 - t) ** 3 * P0 +
            3 * (1 - t) ** 2 * t * P1 +
            3 * (1 - t) * t ** 2 * P2 +
            t ** 3 * P3
        )
        
        return points
    
    def get_width(self) -> torch.Tensor:
        """Get clamped stroke width."""
        return torch.clamp(self.stroke_width, min=0.5, max=10.0)
    
    def get_opacity(self) -> torch.Tensor:
        """Get clamped opacity."""
        return torch.clamp(self.opacity, min=0.0, max=1.0)
    
    def to_polyline(self, num_samples: int = 50) -> torch.Tensor:
        """
        Convert to polyline format for rendering compatibility.
        
        Returns:
            Tensor of shape [num_samples, 2] that can be used with existing renderer.
        """
        return self.sample_curve(num_samples)


class BezierSketch(nn.Module):
    """
    A complete sketch composed of multiple Bézier strokes.
    
    This is the main interface for CLIPasso-style optimization.
    """
    
    def __init__(
        self,
        num_strokes: int = 8,
        canvas_size: int = 224,
        stroke_width: float = 3.0,
        device: str = 'cpu',
        init_strokes: Optional[List[torch.Tensor]] = None,
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.device = device
        
        # Initialize strokes
        if init_strokes is not None:
            # Convert existing polylines to Bézier curves
            self.strokes = nn.ModuleList([
                self._polyline_to_bezier(stroke, stroke_width, device)
                for stroke in init_strokes
            ])
        else:
            self.strokes = nn.ModuleList([
                BezierStroke(
                    canvas_size=canvas_size,
                    stroke_width=stroke_width,
                    device=device,
                )
                for _ in range(num_strokes)
            ])
    
    def _polyline_to_bezier(
        self,
        polyline: torch.Tensor,
        stroke_width: float,
        device: str,
    ) -> BezierStroke:
        """
        Convert a polyline to a cubic Bézier curve.
        
        Uses the first, last, and two intermediate points as control points.
        """
        n_points = polyline.shape[0]
        
        if n_points >= 4:
            # Sample 4 evenly-spaced points
            indices = torch.linspace(0, n_points - 1, 4).long()
            control_points = polyline[indices]
        elif n_points == 3:
            # Duplicate middle point for control
            control_points = torch.stack([
                polyline[0],
                polyline[1],
                polyline[1],
                polyline[2],
            ])
        elif n_points == 2:
            # Create control points between endpoints
            p0, p1 = polyline[0], polyline[1]
            control_points = torch.stack([
                p0,
                p0 + (p1 - p0) * 0.33,
                p0 + (p1 - p0) * 0.67,
                p1,
            ])
        else:
            # Single point - create a small curve
            p = polyline[0]
            offset = torch.tensor([10.0, 10.0], device=polyline.device)
            control_points = torch.stack([
                p - offset,
                p - offset * 0.5,
                p + offset * 0.5,
                p + offset,
            ])
        
        return BezierStroke(
            control_points=control_points,
            stroke_width=stroke_width,
            canvas_size=self.canvas_size,
            device=device,
        )
    
    def add_stroke(self, stroke: Optional[BezierStroke] = None):
        """Add a new stroke to the sketch."""
        if stroke is None:
            stroke = BezierStroke(
                canvas_size=self.canvas_size,
                stroke_width=3.0,
                device=self.device,
            )
        self.strokes.append(stroke)
    
    def to_polylines(self, num_samples: int = 50) -> List[torch.Tensor]:
        """
        Convert all strokes to polyline format for rendering.
        
        Returns:
            List of [N, 2] tensors compatible with existing renderer.
        """
        return [stroke.to_polyline(num_samples) for stroke in self.strokes]
    
    def get_parameters_grouped(self) -> dict:
        """
        Get parameters grouped by type for separate learning rates.
        
        Returns:
            Dict with 'points', 'widths', 'opacities' parameter groups.
        """
        return {
            'points': [s.control_points for s in self.strokes],
            'widths': [s.stroke_width for s in self.strokes],
            'opacities': [s.opacity for s in self.strokes],
        }
    
    def clamp_to_canvas(self):
        """Clamp all control points to canvas bounds."""
        with torch.no_grad():
            for stroke in self.strokes:
                stroke.control_points.clamp_(0, self.canvas_size)
    
    def __len__(self):
        return len(self.strokes)


def render_bezier_strokes(
    sketch: BezierSketch,
    canvas_size: int = 224,
    samples_per_stroke: int = 50,
    stroke_width_override: Optional[float] = None,
) -> torch.Tensor:
    """
    Render a BezierSketch using Gaussian splatting.
    
    Args:
        sketch: BezierSketch to render.
        canvas_size: Output image size.
        samples_per_stroke: Points to sample per Bézier curve.
        stroke_width_override: If set, use this width for all strokes.
    
    Returns:
        Rendered image tensor [1, 3, H, W].
    """
    from src.inverse_clipasso.render import render_strokes_rgb
    
    # Convert to polylines
    polylines = sketch.to_polylines(samples_per_stroke)
    
    # Use per-stroke widths or override
    if stroke_width_override is not None:
        width = stroke_width_override
    else:
        # Average stroke width (could also render each with different widths)
        width = sum(s.get_width().item() for s in sketch.strokes) / len(sketch.strokes)
    
    return render_strokes_rgb(
        polylines,
        canvas_size=canvas_size,
        stroke_width=width,
        samples_per_segment=10,
    )


def initialize_bezier_from_edges(
    image: torch.Tensor,
    num_strokes: int = 8,
    canvas_size: int = 224,
    device: str = 'cpu',
) -> BezierSketch:
    """
    Initialize Bézier strokes from image edges (saliency-based initialization).
    
    Args:
        image: Input image tensor [1, 3, H, W] or [3, H, W].
        num_strokes: Number of strokes to create.
        canvas_size: Canvas size for the sketch.
        device: Device for tensors.
    
    Returns:
        BezierSketch initialized along image edges.
    """
    import numpy as np
    
    # Convert to grayscale numpy
    if image.dim() == 4:
        image = image.squeeze(0)
    img_gray = image.mean(dim=0).cpu().numpy()
    img_gray = (img_gray * 255).astype(np.uint8)
    
    # Simple edge detection using gradients
    grad_x = np.abs(np.diff(img_gray.astype(float), axis=1))
    grad_y = np.abs(np.diff(img_gray.astype(float), axis=0))
    
    # Pad to original size
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)))
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)))
    
    # Combined gradient magnitude
    edge_strength = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # Find strong edge points
    threshold = np.percentile(edge_strength, 90)
    edge_points = np.argwhere(edge_strength > threshold)
    
    if len(edge_points) < 4 * num_strokes:
        # Not enough edge points, use random initialization
        return BezierSketch(
            num_strokes=num_strokes,
            canvas_size=canvas_size,
            device=device,
        )
    
    # Create strokes from edge point clusters
    strokes = []
    np.random.shuffle(edge_points)
    
    for i in range(num_strokes):
        # Sample 4 nearby points for each stroke
        start_idx = i * 4 % len(edge_points)
        points_yx = edge_points[start_idx:start_idx + 4]
        
        if len(points_yx) < 4:
            points_yx = edge_points[:4]
        
        # Convert yx to xy
        points_xy = points_yx[:, ::-1].copy()
        control_points = torch.tensor(points_xy, dtype=torch.float32, device=device)
        
        strokes.append(BezierStroke(
            control_points=control_points,
            canvas_size=canvas_size,
            device=device,
        ))
    
    sketch = BezierSketch(
        num_strokes=0,
        canvas_size=canvas_size,
        device=device,
    )
    sketch.strokes = nn.ModuleList(strokes)
    
    return sketch



