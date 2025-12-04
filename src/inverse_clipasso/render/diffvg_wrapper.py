"""
Differentiable soft line renderer for stroke-based sketches.

Uses Gaussian splatting along line segments to create a fully differentiable
rendering pipeline. Gradients flow from the rendered image back to stroke
control points, enabling gradient-based optimization.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def render_strokes_soft(
    stroke_params: List[torch.Tensor],
    canvas_size: int = 224,
    stroke_width: float = 3.0,
    samples_per_segment: int = 10,
    background: float = 1.0,
    ink_color: float = 0.0,
) -> torch.Tensor:
    """
    Differentiable soft line renderer using Gaussian splatting.
    
    Args:
        stroke_params: List of [N_i, 2] tensors, each representing a stroke's
                      control points as (x, y) coordinates in pixel space.
        canvas_size: Output image size (square). Default 224 for CLIP.
        stroke_width: Line thickness (controls Gaussian sigma).
        samples_per_segment: Number of points to sample between consecutive
                            control points. More samples = smoother lines.
        background: Background intensity (1.0 = white).
        ink_color: Stroke intensity (0.0 = black).
    
    Returns:
        torch.Tensor: Shape [1, 1, H, W] grayscale image tensor.
                     Values in [0, 1], differentiable w.r.t. stroke_params.
    
    Example:
        >>> strokes = [torch.tensor([[10., 10.], [100., 100.]], requires_grad=True)]
        >>> img = render_strokes_soft(strokes)
        >>> loss = some_loss(img)
        >>> loss.backward()  # Gradients flow to stroke coordinates!
    """
    if not stroke_params:
        device = torch.device('cpu')
        dtype = torch.float32
    else:
        device = stroke_params[0].device
        dtype = stroke_params[0].dtype
    
    # Collect all sample points from all strokes
    all_points = []
    
    for stroke in stroke_params:
        if stroke.shape[0] < 2:
            continue
        
        # Get segment start and end points
        p1 = stroke[:-1]  # [N-1, 2] start points
        p2 = stroke[1:]   # [N-1, 2] end points
        n_segments = p1.shape[0]
        
        # Sample points along each segment via linear interpolation
        t = torch.linspace(0, 1, samples_per_segment, device=device, dtype=dtype)
        
        # Broadcast interpolation: [N-1, S, 2]
        # p1: [N-1, 1, 2], t: [1, S, 1], (p2-p1): [N-1, 1, 2]
        segment_points = (
            p1.unsqueeze(1) + 
            t.view(1, -1, 1) * (p2 - p1).unsqueeze(1)
        )
        
        # Flatten to [N-1 * S, 2]
        all_points.append(segment_points.reshape(-1, 2))
    
    # Handle empty strokes case
    if not all_points:
        return torch.full(
            (1, 1, canvas_size, canvas_size), 
            background, 
            device=device, 
            dtype=dtype
        )
    
    all_points = torch.cat(all_points, dim=0)  # [P, 2]
    n_points = all_points.shape[0]
    
    # Create pixel coordinate grid
    coords = torch.arange(canvas_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')  # [H, W] each
    
    # Compute Gaussian splats efficiently
    # For memory efficiency with large point counts, process in chunks
    sigma = stroke_width / 2.0
    sigma_sq_2 = 2.0 * sigma * sigma
    
    chunk_size = 1000  # Process points in chunks to manage memory
    ink = torch.zeros(canvas_size, canvas_size, device=device, dtype=dtype)
    
    for i in range(0, n_points, chunk_size):
        chunk = all_points[i:i + chunk_size]  # [C, 2]
        
        # Extract x, y coordinates and reshape for broadcasting
        px = chunk[:, 0].view(-1, 1, 1)  # [C, 1, 1]
        py = chunk[:, 1].view(-1, 1, 1)  # [C, 1, 1]
        
        # Squared distance from each point to each pixel: [C, H, W]
        dist_sq = (xx.unsqueeze(0) - px) ** 2 + (yy.unsqueeze(0) - py) ** 2
        
        # Gaussian splat contribution
        splats = torch.exp(-dist_sq / sigma_sq_2)  # [C, H, W]
        
        # Accumulate ink
        ink = ink + splats.sum(dim=0)
    
    # Normalize ink to [0, 1] range
    ink_max = ink.max()
    if ink_max > 0:
        ink = ink / ink_max
    ink = torch.clamp(ink, 0, 1)
    
    # Composite: interpolate between background and ink_color based on ink amount
    image = background * (1 - ink) + ink_color * ink
    
    return image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def render_strokes_rgb(
    stroke_params: List[torch.Tensor],
    canvas_size: int = 224,
    stroke_width: float = 3.0,
    samples_per_segment: int = 10,
) -> torch.Tensor:
    """
    Render strokes to RGB image (3 channels) for CLIP compatibility.
    
    Same as render_strokes_soft but returns [1, 3, H, W] with identical
    R, G, B channels (grayscale expanded to RGB).
    
    Args:
        stroke_params: List of [N_i, 2] tensors (stroke control points).
        canvas_size: Output size (default 224 for CLIP).
        stroke_width: Line thickness.
        samples_per_segment: Interpolation density.
    
    Returns:
        torch.Tensor: Shape [1, 3, H, W] RGB image tensor.
    """
    gray = render_strokes_soft(
        stroke_params, 
        canvas_size=canvas_size,
        stroke_width=stroke_width,
        samples_per_segment=samples_per_segment,
    )  # [1, 1, H, W]
    
    # Expand grayscale to RGB by repeating across channel dimension
    return gray.expand(-1, 3, -1, -1)  # [1, 3, H, W]


# Legacy interface for compatibility with existing code
def render_svg(
    paths: List[torch.Tensor], 
    *, 
    canvas_size: Tuple[int, int] = (224, 224), 
    samples: int = 10,
    stroke_width: float = 3.0,
) -> torch.Tensor:
    """
    Legacy interface matching the original diffvg_wrapper signature.
    
    Args:
        paths: List of stroke tensors [N_i, 2].
        canvas_size: (height, width) tuple.
        samples: Points per segment.
        stroke_width: Line thickness.
    
    Returns:
        torch.Tensor: Rendered image [1, 1, H, W].
    """
    h, w = canvas_size
    if h != w:
        raise ValueError(f"Only square canvases supported, got {canvas_size}")
    
    return render_strokes_soft(
        paths,
        canvas_size=h,
        stroke_width=stroke_width,
        samples_per_segment=samples,
    )
