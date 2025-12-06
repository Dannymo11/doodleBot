"""
Visualization utilities for sketch optimization experiments.

Provides functions for:
- Rendering strokes to PIL images
- Creating progression grids
- Creating animated GIFs
- Full result visualization
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.inverse_clipasso.render import render_strokes_rgb


__all__ = [
    "render_to_pil",
    "create_progression_grid",
    "create_gif",
    "visualize_full_results",
]


@torch.no_grad()
def render_to_pil(strokes: List[torch.Tensor], canvas_size: int = 224) -> Image.Image:
    """
    Render strokes to a PIL Image.
    
    Args:
        strokes: List of stroke tensors.
        canvas_size: Size of the canvas.
    
    Returns:
        PIL Image of the rendered sketch.
    """
    img_tensor = render_strokes_rgb(strokes, canvas_size=canvas_size)
    img_np = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def create_progression_grid(
    snapshots: List[Tuple[int, float, Image.Image]],
    cols: int = 8,
) -> Image.Image:
    """
    Create a grid image showing the progression of optimization.
    
    Args:
        snapshots: List of (step, similarity, PIL.Image) tuples.
        cols: Number of columns in the grid.
    
    Returns:
        PIL.Image showing the progression.
    """
    n = len(snapshots)
    rows = (n + cols - 1) // cols
    
    # Get image size from first snapshot
    img_size = snapshots[0][2].size[0]
    label_height = 30
    cell_height = img_size + label_height
    
    # Create canvas
    grid = Image.new('RGB', (cols * img_size, rows * cell_height), 'white')
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid)
    
    # Try to use a nicer font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    
    for idx, (step, sim, img) in enumerate(snapshots):
        row = idx // cols
        col = idx % cols
        
        x = col * img_size
        y = row * cell_height
        
        # Paste the image
        grid.paste(img, (x, y))
        
        # Draw label
        label = f"Step {step}: {sim:.1%}"
        draw.text((x + 5, y + img_size + 5), label, fill='black', font=font)
    
    return grid


def create_gif(
    snapshots: List[Tuple[int, float, Image.Image]],
    output_path: str,
    duration: int = 200,
) -> None:
    """
    Create an animated GIF showing the optimization progression.
    
    Args:
        snapshots: List of (step, similarity, PIL.Image) tuples.
        output_path: Path to save the GIF.
        duration: Duration of each frame in milliseconds.
    """
    frames = [snap[2] for snap in snapshots]
    
    if len(frames) > 1:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"  Saved animation to {output_path}")


def visualize_full_results(
    result,
    init_strokes: List[torch.Tensor],
    snapshots: List[Tuple[int, float, Image.Image]],
    source_label: str,
    target_label: str,
    output_dir: str = "outputs/long_optimization",
) -> Tuple[Path, Path, Path]:
    """
    Create comprehensive visualization with progression grid.
    
    Args:
        result: OptimizationResult from the optimizer.
        init_strokes: Initial stroke parameters.
        snapshots: List of (step, similarity, PIL.Image) tuples.
        source_label: Label for the source sketch.
        target_label: Label for the target.
        output_dir: Directory to save outputs.
    
    Returns:
        Tuple of (grid_path, gif_path, plot_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = init_strokes[0].device if init_strokes else torch.device("cpu")
    
    # 1. Create progression grid
    print("\n  Creating progression grid...")
    grid = create_progression_grid(snapshots, cols=8)
    grid_path = output_dir / f"{source_label}_to_{target_label}_progression.png"
    grid.save(grid_path)
    print(f"  Saved: {grid_path}")
    
    # 2. Create GIF animation
    print("  Creating animation...")
    gif_path = output_dir / f"{source_label}_to_{target_label}_animation.gif"
    create_gif(snapshots, str(gif_path), duration=150)
    
    # 3. Create detailed loss plot
    print("  Creating loss plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Initial vs Final
    ax = axes[0, 0]
    img_init = render_to_pil(init_strokes)
    ax.imshow(img_init)
    ax.set_title(f'Initial: {snapshots[0][1]:.1%} similarity', fontsize=12)
    ax.axis('off')
    
    ax = axes[0, 1]
    final_strokes = [s.to(device) for s in result.final_strokes]
    img_final = render_to_pil(final_strokes)
    ax.imshow(img_final)
    ax.set_title(f'Final: {result.final_similarity:.1%} similarity', fontsize=12)
    ax.axis('off')
    
    # Loss curve
    ax = axes[1, 0]
    steps = range(len(result.loss_history['total']))
    ax.plot(steps, result.loss_history['total'], 'b-', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Over Time')
    ax.grid(True, alpha=0.3)
    
    # Similarity curve
    ax = axes[1, 1]
    ax.plot(steps, result.loss_history['similarity'], 'g-', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('CLIP Similarity')
    ax.set_title('Semantic Similarity Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(result.loss_history['similarity']) * 1.1)
    
    plt.suptitle(
        f"'{source_label}' → '{target_label}' ({len(result.loss_history['total'])} steps)",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    plot_path = output_dir / f"{source_label}_to_{target_label}_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()
    
    return grid_path, gif_path, plot_path


