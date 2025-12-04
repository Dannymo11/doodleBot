#!/usr/bin/env python3
"""
Long optimization run with progression visualization.

Saves intermediate snapshots and creates a grid showing the transformation over time.
"""

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io

from src.inverse_clipasso.clip.clip_api import ClipModel
from src.inverse_clipasso.render import render_strokes_rgb
from src.inverse_clipasso.optimize import (
    InverseClipassoOptimizer,
    OptimizationConfig,
)


def load_quickdraw_exemplars(
    category: str,
    n_exemplars: int = 20,
    clip_model: ClipModel = None,
    canvas_size: int = 224,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load QuickDraw sketches and encode them with CLIP to create exemplar embeddings.
    
    This gives us a "cluster" of what good sketches of this category look like
    in CLIP space, which we can use to guide the optimization.
    
    Args:
        category: QuickDraw category (e.g., "fish", "cat")
        n_exemplars: Number of exemplar sketches to load
        clip_model: CLIP model for encoding
        canvas_size: Size to render sketches
        device: Device for tensors
    
    Returns:
        Tensor of CLIP embeddings [N, D]
    """
    data_dir = Path(PROJECT_ROOT) / "data" / "quickdraw" / "raw" / "strokes"
    
    # Find the file
    category_file = data_dir / f"{category}.ndjson"
    if not category_file.exists():
        print(f"Warning: {category_file} not found, returning None")
        return None
    
    # Load samples
    with open(category_file) as f:
        lines = f.readlines()
    
    # Sample random exemplars
    if len(lines) > n_exemplars:
        sample_lines = random.sample(lines, n_exemplars)
    else:
        sample_lines = lines
    
    print(f"  Loading {len(sample_lines)} exemplar sketches for '{category}'...")
    
    # Render and encode each exemplar
    embeddings = []
    for line in sample_lines:
        sample = json.loads(line)
        drawing = sample["drawing"]
        
        # Convert to stroke params and render
        strokes = quickdraw_to_stroke_params(drawing, size=canvas_size, device=device)
        img_tensor = render_strokes_rgb(strokes, canvas_size=canvas_size)
        
        # Encode with CLIP
        img_normalized = (img_tensor - torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)) / \
                         torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(img_normalized, requires_grad=False)
        embeddings.append(emb)
    
    # Stack into single tensor
    exemplar_embeddings = torch.cat(embeddings, dim=0)
    print(f"  Exemplar embeddings shape: {exemplar_embeddings.shape}")
    
    return exemplar_embeddings


def quickdraw_to_stroke_params(drawing, size=224, padding=10, device="cpu"):
    """Convert QuickDraw drawing to list of stroke parameter tensors."""
    xs_all = []
    ys_all = []
    for stroke in drawing:
        xs_all.append(np.array(stroke[0], dtype=np.float32))
        ys_all.append(np.array(stroke[1], dtype=np.float32))
    
    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    w = max_x - min_x + 1e-6
    h = max_y - min_y + 1e-6
    
    scale = (size - 2 * padding) / max(w, h)
    ox = (size - scale * w) / 2.0
    oy = (size - scale * h) / 2.0
    
    stroke_params = []
    for stroke in drawing:
        x_coords = np.array(stroke[0], dtype=np.float32)
        y_coords = np.array(stroke[1], dtype=np.float32)
        
        X = (x_coords - min_x) * scale + ox
        Y = (y_coords - min_y) * scale + oy
        
        pts = np.stack([X, Y], axis=-1)
        param = torch.tensor(pts, dtype=torch.float32, device=device)
        stroke_params.append(param)
    
    return stroke_params


def render_to_pil(strokes, canvas_size=224):
    """Render strokes to a PIL Image."""
    img_tensor = render_strokes_rgb(strokes, canvas_size=canvas_size)
    img_np = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def create_progression_grid(snapshots, cols=8):
    """
    Create a grid image showing the progression of optimization.
    
    Args:
        snapshots: List of (step, similarity, PIL.Image) tuples
        cols: Number of columns in the grid
    
    Returns:
        PIL.Image showing the progression
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
    except:
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


def create_gif(snapshots, output_path, duration=200):
    """
    Create an animated GIF showing the optimization progression.
    
    Args:
        snapshots: List of (step, similarity, PIL.Image) tuples
        output_path: Path to save the GIF
        duration: Duration of each frame in milliseconds
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


def run_long_optimization(
    init_strokes,
    target_label,
    source_label="sketch",
    steps=1000,
    snapshot_every=50,
    noise_scale=0.3,
    lr=3.0,
    use_multi_prompt=False,
    refinement_mode=False,
    original_weight=0.5,
    original_weight_schedule="constant",  # "constant", "linear_up", "cosine_up"
    original_weight_start=0.1,  # Starting weight (for scheduled modes)
    # Exemplar-guided refinement
    use_exemplar_loss=False,
    exemplar_category=None,  # QuickDraw category for exemplars
    n_exemplars=20,          # Number of exemplar sketches
    exemplar_weight=0.5,     # Weight for exemplar loss
    exemplar_mode="mean",    # "mean", "min", "softmin"
):
    """
    Run a long optimization and capture snapshots.
    
    Args:
        init_strokes: Initial stroke parameters.
        target_label: Target semantic label.
        source_label: Source label (for display only).
        steps: Number of optimization steps.
        snapshot_every: Save snapshot every N steps.
        noise_scale: Scale of noise for exploration.
        lr: Learning rate.
        use_multi_prompt: If True, average multiple text prompts.
        refinement_mode: If True, preserve original stroke positions.
        original_weight: Weight for original strokes preservation (refinement mode).
        use_exemplar_loss: If True, guide toward QuickDraw exemplars.
        exemplar_category: QuickDraw category for loading exemplars.
        n_exemplars: Number of exemplar sketches to load.
        exemplar_weight: Weight for exemplar loss.
        exemplar_mode: How to aggregate exemplar similarities.
    
    Returns:
        result: OptimizationResult
        snapshots: List of (step, similarity, PIL.Image) tuples
    """
    device = init_strokes[0].device if isinstance(init_strokes[0], torch.Tensor) else "cpu"
    
    # Storage for snapshots
    snapshots = []
    
    print(f"\n{'='*70}")
    print(f"Long Optimization: '{source_label}' → '{target_label}'")
    print(f"{'='*70}")
    print(f"  Steps: {steps}")
    print(f"  Snapshot every: {snapshot_every} steps")
    print(f"  Noise scale: {noise_scale}")
    print(f"  Learning rate: {lr}")
    print(f"  Multi-prompt: {use_multi_prompt}")
    print(f"  Refinement mode: {refinement_mode}")
    if refinement_mode:
        print(f"  Original weight: {original_weight_start} → {original_weight} ({original_weight_schedule})")
    if use_exemplar_loss:
        print(f"  Exemplar loss: {exemplar_category} ({n_exemplars} samples, weight={exemplar_weight}, mode={exemplar_mode})")
    print(f"  Device: {device}")
    
    # Load CLIP
    print("  Loading CLIP model...")
    clip_model = ClipModel(device=device, use_float32=True)
    
    # Load exemplar embeddings if requested
    exemplar_embeddings = None
    if use_exemplar_loss and exemplar_category:
        exemplar_embeddings = load_quickdraw_exemplars(
            category=exemplar_category,
            n_exemplars=n_exemplars,
            clip_model=clip_model,
            canvas_size=224,
            device=device,
        )
    
    # Build loss weights
    loss_weights = {
        "semantic": 1.0,
        "stroke": 0.05,
        "tv": 0.001,
    }
    if use_exemplar_loss:
        loss_weights["exemplar"] = exemplar_weight
    
    # Configure optimization
    config = OptimizationConfig(
        label=target_label,
        steps=steps,
        lr=lr,
        lr_schedule="warmup_cosine",
        warmup_steps=min(100, steps // 10),
        min_lr=0.05,
        noise_scale=noise_scale,
        noise_decay=0.995,
        loss_weights=loss_weights,
        use_tqdm=True,
        # Multi-prompt settings
        use_multi_prompt=use_multi_prompt,
        # Refinement settings  
        refinement_mode=refinement_mode,
        original_weight=original_weight,
        original_weight_schedule=original_weight_schedule,
        original_weight_start=original_weight_start,
        # Exemplar settings
        use_exemplar_loss=use_exemplar_loss,
        exemplar_embeddings=exemplar_embeddings,
        exemplar_weight=exemplar_weight,
        exemplar_mode=exemplar_mode,
    )
    
    # Callback to capture snapshots
    def snapshot_callback(step, losses, strokes):
        if step % snapshot_every == 0 or step == steps - 1:
            # Render current state
            strokes_on_device = [s.to(device) for s in strokes]
            img = render_to_pil(strokes_on_device)
            similarity = 1.0 - losses.get("semantic", 1.0)
            snapshots.append((step, similarity, img))
    
    # Create optimizer and run
    optimizer = InverseClipassoOptimizer(clip_model, config)
    
    print(f"\n  Starting optimization...")
    result = optimizer.optimize(init_strokes, callback=snapshot_callback)
    
    # Capture final state if not already captured
    if snapshots[-1][0] != steps - 1:
        final_img = render_to_pil([s.to(device) for s in result.final_strokes])
        snapshots.append((steps - 1, result.final_similarity, final_img))
    
    print(f"\n  ✓ Optimization complete!")
    print(f"    Initial similarity: {snapshots[0][1]:.1%}")
    print(f"    Final similarity: {result.final_similarity:.1%}")
    print(f"    Improvement: {(result.final_similarity - snapshots[0][1]):.1%}")
    print(f"    Captured {len(snapshots)} snapshots")
    
    return result, snapshots


def visualize_full_results(result, init_strokes, snapshots, source_label, target_label):
    """Create comprehensive visualization with progression grid."""
    
    output_dir = Path("outputs/long_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create progression grid
    print("\n  Creating progression grid...")
    grid = create_progression_grid(snapshots, cols=8)
    grid_path = output_dir / f"{source_label}_to_{target_label}_progression.png"
    grid.save(grid_path)
    print(f"  Saved: {grid_path}")
    
    # 2. Create GIF animation
    print("  Creating animation...")
    gif_path = output_dir / f"{source_label}_to_{target_label}_animation.gif"
    create_gif(snapshots, gif_path, duration=150)
    
    # 3. Create detailed loss plot
    print("  Creating loss plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    device = init_strokes[0].device
    
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
    
    plt.suptitle(f"'{source_label}' → '{target_label}' ({len(result.loss_history['total'])} steps)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_dir / f"{source_label}_to_{target_label}_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.show()
    
    return grid_path, gif_path, plot_path


def get_best_device():

    if torch.cuda.is_available():
        return "cuda"
    # Skip MPS - it's ~15x slower for this workload due to small tensor overhead
    # elif torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


def main():
    device = get_best_device()
    print(f"\n{'='*70}")
    print("LONG OPTIMIZATION WITH PROGRESSION TRACKING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Load a QuickDraw sketch
    data_dir = Path(PROJECT_ROOT) / "data" / "quickdraw" / "raw" / "strokes"
    available = list(data_dir.glob("*.ndjson"))
    
    if not available:
        print("No QuickDraw data found. Using synthetic drawing.")
        # Fallback: synthetic house
        init_strokes = [
            torch.tensor([
                [50., 50.], [100., 20.], [150., 50.],
            ], device=device),
            torch.tensor([
                [60., 50.], [60., 150.], [140., 150.], [140., 50.], [60., 50.],
            ], device=device),
        ]
        source_label = "house"
        target_label = "a castle"
    else:
        # Pick a specific or random sketch
        # Let's try fish → bird again for comparison
        source_file = None
        for f in available:
            if f.stem == "fish":
                source_file = f
                break
        
        if source_file is None:
            source_file = random.choice(available)
        
        source_label = source_file.stem
        
        with open(source_file) as f:
            lines = f.readlines()
        sample = json.loads(random.choice(lines))
        drawing = sample["drawing"]
        
        init_strokes = quickdraw_to_stroke_params(drawing, device=device)
        
        # Choose target
        target_label = "A perfect sketch of a fish"  # Refine fish into a better fish
    
    print(f"\nSource: {source_label}")
    print(f"Target: {target_label}")
    print(f"Strokes: {len(init_strokes)}")
    
    # Detect if this is a refinement (same category) or transformation
    is_refinement = source_label in target_label or target_label.replace("a ", "") in source_label
    
    if is_refinement:
        print(f"\n🔧 REFINEMENT MODE: Improving '{source_label}' while preserving structure")
    else:
        print(f"\n🔄 TRANSFORMATION MODE: Converting '{source_label}' → '{target_label}'")
    
    # Extract the category name for exemplar loading
    exemplar_category = source_label  # e.g., "fish" for fish → fish refinement
    
    # Run optimization with appropriate settings
    result, snapshots = run_long_optimization(
        init_strokes=init_strokes,
        target_label=target_label,
        source_label=source_label,
        steps=1500,                                   # More steps for better convergence
        snapshot_every=50,
        noise_scale=0.3 if is_refinement else 0.4,   # Moderate noise for exploration
        lr=2.5 if is_refinement else 3.0,            # Higher LR for faster initial progress
        use_multi_prompt=True,                        # Always use multi-prompt now!
        refinement_mode=is_refinement,                # Preserve original when refining
        # SCHEDULED ORIGINAL WEIGHT: Start low (more CLIP exploration), end high (lock structure)
        original_weight=0.5 if is_refinement else 0,  # Reduced since exemplar loss helps
        original_weight_schedule="cosine_up" if is_refinement else "constant",
        original_weight_start=0.05,                   # Start very low to allow CLIP exploration!
        # EXEMPLAR LOSS: Guide toward "good sketches" from QuickDraw
        use_exemplar_loss=is_refinement,              # Use exemplar loss for refinement
        exemplar_category=exemplar_category,          # Load exemplars from same category
        n_exemplars=30,                               # Number of reference sketches
        exemplar_weight=1.0,                          # Strong pull toward good examples
        exemplar_mode="softmin",                      # Pull toward nearest good examples
    )
    
    # Create visualizations
    grid_path, gif_path, plot_path = visualize_full_results(
        result, init_strokes, snapshots, source_label, target_label
    )
    
    print(f"\n{'='*70}")
    print("COMPLETE! Generated files:")
    print(f"{'='*70}")
    print(f"  📊 Progression grid: {grid_path}")
    print(f"  🎬 Animation: {gif_path}")
    print(f"  📈 Summary plot: {plot_path}")
    print(f"\nFinal similarity: {result.final_similarity:.1%}")


if __name__ == "__main__":
    main()

