#!/usr/bin/env python3
"""
Test the full optimization loop.

Demonstrates optimizing a simple sketch toward a target semantic label.
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

from src.inverse_clipasso.clip.clip_api import ClipModel
from src.inverse_clipasso.render import render_strokes_rgb
from src.inverse_clipasso.optimize import (
    InverseClipassoOptimizer,
    OptimizationConfig,
)


def render_strokes_pil(drawing, size=256, stroke_width=3, padding=10):
    """Render QuickDraw strokes to PIL image (for loading data)."""
    from PIL import Image, ImageDraw
    
    xs = np.concatenate([np.array(s[0]) for s in drawing])
    ys = np.concatenate([np.array(s[1]) for s in drawing])
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    w, h = max_x - min_x + 1e-6, max_y - min_y + 1e-6
    scale = (size - 2 * padding) / max(w, h)
    ox = (size - scale * w) / 2
    oy = (size - scale * h) / 2

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    for s in drawing:
        x, y = np.array(s[0]), np.array(s[1])
        X, Y = (x - min_x) * scale + ox, (y - min_y) * scale + oy
        pts = list(zip(X, Y))
        if len(pts) >= 2:
            draw.line(pts, fill="black", width=stroke_width, joint="curve")
    return img


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


def get_best_device():
    """Get the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def test_basic_optimization():
    """Test optimization with a simple synthetic drawing."""
    print("=" * 70)
    print("Test 1: Basic Optimization (synthetic drawing)")
    print("=" * 70)
    
    device = get_best_device()
    print(f"Using device: {device}")
    
    # Create a simple initial drawing (random lines)
    init_strokes = [
        torch.tensor([
            [50., 50.],
            [100., 80.],
            [150., 50.],
        ], device=device),
        torch.tensor([
            [60., 100.],
            [140., 100.],
            [140., 170.],
            [60., 170.],
            [60., 100.],
        ], device=device),
    ]
    
    # Load CLIP
    print("Loading CLIP model...")
    clip_model = ClipModel(device=device, use_float32=True)
    
    # Configure optimization with improved settings
    config = OptimizationConfig(
        label="a house",
        steps=500,           # Moderate steps with better LR schedule
        lr=5.0,              # Higher initial LR
        lr_schedule="warmup_cosine",  # Warmup then cosine decay
        warmup_steps=50,     # Warmup for first 50 steps
        min_lr=0.1,          # Don't go below 0.1
        noise_scale=0.5,     # Add noise to escape local minima
        noise_decay=0.995,   # Decay noise over time
        loss_weights={
            "semantic": 1.0,
            "stroke": 0.001,  # Very light regularization
            "tv": 0.0001,
        },
        use_tqdm=True,
    )
    
    # Create optimizer
    optimizer = InverseClipassoOptimizer(clip_model, config)
    
    # Run optimization
    print(f"\nOptimizing toward '{config.label}'...")
    result = optimizer.optimize(init_strokes)
    
    print(f"\n✓ Optimization complete!")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Final similarity: {result.final_similarity:.4f}")
    print(f"  Steps: {len(result.loss_history['total'])}")
    
    return result, init_strokes


def test_quickdraw_optimization():
    """Test optimization starting from a real QuickDraw sketch."""
    print("\n" + "=" * 70)
    print("Test 2: QuickDraw Sketch Optimization")
    print("=" * 70)
    
    device = get_best_device()
    
    # Load a random QuickDraw sample
    data_dir = Path(PROJECT_ROOT) / "data" / "quickdraw" / "raw" / "strokes"
    available = list(data_dir.glob("*.ndjson"))
    
    if not available:
        print("  ⚠ No QuickDraw data found, skipping test")
        return None, None
    
    # Pick a random file
    source_file = random.choice(available)
    source_label = source_file.stem
    
    with open(source_file) as f:
        lines = f.readlines()
    sample = json.loads(random.choice(lines))
    drawing = sample["drawing"]
    
    print(f"  Source: {source_label} sketch ({len(drawing)} strokes)")
    
    # Convert to stroke parameters
    init_strokes = quickdraw_to_stroke_params(drawing, device=device)
    
    # Pick a DIFFERENT target label
    all_labels = [f.stem for f in available]
    target_options = [l for l in all_labels if l != source_label]
    target_label = random.choice(target_options) if target_options else source_label
    
    print(f"  Target: '{target_label}'")
    
    # Load CLIP
    print("  Loading CLIP model...")
    clip_model = ClipModel(device=device, use_float32=True)
    
    # Configure optimization with improved settings
    config = OptimizationConfig(
        label=f"a {target_label}",
        steps=500,           # Moderate steps with better LR schedule
        lr=3.0,              # Higher initial LR (slightly lower for complex strokes)
        lr_schedule="warmup_cosine",
        warmup_steps=30,
        min_lr=0.05,
        noise_scale=0.3,     # Less noise for complex strokes
        noise_decay=0.99,
        loss_weights={
            "semantic": 1.0,
            "stroke": 0.005,  # Light regularization
            "tv": 0.0001,
        },
        use_tqdm=True,
    )
    
    # Optimize
    optimizer = InverseClipassoOptimizer(clip_model, config)
    
    print(f"\n  Optimizing {source_label} → {target_label}...")
    result = optimizer.optimize(init_strokes)
    
    print(f"\n  ✓ Optimization complete!")
    print(f"    Final similarity to '{target_label}': {result.final_similarity:.4f}")
    
    return result, init_strokes, source_label, target_label


def visualize_optimization(result, init_strokes, title="Optimization Results"):
    """Visualize optimization results."""
    print("\n" + "=" * 70)
    print("Visualization")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Row 1: Before and After images
    device = init_strokes[0].device
    
    # Initial rendering
    img_init = render_strokes_rgb(init_strokes, canvas_size=224)
    axes[0, 0].imshow(img_init.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    axes[0, 0].set_title('Initial Sketch', fontsize=12)
    axes[0, 0].axis('off')
    
    # Final rendering
    final_strokes = [s.to(device) for s in result.final_strokes]
    img_final = render_strokes_rgb(final_strokes, canvas_size=224)
    axes[0, 1].imshow(img_final.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    axes[0, 1].set_title(f'After Optimization\n(similarity: {result.final_similarity:.3f})', fontsize=12)
    axes[0, 1].axis('off')
    
    # Overlay: initial (blue) vs final (red)
    ax_overlay = axes[0, 2]
    ax_overlay.set_xlim(0, 224)
    ax_overlay.set_ylim(224, 0)
    
    for stroke in init_strokes:
        pts = stroke.detach().cpu().numpy()
        ax_overlay.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=2, alpha=0.5, label='Initial')
    
    for stroke in result.final_strokes:
        pts = stroke.detach().cpu().numpy()
        ax_overlay.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=2, alpha=0.7, label='Final')
    
    ax_overlay.set_title('Stroke Movement\n(blue=initial, red=final)', fontsize=12)
    ax_overlay.set_aspect('equal')
    ax_overlay.grid(True, alpha=0.3)
    
    # Row 2: Loss curves
    steps = range(len(result.loss_history['total']))
    
    # Total loss
    axes[1, 0].plot(steps, result.loss_history['total'], 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Total Loss')
    axes[1, 0].set_title('Total Loss', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Semantic loss / similarity
    ax_sem = axes[1, 1]
    ax_sem.plot(steps, result.loss_history['semantic'], 'r-', linewidth=2, label='Semantic Loss')
    ax_sem.set_xlabel('Step')
    ax_sem.set_ylabel('Semantic Loss', color='red')
    ax_sem.tick_params(axis='y', labelcolor='red')
    
    ax_sim = ax_sem.twinx()
    ax_sim.plot(steps, result.loss_history['similarity'], 'g-', linewidth=2, label='Similarity')
    ax_sim.set_ylabel('CLIP Similarity', color='green')
    ax_sim.tick_params(axis='y', labelcolor='green')
    ax_sem.set_title('Semantic Loss & Similarity', fontsize=12)
    ax_sem.grid(True, alpha=0.3)
    
    # Regularization losses
    axes[1, 2].plot(steps, result.loss_history['stroke'], 'orange', linewidth=2, label='Stroke Reg')
    axes[1, 2].plot(steps, result.loss_history['tv'], 'purple', linewidth=2, label='Total Variation')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title('Regularization Losses', fontsize=12)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to optimization_results.png")
    plt.show()


def main():
    print("\n" + "=" * 70)
    print("OPTIMIZATION LOOP TESTS")
    print("=" * 70)
    
    # Test 1: Basic optimization
    result1, init_strokes1 = test_basic_optimization()
    
    # Test 2: QuickDraw optimization
    qd_result = test_quickdraw_optimization()
    
    # Visualize the basic test
    if result1 is not None:
        visualize_optimization(result1, init_strokes1, "Basic Optimization: → 'a house'")
    
    # Visualize QuickDraw test if it ran
    if qd_result is not None and qd_result[0] is not None:
        result2, init_strokes2, source, target = qd_result
        visualize_optimization(
            result2, init_strokes2, 
            f"QuickDraw: '{source}' → 'a {target}'"
        )
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()

