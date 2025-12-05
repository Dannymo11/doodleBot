"""
Test script for CLIPasso-style optimizer.

This tests the new architectural improvements:
1. Bézier curve representation
2. Augmentations during optimization
3. Multi-scale perceptual loss

Can load QuickDraw sketches and optimize them using the CLIPasso pipeline!
"""

import sys
import os
import json
import random
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

from src.inverse_clipasso.clip.clip_api import ClipModel
from src.inverse_clipasso.optimize.clipasso_optimizer import (
    CLIPassoOptimizer,
    CLIPassoConfig,
    CLIPassoResult,
)
from src.inverse_clipasso.render import (
    render_strokes_rgb,
    BezierSketch,
    render_bezier_strokes,
)


def get_device():
    """Get best available device (prefer CPU for this workload due to MPS overhead)."""
    if torch.cuda.is_available():
        return "cuda"
    # MPS has high overhead for many small tensor ops, CPU is faster here
    return "cpu"


def render_sketch_to_pil(sketch: BezierSketch, canvas_size: int = 224) -> Image.Image:
    """Render BezierSketch to PIL Image."""
    rendered = render_bezier_strokes(sketch, canvas_size=canvas_size)
    
    # Convert to PIL (detach to remove grad requirement)
    img_np = rendered.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np)


def render_strokes_to_pil(strokes, canvas_size: int = 224) -> Image.Image:
    """Render polylines to PIL Image."""
    rendered = render_strokes_rgb(strokes, canvas_size=canvas_size)
    
    img_np = rendered.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np)


def create_comparison_figure(
    initial_img: Image.Image,
    final_img: Image.Image,
    history: dict,
    title: str,
    output_path: Path,
):
    """Create a figure comparing initial vs final with loss plots."""
    fig = plt.figure(figsize=(16, 8))
    
    # Initial sketch
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(initial_img)
    ax1.set_title("Initial Sketch", fontsize=12)
    ax1.axis('off')
    
    # Final sketch
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(final_img)
    ax2.set_title(f"Final Sketch\n(sim: {history['similarity'][-1]:.4f})", fontsize=12)
    ax2.axis('off')
    
    # Total loss
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(history['total'], 'b-', alpha=0.7, label='Total')
    ax3.set_title("Total Loss", fontsize=12)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)
    
    # Semantic loss & similarity
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(history['semantic'], 'r-', alpha=0.7, label='Semantic')
    ax4.plot(history['similarity'], 'g-', alpha=0.7, label='Similarity')
    ax4.set_title("Semantic Loss & Similarity", fontsize=12)
    ax4.set_xlabel("Step")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Perceptual loss (if available)
    ax5 = fig.add_subplot(2, 3, 5)
    if max(history.get('perceptual', [0])) > 0:
        ax5.plot(history['perceptual'], 'm-', alpha=0.7, label='Perceptual')
        ax5.set_title("Perceptual Loss", fontsize=12)
    else:
        ax5.plot(history.get('stroke_reg', [0]), 'c-', alpha=0.7, label='Stroke Reg')
        ax5.set_title("Stroke Regularization", fontsize=12)
    ax5.set_xlabel("Step")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Learning rate & stroke count
    ax6 = fig.add_subplot(2, 3, 6)
    ax6_twin = ax6.twinx()
    ax6.plot(history['lr'], 'b-', alpha=0.7, label='LR')
    ax6_twin.plot(history['num_strokes'], 'r-', alpha=0.7, label='# Strokes')
    ax6.set_title("LR & Stroke Count", fontsize=12)
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Learning Rate", color='b')
    ax6_twin.set_ylabel("# Strokes", color='r')
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved comparison figure to {output_path}")


def create_progression_grid(
    snapshots: list,
    title: str,
    output_path: Path,
    max_cols: int = 6,
):
    """Create a grid showing optimization progression."""
    n_snapshots = len(snapshots)
    n_cols = min(n_snapshots, max_cols)
    n_rows = (n_snapshots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i, (step, img, sim) in enumerate(snapshots):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(f"Step {step}\n(sim: {sim:.3f})", fontsize=10)
            axes[i].axis('off')
    
    # Hide unused axes
    for i in range(len(snapshots), len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved progression grid to {output_path}")


def test_clipasso_optimization(
    target_label: str = "a fish",
    steps: int = 1000,
    use_bezier: bool = True,
    use_augmentations: bool = True,
    use_perceptual: bool = False,  # Requires target image
    num_strokes: int = 8,
    init_strokes: list = None,
    output_dir: Path = None,
):
    """
    Test the CLIPasso-style optimizer.
    
    Args:
        target_label: Text target for optimization.
        steps: Number of optimization steps.
        use_bezier: Use Bézier curves.
        use_augmentations: Apply augmentations during CLIP encoding.
        use_perceptual: Use perceptual loss (requires target image).
        num_strokes: Number of strokes.
        init_strokes: Optional initial strokes.
        output_dir: Output directory for results.
    """
    print(f"\n{'='*60}")
    print(f"CLIPasso-Style Optimization Test")
    print(f"{'='*60}")
    print(f"Target: '{target_label}'")
    print(f"Steps: {steps}")
    print(f"Bézier: {use_bezier}")
    print(f"Augmentations: {use_augmentations}")
    print(f"Perceptual Loss: {use_perceptual}")
    print(f"Strokes: {num_strokes}")
    
    # Setup
    device = get_device()
    print(f"Device: {device}")
    
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "clipasso_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP
    print("\n📦 Loading CLIP model...")
    clip_model = ClipModel(device=device, use_float32=True)
    
    # Configure optimizer
    config = CLIPassoConfig(
        target_label=target_label,
        use_bezier=use_bezier,
        num_strokes=num_strokes,
        steps=steps,
        lr_points=1.0 if use_bezier else 2.0,
        lr_width=0.1,
        lr_opacity=0.01,
        lr_schedule="warmup_cosine",
        warmup_steps=min(100, steps // 10),
        canvas_size=224,
        stroke_width=3.0,
        use_augmentations=use_augmentations,
        num_augments=4 if use_augmentations else 1,
        augment_type="standard",
        crop_scale=(0.7, 1.0),
        perspective_scale=0.15,
        rotation_degrees=10.0,
        use_perceptual_loss=use_perceptual,
        perceptual_weight=0.3,
        semantic_weight=1.0,
        use_multi_prompt=True,
        stroke_reg_weight=0.01,
        tv_weight=0.001,
        progressive_strokes=True,
        stroke_add_interval=steps // 5,  # Add stroke every 20%
        max_strokes=num_strokes + 4,
        noise_scale=0.1,
        noise_decay=0.995,
        clip_grad_norm=1.0,
        use_tqdm=True,
    )
    
    # Create optimizer
    optimizer = CLIPassoOptimizer(clip_model, config)
    
    # Snapshot collection
    snapshots = []
    snapshot_interval = max(1, steps // 20)  # ~20 snapshots
    
    def snapshot_callback(step, losses, sketch):
        if step % snapshot_interval == 0 or step == steps - 1:
            img = render_sketch_to_pil(sketch)
            snapshots.append((step, img, losses['similarity']))
    
    # Render initial state
    if init_strokes is not None:
        initial_sketch = BezierSketch(
            num_strokes=0,
            canvas_size=224,
            device=device,
            init_strokes=init_strokes,
        )
    else:
        initial_sketch = BezierSketch(
            num_strokes=num_strokes,
            canvas_size=224,
            device=device,
        )
    initial_img = render_sketch_to_pil(initial_sketch)
    
    # Run optimization
    print(f"\n🎨 Starting CLIPasso optimization...")
    result = optimizer.optimize(init_strokes=init_strokes, callback=snapshot_callback)
    
    # Results
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"{'='*60}")
    print(f"Final Loss: {result.final_loss:.4f}")
    print(f"Final Similarity: {result.final_similarity:.4f}")
    print(f"Final Strokes: {len(result.sketch)}")
    
    # Render final
    final_img = render_sketch_to_pil(result.sketch)
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = target_label.replace(" ", "_").replace("/", "_")
    
    # Save images
    initial_img.save(output_dir / f"{safe_label}_initial_{timestamp}.png")
    final_img.save(output_dir / f"{safe_label}_final_{timestamp}.png")
    print(f"\n💾 Saved initial and final images")
    
    # Create comparison figure
    create_comparison_figure(
        initial_img,
        final_img,
        result.loss_history,
        f"CLIPasso: '{target_label}' (Bézier={use_bezier}, Aug={use_augmentations})",
        output_dir / f"{safe_label}_comparison_{timestamp}.png",
    )
    
    # Create progression grid
    if snapshots:
        create_progression_grid(
            snapshots,
            f"Optimization Progression: '{target_label}'",
            output_dir / f"{safe_label}_progression_{timestamp}.png",
        )
    
    # Create animation
    if snapshots:
        frames = [s[1] for s in snapshots]
        gif_path = output_dir / f"{safe_label}_animation_{timestamp}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"🎬 Saved animation to {gif_path}")
    
    return result


def test_with_quickdraw_init(
    category: str = "fish",
    target_label: str = None,
    steps: int = 1000,
):
    """Test CLIPasso with QuickDraw sketch as initialization."""
    print(f"\n{'='*60}")
    print(f"CLIPasso with QuickDraw Initialization")
    print(f"{'='*60}")
    
    if not HAS_QUICKDRAW:
        print("❌ QuickDraw loading not available")
        return None
    
    if target_label is None:
        target_label = f"a {category}"
    
    # Load QuickDraw sketch
    print(f"\n📦 Loading QuickDraw '{category}' sample...")
    
    # Try to find QuickDraw data
    data_dir = PROJECT_ROOT / "data" / "quickdraw"
    if not data_dir.exists():
        print(f"❌ QuickDraw data directory not found: {data_dir}")
        print("   Run without --quickdraw flag for random initialization.")
        return None
    
    try:
        samples = load_vector_samples(
            data_dir,
            [category],
            samples_per_label=1,
        )
        
        if len(samples) == 0:
            print(f"❌ No samples found for '{category}'")
            return None
        
        # Convert sample to strokes
        # Vector samples are lists of strokes, each stroke is [x_points, y_points]
        device = get_device()
        strokes = []
        for stroke_data in samples[0]:
            x_points, y_points = stroke_data
            # Normalize to 0-224 range
            x = np.array(x_points) * 224 / 255
            y = np.array(y_points) * 224 / 255
            points = np.stack([x, y], axis=1)
            strokes.append(torch.tensor(points, dtype=torch.float32, device=device))
        
        print(f"✅ Loaded sketch with {len(strokes)} strokes")
        
    except Exception as e:
        print(f"❌ Error loading QuickDraw data: {e}")
        print("   Run without --quickdraw flag for random initialization.")
        return None
    
    # Run CLIPasso optimization
    result = test_clipasso_optimization(
        target_label=target_label,
        steps=steps,
        use_bezier=True,
        use_augmentations=True,
        use_perceptual=False,
        num_strokes=len(strokes),
        init_strokes=strokes,
        output_dir=PROJECT_ROOT / "outputs" / "clipasso_quickdraw",
    )
    
    return result


def compare_with_and_without_features():
    """Compare optimization with different feature combinations."""
    print("\n" + "="*70)
    print("COMPARISON: With vs Without CLIPasso Features")
    print("="*70)
    
    configs = [
        ("Baseline (no features)", False, False),
        ("+ Bézier curves", True, False),
        ("+ Augmentations", False, True),
        ("+ Both (Full CLIPasso)", True, True),
    ]
    
    results = {}
    
    for name, use_bezier, use_aug in configs:
        print(f"\n\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        result = test_clipasso_optimization(
            target_label="a cat",
            steps=500,  # Shorter for comparison
            use_bezier=use_bezier,
            use_augmentations=use_aug,
            num_strokes=8,
            output_dir=PROJECT_ROOT / "outputs" / "clipasso_comparison",
        )
        
        results[name] = {
            'similarity': result.final_similarity,
            'loss': result.final_loss,
        }
    
    # Print comparison
    print("\n\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Configuration':<35} {'Similarity':>12} {'Loss':>12}")
    print("-"*70)
    for name, metrics in results.items():
        print(f"{name:<35} {metrics['similarity']:>12.4f} {metrics['loss']:>12.4f}")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CLIPasso-style optimizer")
    parser.add_argument("--target", type=str, default="a fish", help="Target label")
    parser.add_argument("--steps", type=int, default=1000, help="Optimization steps")
    parser.add_argument("--strokes", type=int, default=8, help="Number of strokes")
    parser.add_argument("--no-bezier", action="store_true", help="Disable Bézier curves")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentations")
    parser.add_argument("--quickdraw", type=str, help="Use QuickDraw category as init")
    parser.add_argument("--compare", action="store_true", help="Run comparison test")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_and_without_features()
    elif args.quickdraw:
        test_with_quickdraw_init(
            category=args.quickdraw,
            target_label=args.target,
            steps=args.steps,
        )
    else:
        test_clipasso_optimization(
            target_label=args.target,
            steps=args.steps,
            use_bezier=not args.no_bezier,
            use_augmentations=not args.no_augment,
            num_strokes=args.strokes,
        )


if __name__ == "__main__":
    main()

