"""
Experiment runners for sketch optimization.

Provides high-level functions for running optimization experiments
with various configurations and capturing results.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch

from src.inverse_clipasso.clip.clip_api import ClipModel
from src.inverse_clipasso.optimize import InverseClipassoOptimizer, OptimizationConfig
from .data_loaders import load_quickdraw_exemplars, load_reference_image, load_photo_exemplars
from .visualization import render_to_pil


__all__ = [
    "run_long_optimization",
    "get_best_device",
]


def get_best_device() -> str:
    """
    Get the best available device for optimization.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    # Skip MPS - it's ~15x slower for this workload due to small tensor overhead
    # elif torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


def run_long_optimization(
    init_strokes: List[torch.Tensor],
    target_label: str,
    source_label: str = "sketch",
    steps: int = 1000,
    snapshot_every: int = 50,
    noise_scale: float = 0.3,
    lr: float = 3.0,
    use_multi_prompt: bool = False,
    refinement_mode: bool = False,
    original_weight: float = 0.5,
    original_weight_schedule: str = "constant",
    original_weight_start: float = 0.1,
    # Exemplar-guided refinement (QuickDraw sketches)
    use_exemplar_loss: bool = False,
    exemplar_category: Optional[str] = None,
    n_exemplars: int = 20,
    exemplar_weight: float = 0.5,
    exemplar_mode: str = "mean",
    # Photo exemplar guidance
    use_photo_exemplars: bool = False,
    photo_exemplar_dir: Optional[str] = None,
    photo_exemplar_weight: float = 0.5,
    # Reference image guidance
    use_reference_image: bool = False,
    reference_image_path: Optional[str] = None,
    reference_image_weight: float = 1.0,
    # Stroke addition
    allow_stroke_addition: bool = False,
    stroke_add_interval: int = 200,
    max_strokes: int = 20,
    new_stroke_points: int = 4,
    stroke_init_mode: str = "random",
    # CLIPasso-style features
    use_bezier: bool = False,
    bezier_samples: int = 50,
    use_augmentations: bool = False,
    num_augments: int = 4,
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
        original_weight: Weight for original strokes preservation.
        original_weight_schedule: Schedule for original weight.
        original_weight_start: Starting weight for scheduled modes.
        use_exemplar_loss: If True, guide toward QuickDraw exemplars.
        exemplar_category: QuickDraw category for loading exemplars.
        n_exemplars: Number of exemplar sketches to load.
        exemplar_weight: Weight for exemplar loss.
        exemplar_mode: How to aggregate exemplar similarities.
        use_photo_exemplars: If True, guide toward photo exemplars.
        photo_exemplar_dir: Directory with real photos.
        photo_exemplar_weight: Weight for photo exemplar loss.
        use_reference_image: If True, guide toward reference image.
        reference_image_path: Path to reference image.
        reference_image_weight: Weight for reference image loss.
        allow_stroke_addition: If True, allow adding strokes.
        stroke_add_interval: Steps between adding strokes.
        max_strokes: Maximum number of strokes.
        new_stroke_points: Points per new stroke.
        stroke_init_mode: How to initialize new strokes.
        use_bezier: If True, use Bézier curves.
        bezier_samples: Points per Bézier curve.
        use_augmentations: If True, apply augmentations.
        num_augments: Number of augmented views.
    
    Returns:
        Tuple of (result, snapshots) where:
        - result: OptimizationResult from the optimizer
        - snapshots: List of (step, similarity, PIL.Image) tuples
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
        print(f"  Sketch exemplar loss: {exemplar_category} ({n_exemplars} samples, weight={exemplar_weight}, mode={exemplar_mode})")
    if use_photo_exemplars:
        print(f"  Photo exemplar loss: {photo_exemplar_dir} (weight={photo_exemplar_weight}, mode={exemplar_mode})")
    if use_reference_image:
        print(f"  Reference image: {reference_image_path} (weight={reference_image_weight})")
    if allow_stroke_addition:
        print(f"  Stroke addition: every {stroke_add_interval} steps, max {max_strokes} strokes ({stroke_init_mode})")
    if use_bezier:
        print(f"  🔷 Bézier curves: ENABLED ({bezier_samples} samples)")
    if use_augmentations:
        print(f"  🔄 Augmentations: ENABLED ({num_augments} views)")
    print(f"  Device: {device}")
    
    # Load CLIP
    print("  Loading CLIP model...")
    clip_model = ClipModel(device=str(device), use_float32=True)
    
    # Load exemplar embeddings if requested (QuickDraw sketches)
    exemplar_embeddings = None
    if use_exemplar_loss and exemplar_category:
        exemplar_embeddings = load_quickdraw_exemplars(
            category=exemplar_category,
            n_exemplars=n_exemplars,
            clip_model=clip_model,
            canvas_size=224,
            device=str(device),
        )
    
    # Load photo exemplar embeddings if requested
    if use_photo_exemplars and photo_exemplar_dir:
        photo_embeddings = load_photo_exemplars(
            image_dir=photo_exemplar_dir,
            clip_model=clip_model,
            device=str(device),
        )
        # Combine with sketch exemplars if both are used
        if exemplar_embeddings is not None and photo_embeddings is not None:
            print(f"  Combining {exemplar_embeddings.shape[0]} sketch + {photo_embeddings.shape[0]} photo exemplars")
            exemplar_embeddings = torch.cat([exemplar_embeddings, photo_embeddings], dim=0)
        elif photo_embeddings is not None:
            exemplar_embeddings = photo_embeddings
    
    # Load reference image embedding if requested
    reference_image_embedding = None
    if use_reference_image and reference_image_path:
        reference_image_embedding = load_reference_image(
            image_path=reference_image_path,
            clip_model=clip_model,
            device=str(device),
        )
    
    # Build loss weights
    loss_weights = {
        "semantic": 3.0,
        "stroke": 0.01,
        "tv": 0.0005,
    }
    if use_exemplar_loss or use_photo_exemplars:
        loss_weights["exemplar"] = photo_exemplar_weight if use_photo_exemplars else exemplar_weight
    if use_reference_image:
        loss_weights["reference"] = reference_image_weight
    
    # Configure optimization
    config = OptimizationConfig(
        label=target_label,
        steps=steps,
        lr=lr,
        lr_schedule="warmup_cosine",
        warmup_steps=min(100, steps // 10),
        min_lr=0.5,
        noise_scale=noise_scale,
        noise_decay=0.995,
        loss_weights=loss_weights,
        use_tqdm=True,
        use_multi_prompt=use_multi_prompt,
        refinement_mode=refinement_mode,
        original_weight=original_weight,
        original_weight_schedule=original_weight_schedule,
        original_weight_start=original_weight_start,
        use_exemplar_loss=use_exemplar_loss or use_photo_exemplars,
        exemplar_embeddings=exemplar_embeddings,
        exemplar_weight=photo_exemplar_weight if use_photo_exemplars else exemplar_weight,
        exemplar_mode=exemplar_mode,
        use_reference_image=use_reference_image,
        reference_image_embedding=reference_image_embedding,
        reference_image_weight=reference_image_weight,
        allow_stroke_addition=allow_stroke_addition,
        use_bezier=use_bezier,
        bezier_samples=bezier_samples,
        use_augmentations=use_augmentations,
        num_augments=num_augments,
        stroke_add_interval=stroke_add_interval,
        max_strokes=max_strokes,
        new_stroke_points=new_stroke_points,
        stroke_init_mode=stroke_init_mode,
    )
    
    # Callback to capture snapshots
    def snapshot_callback(step, losses, strokes):
        if step % snapshot_every == 0 or step == steps - 1:
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
    print(f"    Final strokes: {len(result.final_strokes)}")
    
    return result, snapshots


