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


def load_reference_image(
    image_path: str,
    clip_model: ClipModel,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load a reference image (e.g., a perfect sketch) and encode it with CLIP.
    
    This is the KEY for sketch refinement:
    - Instead of optimizing toward generic text like "a fish"
    - We optimize toward what a PERFECT fish sketch looks like
    - This is the INVERSE of CLIPasso (photo→sketch becomes bad_sketch→good_sketch)
    
    Args:
        image_path: Path to the reference image (e.g., perfectFish.jpeg)
        clip_model: CLIP model for encoding
        device: Device for tensors
    
    Returns:
        CLIP embedding of the reference image [1, D]
    """
    from torchvision import transforms
    
    # Load the image
    img = Image.open(image_path).convert("RGB")
    
    # Preprocess for CLIP (resize to 224x224, normalize)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])
    
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Encode with CLIP
    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor, requires_grad=False)
    
    print(f"  Reference image embedding shape: {embedding.shape}")
    print(f"  Loaded reference: {image_path}")
    
    return embedding


def load_photo_exemplars(
    image_dir: str,
    clip_model: ClipModel,
    device: str = "cpu",
    max_images: int = 50,
) -> torch.Tensor:
    """
    Load real photos from a directory and encode them with CLIP as exemplars.
    
    This lets you guide the sketch toward what REAL fish/cats/etc look like,
    using the rich semantic understanding CLIP learned from photos.
    
    Args:
        image_dir: Directory containing photos (jpg, png, jpeg)
        clip_model: CLIP model for encoding
        device: Device for tensors
        max_images: Maximum number of images to load
    
    Returns:
        Tensor of CLIP embeddings [N, D]
    """
    from torchvision import transforms
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Warning: {image_dir} not found, returning None")
        return None
    
    # Find all images
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(ext))
    
    if not image_paths:
        print(f"Warning: No images found in {image_dir}")
        return None
    
    # Limit number of images
    if len(image_paths) > max_images:
        image_paths = random.sample(image_paths, max_images)
    
    print(f"  Loading {len(image_paths)} photo exemplars from '{image_dir}'...")
    
    # CLIP preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])
    
    # Encode each image
    embeddings = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = clip_model.encode_image(img_tensor, requires_grad=False)
            embeddings.append(emb)
        except Exception as e:
            print(f"    Skipping {img_path.name}: {e}")
    
    if not embeddings:
        return None
    
    # Stack into single tensor
    photo_embeddings = torch.cat(embeddings, dim=0)
    print(f"  Photo exemplar embeddings shape: {photo_embeddings.shape}")
    
    return photo_embeddings


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
    use_multi_prompt=True,
    refinement_mode=True,
    original_weight=0.25,
    original_weight_schedule="constant",  # "constant", "linear_up", "cosine_up"
    original_weight_start=0.1,  # Starting weight (for scheduled modes)
    # Exemplar-guided refinement (QuickDraw sketches)
    use_exemplar_loss=False,
    exemplar_category=None,  # QuickDraw category for exemplars
    n_exemplars=20,          # Number of exemplar sketches
    exemplar_weight=0.5,     # Weight for exemplar loss
    exemplar_mode="mean",    # "mean", "min", "softmin"
    # Photo exemplar guidance (real photos instead of sketches)
    use_photo_exemplars=True,
    photo_exemplar_dir="data/photos/fish/",  # Directory with real photos (e.g., "data/photos/fish/")
    photo_exemplar_weight=0.5,  # Weight for photo exemplar loss
    # Reference image guidance (use a perfect sketch as target)
    use_reference_image=False,
    reference_image_path=None,  # Path to perfect sketch (e.g., perfectFish.jpeg)
    reference_image_weight=1.0,  # Weight for reference image loss
    # Stroke addition during optimization
    allow_stroke_addition=False,
    stroke_add_interval=3,  # Add stroke every N steps
    max_strokes=10,           # Maximum strokes to add
    new_stroke_points=10,      # Control points per new stroke
    stroke_init_mode="edge",  # "random", "center", "edge"
    # ===== CLIPasso-style features =====
    use_bezier=False,         # Use Bézier curves for smoother strokes
    bezier_samples=50,        # Points to sample per Bézier curve
    use_augmentations=False,  # Apply augmentations during CLIP encoding
    num_augments=4,           # Number of augmented views
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
    clip_model = ClipModel(device=device, use_float32=True)
    
    # Load exemplar embeddings if requested (QuickDraw sketches)
    exemplar_embeddings = None
    if use_exemplar_loss and exemplar_category:
        exemplar_embeddings = load_quickdraw_exemplars(
            category=exemplar_category,
            n_exemplars=n_exemplars,
            clip_model=clip_model,
            canvas_size=224,
            device=device,
        )
    
    # Load photo exemplar embeddings if requested (real photos)
    if use_photo_exemplars and photo_exemplar_dir:
        photo_embeddings = load_photo_exemplars(
            image_dir=photo_exemplar_dir,
            clip_model=clip_model,
            device=device,
        )
        # Combine with sketch exemplars if both are used, otherwise use photo embeddings
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
            device=device,
        )
    
    # Build loss weights
    # SEMANTIC is the CLIP text embedding loss - increase to emphasize text more
    loss_weights = {
        "semantic": 3.0,  # HIGH weight for CLIP text guidance (was 1.0)
        "stroke": 0.01,   # Lower regularization to allow more freedom
        "tv": 0.0005,     # Lower TV to allow more changes
    }
    if use_exemplar_loss or use_photo_exemplars:
        # Use photo weight if using photos, otherwise sketch weight
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
        min_lr=0.5,  # Higher min LR - less decay (was 0.05)
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
        # Exemplar settings (works for both sketch and photo exemplars)
        use_exemplar_loss=use_exemplar_loss or use_photo_exemplars,
        exemplar_embeddings=exemplar_embeddings,
        exemplar_weight=photo_exemplar_weight if use_photo_exemplars else exemplar_weight,
        exemplar_mode=exemplar_mode,
        # Reference image settings
        use_reference_image=use_reference_image,
        reference_image_embedding=reference_image_embedding,
        reference_image_weight=reference_image_weight,
        # Stroke addition settings
        allow_stroke_addition=allow_stroke_addition,
        # CLIPasso-style features
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
    print(f"    Final strokes: {len(result.final_strokes)}")
    
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
        target_label = "a perfect sketch of a fish"  
    
    print(f"\nSource: {source_label}")
    print(f"Target: {target_label}")
    print(f"Strokes: {len(init_strokes)}")
    
    # Detect if this is a refinement (same category) or transformation
    is_refinement = source_label in target_label or target_label.replace("a ", "") in source_label
    
    if is_refinement:
        print(f"\n🔧 REFINEMENT MODE: Improving '{source_label}' while preserving structure")
    else:
        print(f"\n🔄 TRANSFORMATION MODE: Converting '{source_label}' → '{target_label}'")
    
    # Check if we have a perfect reference image for this category
    reference_image_path = Path(PROJECT_ROOT) / f"perfect{source_label.capitalize()}.jpeg"
    has_reference_image = reference_image_path.exists()
    
    # Check if we have a directory of real photos for this category
    photo_dir = Path(PROJECT_ROOT) / "data" / "photos" / source_label
    has_photo_exemplars = photo_dir.exists() and any(photo_dir.glob("*"))
    
    if has_reference_image:
        print(f"\n✨ Found reference image: {reference_image_path}")
        print("   Using REFERENCE IMAGE GUIDANCE (inverse CLIPasso approach)")
    
    if has_photo_exemplars:
        print(f"\n📷 Found photo exemplars: {photo_dir}")
        print("   Using REAL PHOTOS to guide toward realistic semantics")
    
    # Run optimization with appropriate settings
    result, snapshots = run_long_optimization(
        init_strokes=init_strokes,
        target_label=target_label,
        source_label=source_label,
        steps=1000,                                   # More steps for better convergence
        snapshot_every=50,
        noise_scale=0.3 if is_refinement else 0.4,   # Moderate noise for exploration
        lr=2.5 if is_refinement else 3.0,            # Higher LR for faster initial progress
        use_multi_prompt=True,                        # Always use multi-prompt now!
        refinement_mode=is_refinement,                # Preserve original when refining
        # ORIGINAL WEIGHT: Lower = more freedom for CLIP text to change the sketch
        original_weight=0.1 if is_refinement else 0,  # LOW - let text dominate!
        original_weight_schedule="constant",          # Keep it low throughout
        original_weight_start=0.05,                   # Start very low
        # REFERENCE IMAGE: Disabled to let text dominate
        use_reference_image=False,                    # OFF - pure text guidance
        reference_image_path=None,
        reference_image_weight=0.0,                   # Unused
        # PHOTO EXEMPLARS: Guide toward real photos (rich semantics)
        use_photo_exemplars=has_photo_exemplars,
        photo_exemplar_dir=str(photo_dir) if has_photo_exemplars else None,
        photo_exemplar_weight=0.5,                    # Balance with text-based semantic loss
        # Disable QuickDraw sketch exemplars (using photos or reference image instead)
        use_exemplar_loss=False,
        exemplar_category=None,
        n_exemplars=0,
        exemplar_weight=0.0,
        exemplar_mode="mean",
        # STROKE ADDITION: Disabled - keep original stroke count
        allow_stroke_addition=True,    # No new strokes
        stroke_add_interval=10,
        max_strokes=1,
        new_stroke_points=4,
        stroke_init_mode="random",
        # ===== CLIPasso-style features =====
        use_bezier=True,           # 🔷 ENABLED - smoother Bézier curves!
        bezier_samples=50,         # Points per curve
        use_augmentations=True,    # 🔄 ENABLED - robust CLIP guidance!
        num_augments=4,            # Number of augmented views
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
    print(f"Final strokes: {len(result.final_strokes)}")


if __name__ == "__main__":
    main()

