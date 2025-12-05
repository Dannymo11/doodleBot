#!/usr/bin/env python3
"""
Reference-Image-Anchored Sketch Optimization.

This follows the ORIGINAL CLIPasso approach more closely:
- Use a reference image (perfectFish.jpeg) as the PRIMARY visual anchor
- The goal is to make the sketch LOOK LIKE the reference
- Text is secondary/optional
- Strong perceptual loss preserves structure
"""

import sys
import os
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.inverse_clipasso.clip.clip_api import ClipModel
from src.inverse_clipasso.render import render_strokes_rgb


# CLIP normalization
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def load_image(path: str, size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for CLIP."""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def normalize_for_clip(img: torch.Tensor, device: str) -> torch.Tensor:
    """Apply CLIP normalization."""
    mean = CLIP_MEAN.view(1, 3, 1, 1).to(device)
    std = CLIP_STD.view(1, 3, 1, 1).to(device)
    return (img.to(device) - mean) / std


def quickdraw_to_strokes(drawing, size=224, padding=10, device="cpu"):
    """Convert QuickDraw drawing to stroke tensors."""
    xs_all, ys_all = [], []
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
    """Render strokes to PIL Image."""
    img_tensor = render_strokes_rgb(strokes, canvas_size=canvas_size)
    img_np = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


class ReferenceAnchoredOptimizer:
    """
    Optimize a sketch to match a reference image using CLIP features.
    
    This is closer to the original CLIPasso approach:
    - Reference image provides the visual anchor
    - Perceptual loss at multiple CLIP layers
    - Structure is preserved by matching the reference
    """
    
    def __init__(
        self,
        clip_model: ClipModel,
        reference_image: torch.Tensor,
        device: str = "cpu",
    ):
        self.clip = clip_model
        self.device = device
        
        # Pre-compute reference image embedding and features
        ref_normalized = normalize_for_clip(reference_image, device)
        
        with torch.no_grad():
            self.reference_embedding = self.clip.encode_image(ref_normalized)
        
        print(f"Reference embedding shape: {self.reference_embedding.shape}")
    
    def optimize(
        self,
        init_strokes: list,
        steps: int = 500,
        lr: float = 0.5,
        structure_weight: float = 0.5,  # How much to preserve original structure
        canvas_size: int = 224,
        stroke_width: float = 3.0,
        use_tqdm: bool = True,
    ):
        """
        Optimize strokes to match reference image.
        
        Args:
            init_strokes: Initial stroke positions
            steps: Number of optimization steps
            lr: Learning rate
            structure_weight: Weight for preserving original stroke positions (0-1)
            canvas_size: Canvas size
            stroke_width: Stroke width
        """
        from tqdm import tqdm
        
        # Clone strokes
        strokes = []
        for s in init_strokes:
            stroke = s.clone().detach().to(self.device).requires_grad_(True)
            strokes.append(stroke)
        
        # Save original positions for structure preservation
        original_strokes = [s.clone().detach() for s in strokes]
        
        # Optimizer
        optimizer = torch.optim.Adam(strokes, lr=lr)
        
        # History
        history = {
            "total": [], "perceptual": [], "structure": [], "similarity": []
        }
        
        # Snapshots
        snapshots = []
        snapshot_every = max(1, steps // 20)
        
        steps_iter = range(steps)
        if use_tqdm:
            steps_iter = tqdm(steps_iter, desc="Optimizing toward reference")
        
        for step in steps_iter:
            optimizer.zero_grad()
            
            # Render current strokes
            rendered = render_strokes_rgb(
                strokes,
                canvas_size=canvas_size,
                stroke_width=stroke_width,
            )
            
            # Normalize for CLIP
            rendered_norm = normalize_for_clip(rendered, self.device)
            
            # Get current embedding
            current_embedding = self.clip.encode_image(rendered_norm, requires_grad=True)
            
            # Perceptual loss: similarity to reference image
            similarity = torch.nn.functional.cosine_similarity(
                current_embedding, self.reference_embedding
            ).mean()
            perceptual_loss = 1.0 - similarity
            
            # Structure preservation loss
            structure_loss = torch.tensor(0.0, device=self.device)
            if structure_weight > 0:
                for curr, orig in zip(strokes, original_strokes):
                    diff = curr - orig
                    structure_loss = structure_loss + (diff ** 2).mean()
                structure_loss = structure_loss / len(strokes)
            
            # Total loss
            total_loss = perceptual_loss + structure_weight * structure_loss
            
            # Backward
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(strokes, 1.0)
            
            # Update
            optimizer.step()
            
            # Clamp to canvas
            with torch.no_grad():
                for stroke in strokes:
                    stroke.clamp_(0, canvas_size)
            
            # Record
            history["total"].append(total_loss.item())
            history["perceptual"].append(perceptual_loss.item())
            history["structure"].append(structure_loss.item())
            history["similarity"].append(similarity.item())
            
            if use_tqdm:
                steps_iter.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "sim": f"{similarity.item():.4f}",
                })
            
            # Snapshot
            if step % snapshot_every == 0 or step == steps - 1:
                img = render_to_pil(strokes, canvas_size)
                snapshots.append((step, similarity.item(), img))
        
        # Final strokes
        final_strokes = [s.detach().clone() for s in strokes]
        
        return {
            "strokes": final_strokes,
            "history": history,
            "snapshots": snapshots,
            "final_similarity": history["similarity"][-1],
        }


def visualize_results(
    initial_img: Image.Image,
    reference_img: Image.Image,
    result: dict,
    output_dir: Path,
    name: str = "reference_anchor",
):
    """Create visualization of results."""
    snapshots = result["snapshots"]
    history = result["history"]
    
    # Create comparison figure
    fig = plt.figure(figsize=(16, 10))
    
    # Row 1: Initial, Reference, Final
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(initial_img)
    ax1.set_title(f"Initial Sketch\n(sim: {history['similarity'][0]:.3f})", fontsize=11)
    ax1.axis("off")
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(reference_img)
    ax2.set_title("Reference Image\n(TARGET)", fontsize=11)
    ax2.axis("off")
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(snapshots[-1][2])
    ax3.set_title(f"Final Sketch\n(sim: {result['final_similarity']:.3f})", fontsize=11)
    ax3.axis("off")
    
    # Row 2: Loss curves
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(history["total"], "b-", label="Total Loss")
    ax4.plot(history["perceptual"], "r-", alpha=0.7, label="Perceptual")
    ax4.plot(history["structure"], "g-", alpha=0.7, label="Structure")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Loss")
    ax4.set_title("Loss Over Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(history["similarity"], "g-")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("CLIP Similarity")
    ax5.set_title("Similarity to Reference")
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, max(history["similarity"]) * 1.2)
    
    # Progression
    ax6 = fig.add_subplot(2, 3, 6)
    n_show = min(6, len(snapshots))
    indices = np.linspace(0, len(snapshots) - 1, n_show).astype(int)
    
    combined = Image.new("RGB", (224 * n_show, 224), "white")
    for i, idx in enumerate(indices):
        combined.paste(snapshots[idx][2], (i * 224, 0))
    
    ax6.imshow(combined)
    ax6.set_title("Progression")
    ax6.axis("off")
    
    plt.suptitle(f"Reference-Anchored Optimization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"{name}_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")
    
    # Save animation
    if len(snapshots) > 1:
        frames = [s[2] for s in snapshots]
        gif_path = output_dir / f"{name}_animation.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"Saved: {gif_path}")
    
    return fig_path


def main():
    print("\n" + "=" * 70)
    print("REFERENCE-IMAGE-ANCHORED OPTIMIZATION")
    print("=" * 70)
    print("Goal: Make QuickDraw sketch look like the reference image")
    print("This follows the ORIGINAL CLIPasso approach more closely.\n")
    
    # Check for reference image
    reference_path = Path(PROJECT_ROOT) / "data" / "photos" / "fish" / "perfectFish.jpeg"
    if not reference_path.exists():
        # Try alternate location
        reference_path = Path(PROJECT_ROOT) / "perfectFish.jpeg"
    
    if not reference_path.exists():
        print(f"❌ Reference image not found: {reference_path}")
        print("Please provide a reference sketch image (e.g., perfectFish.jpeg)")
        return
    
    print(f"✅ Reference image: {reference_path}")
    
    # Load reference image
    reference_tensor = load_image(str(reference_path))
    reference_pil = Image.open(reference_path).convert("RGB").resize((224, 224))
    
    # Load QuickDraw sketch
    data_dir = Path(PROJECT_ROOT) / "data" / "quickdraw" / "raw" / "strokes"
    fish_file = data_dir / "fish.ndjson"
    
    if not fish_file.exists():
        print(f"❌ QuickDraw data not found: {fish_file}")
        return
    
    with open(fish_file) as f:
        lines = f.readlines()
    
    sample = json.loads(random.choice(lines))
    drawing = sample["drawing"]
    
    device = "cpu"  # Use CPU for stability
    strokes = quickdraw_to_strokes(drawing, device=device)
    
    print(f"✅ Loaded QuickDraw fish with {len(strokes)} strokes")
    
    # Initial rendering
    initial_img = render_to_pil(strokes)
    
    # Load CLIP
    print("\n📦 Loading CLIP model...")
    clip_model = ClipModel(device=device, use_float32=True)
    
    # Create optimizer
    optimizer = ReferenceAnchoredOptimizer(
        clip_model=clip_model,
        reference_image=reference_tensor,
        device=device,
    )
    
    # Run optimization with different structure weights
    output_dir = Path(PROJECT_ROOT) / "outputs" / "reference_anchor"
    
    for structure_weight in [0.0, 0.3, 0.7]:
        print(f"\n{'=' * 50}")
        print(f"Structure weight: {structure_weight}")
        print(f"{'=' * 50}")
        
        result = optimizer.optimize(
            init_strokes=strokes,
            steps=500,
            lr=0.5,
            structure_weight=structure_weight,
        )
        
        print(f"\nFinal similarity: {result['final_similarity']:.4f}")
        
        visualize_results(
            initial_img=initial_img,
            reference_img=reference_pil,
            result=result,
            output_dir=output_dir,
            name=f"fish_struct{structure_weight:.1f}",
        )
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nCompare the different structure weights:")
    print("  - 0.0 = Pure reference matching (may distort shape)")
    print("  - 0.3 = Balanced (recommended)")
    print("  - 0.7 = Heavy structure preservation (subtle changes)")


if __name__ == "__main__":
    main()

