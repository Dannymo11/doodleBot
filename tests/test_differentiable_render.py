#!/usr/bin/env python3
"""
Test the differentiable soft line renderer.

Verifies:
1. Forward pass produces valid images
2. Gradients flow back to stroke parameters
3. Integration with CLIP for semantic loss optimization
"""

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import matplotlib.pyplot as plt

from src.inverse_clipasso.render import render_strokes_soft, render_strokes_rgb


def test_basic_rendering():
    """Test that rendering produces valid output."""
    print("=" * 60)
    print("Test 1: Basic Rendering")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create simple stroke: diagonal line
    stroke1 = torch.tensor([
        [20., 20.],
        [200., 200.]
    ], device=device, requires_grad=True)
    
    # Create another stroke: horizontal line
    stroke2 = torch.tensor([
        [20., 100.],
        [200., 100.]
    ], device=device, requires_grad=True)
    
    strokes = [stroke1, stroke2]
    
    # Render
    img = render_strokes_soft(strokes, canvas_size=224, stroke_width=3.0)
    
    print(f"  Output shape: {img.shape}")
    print(f"  Output range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  Device: {img.device}")
    print(f"  Requires grad: {img.requires_grad}")
    
    assert img.shape == (1, 1, 224, 224), f"Wrong shape: {img.shape}"
    assert img.min() >= 0 and img.max() <= 1, "Values out of [0,1] range"
    assert img.requires_grad, "Output should require gradients"
    
    print("  ✓ Basic rendering passed!")
    return img, strokes


def test_gradient_flow():
    """Test that gradients flow back to stroke parameters."""
    print("\n" + "=" * 60)
    print("Test 2: Gradient Flow")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a stroke with requires_grad=True
    stroke = torch.tensor([
        [50., 50.],
        [150., 150.],
        [150., 50.]
    ], device=device, requires_grad=True)
    
    # Render
    img = render_strokes_soft([stroke], canvas_size=224)
    
    # Compute a simple loss (e.g., mean pixel value)
    loss = img.mean()
    
    # Backward pass
    loss.backward()
    
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Stroke grad shape: {stroke.grad.shape}")
    print(f"  Stroke grad norm: {stroke.grad.norm().item():.6f}")
    print(f"  Stroke grad sample:\n{stroke.grad}")
    
    assert stroke.grad is not None, "Gradient should exist"
    assert stroke.grad.shape == stroke.shape, "Gradient shape mismatch"
    assert stroke.grad.norm() > 0, "Gradient should be non-zero"
    
    print("  ✓ Gradient flow passed!")


def test_optimization_step():
    """Test that we can optimize stroke positions via gradient descent."""
    print("\n" + "=" * 60)
    print("Test 3: Optimization Step")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a stroke
    stroke = torch.tensor([
        [50., 112.],
        [174., 112.]
    ], device=device, requires_grad=True)
    
    # Target: we want the stroke to be more centered (around 112, 112)
    # Loss: distance of rendered "ink" center of mass from image center
    
    optimizer = torch.optim.Adam([stroke], lr=5.0)
    
    print("  Optimizing stroke position...")
    initial_pos = stroke.clone().detach()
    
    for step in range(10):
        optimizer.zero_grad()
        
        img = render_strokes_soft([stroke], canvas_size=224)
        
        # Loss: we want more ink in the center of the image
        # Create a "center preference" mask
        coords = torch.arange(224, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        center_dist = ((xx - 112) ** 2 + (yy - 112) ** 2) / (112 ** 2)
        
        # Ink is where image is dark (close to 0)
        ink = 1 - img.squeeze()
        
        # Loss: ink weighted by distance from center (minimize this)
        loss = (ink * center_dist).sum() / (ink.sum() + 1e-8)
        
        loss.backward()
        optimizer.step()
        
        if step % 3 == 0:
            print(f"    Step {step}: loss = {loss.item():.4f}")
    
    final_pos = stroke.clone().detach()
    movement = (final_pos - initial_pos).norm().item()
    
    print(f"  Initial position:\n{initial_pos}")
    print(f"  Final position:\n{final_pos}")
    print(f"  Total movement: {movement:.2f} pixels")
    
    assert movement > 0.1, "Stroke should have moved during optimization"
    
    print("  ✓ Optimization step passed!")


def test_rgb_rendering():
    """Test RGB rendering for CLIP compatibility."""
    print("\n" + "=" * 60)
    print("Test 4: RGB Rendering")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    stroke = torch.tensor([
        [30., 30.],
        [194., 194.]
    ], device=device, requires_grad=True)
    
    img_rgb = render_strokes_rgb([stroke], canvas_size=224)
    
    print(f"  Output shape: {img_rgb.shape}")
    print(f"  Channels identical: {torch.allclose(img_rgb[0, 0], img_rgb[0, 1])}")
    
    assert img_rgb.shape == (1, 3, 224, 224), f"Wrong shape: {img_rgb.shape}"
    assert img_rgb.requires_grad, "Should require gradients"
    
    print("  ✓ RGB rendering passed!")


def test_clip_integration():
    """Test integration with CLIP model."""
    print("\n" + "=" * 60)
    print("Test 5: CLIP Integration")
    print("=" * 60)
    
    try:
        from src.inverse_clipasso.clip.clip_api import ClipModel
    except ImportError as e:
        print(f"  ⚠ Skipping CLIP test: {e}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a simple "house" shape (triangle + rectangle)
    roof = torch.tensor([
        [112., 50.],
        [50., 120.],
        [174., 120.],
        [112., 50.]
    ], device=device, requires_grad=True)
    
    walls = torch.tensor([
        [60., 120.],
        [60., 180.],
        [164., 180.],
        [164., 120.]
    ], device=device, requires_grad=True)
    
    strokes = [roof, walls]
    
    # Render to RGB
    img = render_strokes_rgb(strokes, canvas_size=224)
    
    # Load CLIP (use_float32=True for gradient-based optimization)
    clip_model = ClipModel(device=device, use_float32=True)
    
    # Get embeddings
    # Note: CLIP expects preprocessed input, but our renderer outputs [0,1] RGB
    # We need to normalize properly
    img_for_clip = img  # Already [1, 3, 224, 224]
    
    # CLIP's preprocess expects PIL, so we'll do manual normalization
    # CLIP normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    img_normalized = (img_for_clip - mean) / std
    
    # Encode (requires_grad=True for optimization)
    img_emb = clip_model.encode_image(img_normalized, requires_grad=True)
    text_emb = clip_model.encode_text("a house")
    
    # Compute similarity
    similarity = (img_emb @ text_emb.T).squeeze()
    loss = 1 - similarity  # Semantic loss
    
    print(f"  Similarity to 'a house': {similarity.item():.4f}")
    print(f"  Semantic loss: {loss.item():.4f}")
    
    # Verify gradients flow
    loss.backward()
    
    print(f"  Roof grad norm: {roof.grad.norm().item():.6f}")
    print(f"  Walls grad norm: {walls.grad.norm().item():.6f}")
    
    assert roof.grad is not None and walls.grad is not None, "Gradients should exist"
    
    print("  ✓ CLIP integration passed!")


def visualize_rendering():
    """Create visualization of the renderer output."""
    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a more complex drawing
    strokes = [
        # Triangle (roof)
        torch.tensor([
            [112., 30.],
            [40., 100.],
            [184., 100.],
            [112., 30.]
        ], device=device),
        # Rectangle (house body)
        torch.tensor([
            [50., 100.],
            [50., 180.],
            [174., 180.],
            [174., 100.],
            [50., 100.]
        ], device=device),
        # Door
        torch.tensor([
            [95., 180.],
            [95., 140.],
            [129., 140.],
            [129., 180.]
        ], device=device),
    ]
    
    img = render_strokes_soft(strokes, canvas_size=224, stroke_width=2.5)
    
    # Save visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img.squeeze().detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Soft Rendered (Differentiable)')
    axes[0].axis('off')
    
    # Also show with different stroke widths
    img_thick = render_strokes_soft(strokes, canvas_size=224, stroke_width=5.0)
    axes[1].imshow(img_thick.squeeze().detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Thick Strokes (width=5)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_render_output.png', dpi=150)
    print(f"  Saved visualization to test_render_output.png")
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DIFFERENTIABLE RENDERER TESTS")
    print("=" * 60)
    
    test_basic_rendering()
    test_gradient_flow()
    test_optimization_step()
    test_rgb_rendering()
    test_clip_integration()
    visualize_rendering()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)

