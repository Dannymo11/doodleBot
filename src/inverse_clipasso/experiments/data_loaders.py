"""
Data loading utilities for sketch optimization experiments.

Provides functions for loading:
- QuickDraw sketch exemplars
- Reference images (perfect sketches)
- Photo exemplars (real photos for semantic guidance)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.inverse_clipasso.clip.clip_api import ClipModel
from src.inverse_clipasso.render import render_strokes_rgb


__all__ = [
    "load_quickdraw_exemplars",
    "load_reference_image",
    "load_photo_exemplars",
    "quickdraw_to_stroke_params",
]


def quickdraw_to_stroke_params(
    drawing,
    size: int = 224,
    padding: int = 10,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """
    Convert QuickDraw drawing format to list of stroke parameter tensors.
    
    Args:
        drawing: QuickDraw drawing format (list of [x_coords, y_coords] pairs).
        size: Canvas size to normalize coordinates to.
        padding: Padding from canvas edges.
        device: Device for output tensors.
    
    Returns:
        List of [N, 2] tensors, one per stroke.
    """
    # Gather all coordinates to compute bounds
    xs_all = []
    ys_all = []
    for stroke in drawing:
        xs_all.append(np.array(stroke[0], dtype=np.float32))
        ys_all.append(np.array(stroke[1], dtype=np.float32))
    
    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    
    # Compute bounding box
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    w = max_x - min_x + 1e-6
    h = max_y - min_y + 1e-6
    
    # Scale to fit in canvas with padding
    scale = (size - 2 * padding) / max(w, h)
    ox = (size - scale * w) / 2.0
    oy = (size - scale * h) / 2.0
    
    # Convert each stroke
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


def load_quickdraw_exemplars(
    category: str,
    n_exemplars: int = 20,
    clip_model: Optional[ClipModel] = None,
    canvas_size: int = 224,
    device: str = "cpu",
    data_dir: Optional[Path] = None,
) -> Optional[torch.Tensor]:
    """
    Load QuickDraw sketches and encode them with CLIP to create exemplar embeddings.
    
    This gives a "cluster" of what good sketches of this category look like
    in CLIP space, which can be used to guide optimization.
    
    Args:
        category: QuickDraw category (e.g., "fish", "cat").
        n_exemplars: Number of exemplar sketches to load.
        clip_model: CLIP model for encoding. If None, returns None.
        canvas_size: Size to render sketches.
        device: Device for tensors.
        data_dir: Path to QuickDraw data. If None, uses default location.
    
    Returns:
        Tensor of CLIP embeddings [N, D], or None if data not found.
    """
    if clip_model is None:
        return None
    
    if data_dir is None:
        # Default data location
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "quickdraw" / "raw" / "strokes"
    else:
        data_dir = Path(data_dir)
    
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
    
    # CLIP normalization constants
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    
    # Render and encode each exemplar
    embeddings = []
    for line in sample_lines:
        sample = json.loads(line)
        drawing = sample["drawing"]
        
        # Convert to stroke params and render
        strokes = quickdraw_to_stroke_params(drawing, size=canvas_size, device=device)
        img_tensor = render_strokes_rgb(strokes, canvas_size=canvas_size, device=torch.device(device))
        
        # Normalize for CLIP
        img_normalized = (img_tensor - mean) / std
        
        # Encode with CLIP
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
        image_path: Path to the reference image (e.g., perfectFish.jpeg).
        clip_model: CLIP model for encoding.
        device: Device for tensors.
    
    Returns:
        CLIP embedding of the reference image [1, D].
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
) -> Optional[torch.Tensor]:
    """
    Load real photos from a directory and encode them with CLIP as exemplars.
    
    This lets you guide the sketch toward what REAL fish/cats/etc look like,
    using the rich semantic understanding CLIP learned from photos.
    
    Args:
        image_dir: Directory containing photos (jpg, png, jpeg).
        clip_model: CLIP model for encoding.
        device: Device for tensors.
        max_images: Maximum number of images to load.
    
    Returns:
        Tensor of CLIP embeddings [N, D], or None if no images found.
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


