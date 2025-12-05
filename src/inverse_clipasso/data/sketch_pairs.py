"""
Dataset for paired sketch training.

Creates training pairs from QuickDraw data using augmentation strategies:
1. Noise injection (rough → original)
2. Stroke simplification (simplified → original)
3. Different renderings of same concept

The goal is to train a model that can "enhance" rough sketches.
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Callable
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as T


def render_strokes_pil(
    strokes: List[List[List[float]]],
    canvas_size: int = 224,
    stroke_width: int = 2,
    padding: int = 10,
) -> Image.Image:
    """
    Render QuickDraw strokes to a PIL Image.
    
    Args:
        strokes: QuickDraw format [[xs, ys], ...]
        canvas_size: Output image size
        stroke_width: Line width
        padding: Canvas padding
    
    Returns:
        Grayscale PIL Image
    """
    # Get bounds
    all_x, all_y = [], []
    for stroke in strokes:
        all_x.extend(stroke[0])
        all_y.extend(stroke[1])
    
    if not all_x:
        return Image.new("L", (canvas_size, canvas_size), 255)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    w = max_x - min_x + 1e-6
    h = max_y - min_y + 1e-6
    
    scale = (canvas_size - 2 * padding) / max(w, h)
    ox = (canvas_size - scale * w) / 2.0
    oy = (canvas_size - scale * h) / 2.0
    
    # Create image
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    
    for stroke in strokes:
        xs = [(x - min_x) * scale + ox for x in stroke[0]]
        ys = [(y - min_y) * scale + oy for y in stroke[1]]
        
        if len(xs) >= 2:
            points = list(zip(xs, ys))
            draw.line(points, fill=0, width=stroke_width)
    
    return img


def add_noise_to_strokes(
    strokes: List[List[List[float]]],
    noise_level: float = 5.0,
) -> List[List[List[float]]]:
    """Add random noise to stroke points."""
    noisy_strokes = []
    for stroke in strokes:
        xs = [x + random.gauss(0, noise_level) for x in stroke[0]]
        ys = [y + random.gauss(0, noise_level) for y in stroke[1]]
        noisy_strokes.append([xs, ys])
    return noisy_strokes


def simplify_strokes(
    strokes: List[List[List[float]]],
    keep_ratio: float = 0.5,
) -> List[List[List[float]]]:
    """Simplify strokes by keeping only a fraction of points."""
    simplified = []
    for stroke in strokes:
        n_points = len(stroke[0])
        n_keep = max(2, int(n_points * keep_ratio))
        
        if n_points <= n_keep:
            simplified.append(stroke)
        else:
            indices = np.linspace(0, n_points - 1, n_keep).astype(int)
            xs = [stroke[0][i] for i in indices]
            ys = [stroke[1][i] for i in indices]
            simplified.append([xs, ys])
    
    return simplified


def jitter_strokes(
    strokes: List[List[List[float]]],
    jitter_amount: float = 3.0,
) -> List[List[List[float]]]:
    """Add consistent jitter to each stroke."""
    jittered = []
    for stroke in strokes:
        dx = random.gauss(0, jitter_amount)
        dy = random.gauss(0, jitter_amount)
        xs = [x + dx for x in stroke[0]]
        ys = [y + dy for y in stroke[1]]
        jittered.append([xs, ys])
    return jittered


class SketchPairDataset(Dataset):
    """
    Dataset that creates pairs of (rough_sketch, clean_sketch).
    
    The "rough" sketch is created by applying degradations to the original.
    The model learns to reverse these degradations.
    """
    
    def __init__(
        self,
        data_dir: str,
        categories: List[str],
        samples_per_category: int = 1000,
        canvas_size: int = 224,
        augment_target: bool = True,
        degradation_types: List[str] = ["noise", "simplify", "jitter"],
    ):
        """
        Args:
            data_dir: Path to QuickDraw ndjson files
            categories: List of categories to load
            samples_per_category: Max samples per category
            canvas_size: Output image size
            augment_target: Apply mild augmentation to targets too
            degradation_types: Types of degradation to apply
        """
        self.canvas_size = canvas_size
        self.augment_target = augment_target
        self.degradation_types = degradation_types
        
        # Load all samples
        self.samples = []
        data_path = Path(data_dir)
        
        for category in categories:
            file_path = data_path / f"{category}.ndjson"
            if not file_path.exists():
                print(f"Warning: {file_path} not found")
                continue
            
            with open(file_path) as f:
                lines = f.readlines()
            
            # Sample
            if len(lines) > samples_per_category:
                lines = random.sample(lines, samples_per_category)
            
            for line in lines:
                sample = json.loads(line)
                self.samples.append({
                    "drawing": sample["drawing"],
                    "category": category,
                })
        
        print(f"Loaded {len(self.samples)} samples from {len(categories)} categories")
        
        # Transforms
        self.to_tensor = T.ToTensor()
    
    def __len__(self):
        return len(self.samples)
    
    def _apply_degradation(self, strokes):
        """Apply random degradation to create rough sketch."""
        deg_type = random.choice(self.degradation_types)
        
        if deg_type == "noise":
            return add_noise_to_strokes(strokes, noise_level=random.uniform(3, 10))
        elif deg_type == "simplify":
            return simplify_strokes(strokes, keep_ratio=random.uniform(0.3, 0.7))
        elif deg_type == "jitter":
            return jitter_strokes(strokes, jitter_amount=random.uniform(2, 8))
        else:
            return strokes
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (input_sketch, target_sketch) both as [1, H, W] tensors
        """
        sample = self.samples[idx]
        strokes = sample["drawing"]
        
        # Create target (clean) sketch
        target_img = render_strokes_pil(
            strokes,
            canvas_size=self.canvas_size,
            stroke_width=2,
        )
        
        # Create input (degraded) sketch
        degraded_strokes = self._apply_degradation(strokes)
        input_img = render_strokes_pil(
            degraded_strokes,
            canvas_size=self.canvas_size,
            stroke_width=random.randint(1, 3),  # Vary stroke width too
        )
        
        # Optional: add blur to input
        if random.random() < 0.3:
            input_img = input_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Convert to tensors and normalize to [-1, 1]
        input_tensor = self.to_tensor(input_img) * 2 - 1  # [1, H, W]
        target_tensor = self.to_tensor(target_img) * 2 - 1
        
        return input_tensor, target_tensor


class RealPairDataset(Dataset):
    """
    Dataset for real paired sketches.
    
    Expects a directory structure like:
    data/pairs/
        input/
            001.png
            002.png
            ...
        target/
            001.png
            002.png
            ...
    """
    
    def __init__(
        self,
        data_dir: str,
        canvas_size: int = 224,
    ):
        self.data_dir = Path(data_dir)
        self.canvas_size = canvas_size
        
        input_dir = self.data_dir / "input"
        target_dir = self.data_dir / "target"
        
        # Find paired files
        self.pairs = []
        for input_file in sorted(input_dir.glob("*")):
            target_file = target_dir / input_file.name
            if target_file.exists():
                self.pairs.append((input_file, target_file))
        
        print(f"Found {len(self.pairs)} paired sketches")
        
        self.transform = T.Compose([
            T.Resize((canvas_size, canvas_size)),
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_path, target_path = self.pairs[idx]
        
        input_img = Image.open(input_path).convert("L")
        target_img = Image.open(target_path).convert("L")
        
        input_tensor = self.transform(input_img) * 2 - 1
        target_tensor = self.transform(target_img) * 2 - 1
        
        return input_tensor, target_tensor


def create_dataloader(
    data_dir: str,
    categories: List[str],
    batch_size: int = 16,
    num_workers: int = 4,
    samples_per_category: int = 1000,
) -> DataLoader:
    """
    Create a DataLoader for sketch pair training.
    
    Args:
        data_dir: Path to QuickDraw data
        categories: List of categories
        batch_size: Batch size
        num_workers: Number of data loading workers
        samples_per_category: Samples per category
    
    Returns:
        DataLoader
    """
    dataset = SketchPairDataset(
        data_dir=data_dir,
        categories=categories,
        samples_per_category=samples_per_category,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

