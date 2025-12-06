"""
Augmentation module for CLIPasso-style optimization.

Applying random augmentations during CLIP encoding makes the optimization:
- More robust to local minima
- Better at capturing semantic content vs. pixel-level details
- More generalizable across different viewpoints

This is a KEY component of CLIPasso's success!
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import math

__all__ = [
    "CLIPAugmentations",
    "DifferentiableAugmentations",
    "get_augmenter",
]


class CLIPAugmentations(nn.Module):
    """
    Differentiable augmentations for CLIP-guided optimization.
    
    During optimization, we apply random augmentations to the rendered sketch
    before encoding with CLIP. This encourages the optimizer to find strokes
    that are semantically meaningful regardless of small geometric variations.
    
    Augmentations include:
    - Random perspective transforms
    - Random crops and resizes
    - Random rotations
    - Color jitter (for robustness)
    """
    
    def __init__(
        self,
        num_augments: int = 4,
        crop_scale: Tuple[float, float] = (0.7, 1.0),
        perspective_scale: float = 0.2,
        rotation_degrees: float = 15.0,
        brightness: float = 0.1,
        contrast: float = 0.1,
        target_size: int = 224,
    ):
        """
        Args:
            num_augments: Number of augmented versions to generate.
            crop_scale: Range for random crop scale (min, max).
            perspective_scale: Maximum perspective distortion.
            rotation_degrees: Maximum rotation in degrees.
            brightness: Brightness jitter range.
            contrast: Contrast jitter range.
            target_size: Output size (224 for CLIP).
        """
        super().__init__()
        self.num_augments = num_augments
        self.crop_scale = crop_scale
        self.perspective_scale = perspective_scale
        self.rotation_degrees = rotation_degrees
        self.brightness = brightness
        self.contrast = contrast
        self.target_size = target_size
    
    def random_perspective(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random perspective transformation."""
        _, _, h, w = img.shape
        
        # Generate random perspective distortion
        half_height = h // 2
        half_width = w // 2
        
        distortion = self.perspective_scale
        
        # Source points (corners)
        src = torch.tensor([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=torch.float32)
        
        # Destination points (distorted corners)
        dst = src.clone()
        for i in range(4):
            dst[i, 0] += random.uniform(-distortion, distortion) * half_width
            dst[i, 1] += random.uniform(-distortion, distortion) * half_height
        
        # Apply perspective transform
        try:
            return TF.perspective(
                img, 
                startpoints=src.tolist(), 
                endpoints=dst.tolist(),
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=1.0,  # White background
            )
        except Exception:
            return img
    
    def random_crop_resize(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random crop and resize to target size."""
        _, _, h, w = img.shape
        
        # Random scale
        scale = random.uniform(*self.crop_scale)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Random position
        top = random.randint(0, max(0, h - new_h))
        left = random.randint(0, max(0, w - new_w))
        
        # Crop
        cropped = img[:, :, top:top + new_h, left:left + new_w]
        
        # Resize back to target
        return F.interpolate(
            cropped, 
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False,
        )
    
    def random_rotation(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random rotation."""
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        return TF.rotate(img, angle, fill=1.0)
    
    def random_color_jitter(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random brightness/contrast adjustments."""
        # Random brightness
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        img = img * brightness_factor
        
        # Random contrast
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = img.mean(dim=(-2, -1), keepdim=True)
        img = (img - mean) * contrast_factor + mean
        
        return torch.clamp(img, 0, 1)
    
    def augment_single(self, img: torch.Tensor) -> torch.Tensor:
        """Apply a random combination of augmentations to a single image."""
        # Randomly apply each augmentation
        if random.random() < 0.5:
            img = self.random_perspective(img)
        
        if random.random() < 0.7:
            img = self.random_crop_resize(img)
        
        if random.random() < 0.3:
            img = self.random_rotation(img)
        
        if random.random() < 0.3:
            img = self.random_color_jitter(img)
        
        # Ensure correct size
        if img.shape[-2:] != (self.target_size, self.target_size):
            img = F.interpolate(
                img,
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False,
            )
        
        return img
    
    def forward(
        self,
        img: torch.Tensor,
        include_original: bool = True,
    ) -> torch.Tensor:
        """
        Generate augmented versions of the input image.
        
        Args:
            img: Input image tensor [1, 3, H, W] or [B, 3, H, W].
            include_original: If True, include the original image in output.
        
        Returns:
            Batch of augmented images [N, 3, H, W].
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        batch_size = img.shape[0]
        augmented = []
        
        for b in range(batch_size):
            single_img = img[b:b+1]
            
            if include_original:
                # First image is always the original (resized if needed)
                if single_img.shape[-2:] != (self.target_size, self.target_size):
                    original = F.interpolate(
                        single_img,
                        size=(self.target_size, self.target_size),
                        mode='bilinear',
                        align_corners=False,
                    )
                else:
                    original = single_img
                augmented.append(original)
                n_aug = self.num_augments - 1
            else:
                n_aug = self.num_augments
            
            # Generate augmented versions
            for _ in range(n_aug):
                aug_img = self.augment_single(single_img.clone())
                augmented.append(aug_img)
        
        return torch.cat(augmented, dim=0)


class DifferentiableAugmentations(nn.Module):
    """
    Fully differentiable augmentations using grid sampling.
    
    These augmentations preserve gradients through the entire pipeline,
    which can help with optimization stability.
    """
    
    def __init__(
        self,
        num_augments: int = 4,
        scale_range: Tuple[float, float] = (0.8, 1.0),
        translate_range: float = 0.1,
        rotation_range: float = 0.1,  # In radians
    ):
        super().__init__()
        self.num_augments = num_augments
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.rotation_range = rotation_range
    
    def _get_affine_matrix(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random affine transformation matrices."""
        # Random scale
        scale = torch.empty(batch_size, device=device).uniform_(*self.scale_range)
        
        # Random translation
        tx = torch.empty(batch_size, device=device).uniform_(
            -self.translate_range, self.translate_range
        )
        ty = torch.empty(batch_size, device=device).uniform_(
            -self.translate_range, self.translate_range
        )
        
        # Random rotation
        angle = torch.empty(batch_size, device=device).uniform_(
            -self.rotation_range, self.rotation_range
        )
        
        # Build affine matrices [B, 2, 3]
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # Combine rotation and scale
        a11 = scale * cos_a
        a12 = -scale * sin_a
        a21 = scale * sin_a
        a22 = scale * cos_a
        
        # Create transformation matrix
        theta = torch.zeros(batch_size, 2, 3, device=device)
        theta[:, 0, 0] = a11
        theta[:, 0, 1] = a12
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = a21
        theta[:, 1, 1] = a22
        theta[:, 1, 2] = ty
        
        return theta
    
    def forward(
        self,
        img: torch.Tensor,
        include_original: bool = True,
    ) -> torch.Tensor:
        """
        Apply differentiable affine augmentations.
        
        Args:
            img: Input image [1, 3, H, W].
            include_original: Include unaugmented original.
        
        Returns:
            Augmented batch [N, 3, H, W].
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        device = img.device
        
        augmented = []
        if include_original:
            augmented.append(img)
            n_aug = self.num_augments - 1
        else:
            n_aug = self.num_augments
        
        if n_aug > 0:
            # Repeat image for batch processing
            img_batch = img.expand(n_aug, -1, -1, -1)
            
            # Get random affine matrices
            theta = self._get_affine_matrix(n_aug, device)
            
            # Apply affine transformation using grid sampling
            grid = F.affine_grid(theta, img_batch.size(), align_corners=False)
            aug_batch = F.grid_sample(
                img_batch, grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=False,
            )
            
            augmented.append(aug_batch)
        
        return torch.cat(augmented, dim=0)


def get_augmenter(
    augment_type: str = "standard",
    num_augments: int = 4,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get an augmentation module.
    
    Args:
        augment_type: "standard" (random augments) or "differentiable" (grid-based).
        num_augments: Number of augmented versions.
        **kwargs: Additional arguments for the augmenter.
    
    Returns:
        Augmentation module.
    """
    if augment_type == "standard":
        return CLIPAugmentations(num_augments=num_augments, **kwargs)
    elif augment_type == "differentiable":
        return DifferentiableAugmentations(num_augments=num_augments, **kwargs)
    else:
        raise ValueError(f"Unknown augment_type: {augment_type}")

