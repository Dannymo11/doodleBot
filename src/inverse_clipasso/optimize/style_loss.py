"""
VGG-based style loss for neural style transfer.

Uses VGG19 intermediate features and Gram matrices to capture texture/style,
following the classic approach from "A Neural Algorithm of Artistic Style" (Gatys et al., 2015).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights


__all__ = [
    "VGGStyleLoss",
    "VGGFeatureExtractor",
]


# VGG normalization (ImageNet stats)
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
VGG_STD = torch.tensor([0.229, 0.224, 0.225])


class VGGFeatureExtractor(nn.Module):
    """
    Extract intermediate features from VGG19 for style transfer.
    
    Uses layers that capture different scales of texture information:
    - conv1_1: Fine textures, edges
    - conv2_1: Small patterns
    - conv3_1: Medium patterns
    - conv4_1: Larger structures
    - conv5_1: High-level patterns
    """
    
    # Default layers for style extraction (after each pooling)
    DEFAULT_STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    
    # Mapping from layer names to VGG19 indices
    LAYER_MAP = {
        'conv1_1': 0,   # First conv
        'conv1_2': 2,
        'conv2_1': 5,   # After first pool
        'conv2_2': 7,
        'conv3_1': 10,  # After second pool
        'conv3_2': 12,
        'conv3_3': 14,
        'conv3_4': 16,
        'conv4_1': 19,  # After third pool
        'conv4_2': 21,
        'conv4_3': 23,
        'conv4_4': 25,
        'conv5_1': 28,  # After fourth pool
        'conv5_2': 30,
        'conv5_3': 32,
        'conv5_4': 34,
    }
    
    def __init__(
        self,
        layers: Optional[List[str]] = None,
        device: str = 'cpu',
    ):
        """
        Args:
            layers: Which VGG layers to extract. Defaults to style layers.
            device: Device to load VGG on.
        """
        super().__init__()
        
        self.layers = layers or self.DEFAULT_STYLE_LAYERS
        self.device = device
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        
        # Freeze all parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Get the max layer index we need
        max_idx = max(self.LAYER_MAP[layer] for layer in self.layers) + 1
        
        # Only keep layers up to the max we need
        self.vgg = nn.Sequential(*list(vgg.children())[:max_idx + 1])
        self.vgg.to(device)
        
        # Setup normalization
        self._mean = VGG_MEAN.view(1, 3, 1, 1).to(device)
        self._std = VGG_STD.view(1, 3, 1, 1).to(device)
    
    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Apply VGG normalization (ImageNet stats)."""
        return (image - self._mean) / self._std
    
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified VGG layers.
        
        Args:
            image: Input image [B, 3, H, W] in range [0, 1].
        
        Returns:
            Dictionary mapping layer name to feature tensor.
        """
        # Normalize for VGG
        x = self._normalize(image)
        
        features = {}
        layer_indices = {self.LAYER_MAP[name]: name for name in self.layers}
        
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx in layer_indices:
                features[layer_indices[idx]] = x
        
        return features


class VGGStyleLoss(nn.Module):
    """
    Style loss using VGG19 features and Gram matrices.
    
    This is the classic neural style transfer approach:
    - Extract features from multiple VGG layers
    - Compute Gram matrices (correlations between feature channels)
    - Match Gram matrices between rendered image and style reference
    
    Gram matrices capture texture/style independent of spatial arrangement.
    """
    
    def __init__(
        self,
        layers: Optional[List[str]] = None,
        layer_weights: Optional[Dict[str, float]] = None,
        device: str = 'cpu',
    ):
        """
        Args:
            layers: VGG layers to use for style matching.
            layer_weights: Weight for each layer's contribution.
            device: Device for VGG model.
        """
        super().__init__()
        
        self.feature_extractor = VGGFeatureExtractor(layers=layers, device=device)
        self.layers = self.feature_extractor.layers
        
        # Default: equal weights for all layers
        if layer_weights is None:
            self.layer_weights = {layer: 1.0 for layer in self.layers}
        else:
            self.layer_weights = layer_weights
        
        self._style_grams: Optional[Dict[str, torch.Tensor]] = None
    
    def _gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix from feature maps.
        
        Args:
            features: Feature tensor [B, C, H, W].
        
        Returns:
            Gram matrix [B, C, C].
        """
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)  # [B, C, H*W]
        
        # Gram matrix: G = F @ F^T
        gram = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
        
        # Normalize by number of elements
        return gram / (c * h * w)
    
    def set_style_target(self, style_image: torch.Tensor) -> None:
        """
        Pre-compute Gram matrices for the style reference image.
        
        Args:
            style_image: Style reference [1, 3, H, W] in range [0, 1].
        """
        with torch.no_grad():
            features = self.feature_extractor(style_image)
            self._style_grams = {
                name: self._gram_matrix(feat) 
                for name, feat in features.items()
            }
    
    def forward(self, rendered_image: torch.Tensor) -> torch.Tensor:
        """
        Compute style loss between rendered image and style target.
        
        Args:
            rendered_image: Rendered sketch [B, 3, H, W] in range [0, 1].
        
        Returns:
            Weighted sum of per-layer style losses.
        """
        if self._style_grams is None:
            raise ValueError("Must call set_style_target() before computing loss")
        
        # Extract features and Gram matrices from rendered image
        features = self.feature_extractor(rendered_image)
        
        # Compute loss for each layer
        total_loss = torch.tensor(0.0, device=rendered_image.device)
        
        for layer_name, weight in self.layer_weights.items():
            if layer_name in features and layer_name in self._style_grams:
                rendered_gram = self._gram_matrix(features[layer_name])
                style_gram = self._style_grams[layer_name].to(rendered_gram.device)
                
                # MSE between Gram matrices
                layer_loss = F.mse_loss(rendered_gram, style_gram)
                total_loss = total_loss + weight * layer_loss
        
        return total_loss
    
    def extract_style_grams(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract Gram matrices from an image (for external use).
        
        Args:
            image: Input image [B, 3, H, W] in range [0, 1].
        
        Returns:
            Dictionary mapping layer name to Gram matrix.
        """
        with torch.no_grad():
            features = self.feature_extractor(image)
            return {
                name: self._gram_matrix(feat) 
                for name, feat in features.items()
            }

