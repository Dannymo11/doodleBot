"""
Multi-scale perceptual loss using CLIP intermediate features.

CLIPasso uses not just the final CLIP embedding, but also features from
intermediate layers. This provides:
- Multi-scale semantic guidance (coarse to fine)
- Richer gradient signal
- Better preservation of structural details

This is a KEY component that differentiates CLIPasso from simple CLIP loss!
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPFeatureExtractor(nn.Module):
    """
    Extract intermediate features from CLIP's visual encoder.
    
    Works with both ResNet and ViT-based CLIP models.
    """
    
    def __init__(self, clip_model, layer_indices: Optional[List[int]] = None):
        """
        Args:
            clip_model: CLIP model (from clip.load()).
            layer_indices: Which layers to extract features from.
                          If None, uses default layers based on architecture.
        """
        super().__init__()
        self.clip_visual = clip_model.visual
        self.layer_indices = layer_indices
        
        # Detect architecture
        self.is_vit = hasattr(self.clip_visual, 'transformer')
        
        if self.is_vit:
            # ViT architecture
            self.num_layers = len(self.clip_visual.transformer.resblocks)
            if layer_indices is None:
                # Use layers at 25%, 50%, 75%, 100%
                self.layer_indices = [
                    self.num_layers // 4,
                    self.num_layers // 2,
                    3 * self.num_layers // 4,
                    self.num_layers - 1,
                ]
        else:
            # ResNet architecture
            if layer_indices is None:
                self.layer_indices = [0, 1, 2, 3]  # layer1 through layer4
        
        # Storage for intermediate features
        self._features: Dict[int, torch.Tensor] = {}
        self._hooks: List = []
    
    def _create_hooks(self):
        """Create forward hooks to capture intermediate features."""
        self._clear_hooks()
        self._features = {}
        
        if self.is_vit:
            # Hook into transformer blocks
            for idx in self.layer_indices:
                block = self.clip_visual.transformer.resblocks[idx]
                hook = block.register_forward_hook(self._get_hook(idx))
                self._hooks.append(hook)
        else:
            # Hook into ResNet layers
            layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
            for idx in self.layer_indices:
                if idx < len(layer_names):
                    layer = getattr(self.clip_visual, layer_names[idx])
                    hook = layer.register_forward_hook(self._get_hook(idx))
                    self._hooks.append(hook)
    
    def _get_hook(self, idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            self._features[idx] = output
        return hook
    
    def _clear_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def forward(self, image: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract intermediate features from CLIP visual encoder.
        
        Args:
            image: Input image tensor [B, 3, H, W], should be CLIP-normalized.
        
        Returns:
            Dictionary mapping layer index to feature tensor.
        """
        self._create_hooks()
        
        try:
            # Run forward pass (features captured by hooks)
            if self.is_vit:
                _ = self.clip_visual(image)
            else:
                _ = self.clip_visual(image)
            
            # Copy features (hooks store references that may be overwritten)
            features = {k: v.clone() for k, v in self._features.items()}
        finally:
            self._clear_hooks()
        
        return features


class MultiScalePerceptualLoss(nn.Module):
    """
    Compute perceptual loss using multiple CLIP layers.
    
    Loss = Σ_l w_l * ||F_l(rendered) - F_l(target)||²
    
    Where F_l is the feature map at layer l.
    """
    
    def __init__(
        self,
        clip_model,
        layer_weights: Optional[Dict[int, float]] = None,
        normalize_features: bool = True,
    ):
        """
        Args:
            clip_model: CLIP model for feature extraction.
            layer_weights: Weight for each layer's loss. If None, uses equal weights.
            normalize_features: If True, L2-normalize features before comparison.
        """
        super().__init__()
        self.feature_extractor = CLIPFeatureExtractor(clip_model)
        self.normalize_features = normalize_features
        
        if layer_weights is None:
            # Equal weights for all layers
            self.layer_weights = {
                idx: 1.0 for idx in self.feature_extractor.layer_indices
            }
        else:
            self.layer_weights = layer_weights
    
    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """L2 normalize features."""
        return F.normalize(features.flatten(start_dim=1), dim=-1)
    
    def extract_features(self, image: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract and optionally normalize features."""
        features = self.feature_extractor(image)
        
        if self.normalize_features:
            features = {k: self._normalize(v) for k, v in features.items()}
        
        return features
    
    def forward(
        self,
        rendered_image: torch.Tensor,
        target_features: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute multi-scale perceptual loss.
        
        Args:
            rendered_image: Rendered sketch [B, 3, H, W], CLIP-normalized.
            target_features: Pre-computed target features (from extract_features).
        
        Returns:
            Weighted sum of per-layer L2 losses.
        """
        # Extract features from rendered image
        rendered_features = self.extract_features(rendered_image)
        
        # Compute loss for each layer
        total_loss = torch.tensor(0.0, device=rendered_image.device)
        
        for layer_idx, weight in self.layer_weights.items():
            if layer_idx in rendered_features and layer_idx in target_features:
                rendered_feat = rendered_features[layer_idx]
                target_feat = target_features[layer_idx].to(rendered_feat.device)
                
                # L2 loss (or cosine distance)
                layer_loss = F.mse_loss(rendered_feat, target_feat)
                total_loss = total_loss + weight * layer_loss
        
        return total_loss


class CLIPassoLoss(nn.Module):
    """
    Complete CLIPasso-style loss combining multiple components.
    
    Total Loss = w_semantic * L_semantic + w_perceptual * L_perceptual + w_geometric * L_geometric
    
    Components:
    - L_semantic: CLIP embedding similarity (final layer)
    - L_perceptual: Multi-scale feature matching (intermediate layers)
    - L_geometric: Stroke regularization, smoothness, etc.
    """
    
    def __init__(
        self,
        clip_model,
        semantic_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        augmenter=None,
    ):
        """
        Args:
            clip_model: CLIP model for encoding and feature extraction.
            semantic_weight: Weight for final embedding similarity.
            perceptual_weight: Weight for intermediate layer matching.
            augmenter: Optional augmentation module.
        """
        super().__init__()
        self.clip_model = clip_model
        self.perceptual_loss = MultiScalePerceptualLoss(clip_model.model)
        self.semantic_weight = semantic_weight
        self.perceptual_weight = perceptual_weight
        self.augmenter = augmenter
        
        # Cache for target features
        self._target_embedding: Optional[torch.Tensor] = None
        self._target_features: Optional[Dict[int, torch.Tensor]] = None
    
    def set_target(
        self,
        target: torch.Tensor,
        is_text: bool = False,
    ):
        """
        Set the optimization target.
        
        Args:
            target: Either text tokens or target image tensor.
            is_text: If True, target is text; if False, target is image.
        """
        with torch.no_grad():
            if is_text:
                self._target_embedding = self.clip_model.encode_text(target)
                self._target_features = None  # No perceptual loss for text targets
            else:
                # For image targets, extract both embedding and intermediate features
                normalized = self._normalize_for_clip(target)
                self._target_embedding = self.clip_model.encode_image(normalized)
                self._target_features = self.perceptual_loss.extract_features(normalized)
    
    def _normalize_for_clip(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image for CLIP (ImageNet normalization)."""
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device)
        return (image - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    
    def forward(
        self,
        rendered_image: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute CLIPasso-style loss.
        
        Args:
            rendered_image: Rendered sketch [1, 3, H, W].
            return_components: If True, return dict of individual losses.
        
        Returns:
            Total loss (or dict if return_components=True).
        """
        if self._target_embedding is None:
            raise ValueError("Must call set_target() before computing loss")
        
        # Normalize for CLIP
        normalized = self._normalize_for_clip(rendered_image)
        
        # Apply augmentations if available
        if self.augmenter is not None:
            augmented = self.augmenter(normalized)
        else:
            augmented = normalized
        
        # Semantic loss (CLIP embedding similarity)
        embeddings = self.clip_model.encode_image(augmented, requires_grad=True)
        # Average over augmentations
        if embeddings.shape[0] > 1:
            embedding = embeddings.mean(dim=0, keepdim=True)
        else:
            embedding = embeddings
        
        semantic_loss = 1.0 - F.cosine_similarity(
            embedding, self._target_embedding
        ).mean()
        
        # Perceptual loss (intermediate features)
        if self._target_features is not None and self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(normalized, self._target_features)
        else:
            perceptual_loss = torch.tensor(0.0, device=rendered_image.device)
        
        # Total loss
        total_loss = (
            self.semantic_weight * semantic_loss +
            self.perceptual_weight * perceptual_loss
        )
        
        if return_components:
            return {
                'total': total_loss,
                'semantic': semantic_loss,
                'perceptual': perceptual_loss,
            }
        
        return total_loss


def compute_clip_loss_with_augmentations(
    rendered_image: torch.Tensor,
    target_embedding: torch.Tensor,
    clip_model,
    augmenter,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute CLIP loss with augmentations (simplified interface).
    
    Args:
        rendered_image: Rendered sketch [1, 3, H, W].
        target_embedding: Target CLIP embedding [1, D].
        clip_model: ClipModel instance.
        augmenter: Augmentation module.
        normalize: Whether to apply CLIP normalization.
    
    Returns:
        Average (1 - similarity) over augmented versions.
    """
    # Normalize if needed
    if normalize:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=rendered_image.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=rendered_image.device)
        normalized = (rendered_image - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    else:
        normalized = rendered_image
    
    # Apply augmentations
    augmented = augmenter(normalized)
    
    # Encode all augmented versions
    embeddings = clip_model.encode_image(augmented, requires_grad=True)
    
    # Compute similarity to target
    similarities = F.cosine_similarity(embeddings, target_embedding)
    
    # Return average loss
    return 1.0 - similarities.mean()

