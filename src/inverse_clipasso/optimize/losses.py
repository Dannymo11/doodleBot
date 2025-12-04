"""
Loss functions for the inverse CLIPasso optimization.
"""

from __future__ import annotations

from typing import Dict, List, Union

import torch


def semantic_loss(image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
    """
    Encourage alignment between sketch and label semantics.
    
    Args:
        image_embed: Image embedding [B, D] from CLIP.
        text_embed: Text embedding [1, D] or [B, D] from CLIP.
    
    Returns:
        Loss tensor (1 - cosine_similarity), higher = less aligned.
    """
    return 1.0 - torch.nn.functional.cosine_similarity(image_embed, text_embed)


def style_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviations from a reference style image using Gram matrices.
    
    Args:
        prediction: Feature maps from predicted image [B, C, H, W].
        target: Feature maps from target style image [B, C, H, W].
    
    Returns:
        Style loss (MSE between Gram matrices).
    """
    def gram_matrix(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    return torch.nn.functional.mse_loss(gram_matrix(prediction), gram_matrix(target))


def stroke_regularizer(strokes: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """
    Encourage smooth, simple strokes with multiple regularization terms.
    
    Combines:
    - Curvature penalty: penalizes sharp turns between consecutive segments
    - Length penalty: penalizes overly long strokes
    - Spacing penalty: encourages even spacing between control points
    
    Args:
        strokes: Either a single tensor [N, 2] or list of tensors [N_i, 2].
    
    Returns:
        Combined regularization loss (lower = smoother, simpler strokes).
    """
    if isinstance(strokes, torch.Tensor):
        strokes = [strokes]
    
    total_loss = torch.tensor(0.0, device=strokes[0].device)
    n_strokes = 0
    
    for stroke in strokes:
        if stroke.shape[0] < 2:
            continue
        
        n_strokes += 1
        
        # Segment vectors: direction from point i to point i+1
        segments = stroke[1:] - stroke[:-1]  # [N-1, 2]
        
        # 1. LENGTH PENALTY
        # Penalize total stroke length (encourages compact drawings)
        segment_lengths = torch.norm(segments, dim=1)  # [N-1]
        length_loss = segment_lengths.mean()
        
        # 2. CURVATURE PENALTY  
        # Penalize angle changes between consecutive segments
        if stroke.shape[0] >= 3:
            # Compute angle between consecutive segments using dot product
            seg1 = segments[:-1]  # [N-2, 2]
            seg2 = segments[1:]   # [N-2, 2]
            
            # Normalize segments
            seg1_norm = seg1 / (torch.norm(seg1, dim=1, keepdim=True) + 1e-8)
            seg2_norm = seg2 / (torch.norm(seg2, dim=1, keepdim=True) + 1e-8)
            
            # Dot product gives cos(angle), we want to penalize when it's low (sharp turns)
            # cos(0) = 1 (straight), cos(180) = -1 (U-turn)
            cos_angles = (seg1_norm * seg2_norm).sum(dim=1)  # [N-2]
            
            # Loss: 1 - cos(angle) is 0 for straight, 2 for U-turn
            curvature_loss = (1 - cos_angles).mean()
        else:
            curvature_loss = torch.tensor(0.0, device=stroke.device)
        
        # 3. SPACING REGULARITY
        # Penalize variance in segment lengths (encourages even spacing)
        if len(segment_lengths) > 1:
            spacing_loss = segment_lengths.var()
        else:
            spacing_loss = torch.tensor(0.0, device=stroke.device)
        
        # Combine with weights
        total_loss = total_loss + length_loss * 0.1 + curvature_loss * 1.0 + spacing_loss * 0.5
    
    # Average over strokes
    if n_strokes > 0:
        total_loss = total_loss / n_strokes
    
    return total_loss


def total_variation(image: torch.Tensor) -> torch.Tensor:
    """
    Total variation over rendered sketch.
    """

    x_diff = image[:, :, 1:] - image[:, :, :-1]
    y_diff = image[:, 1:, :] - image[:, :-1, :]
    return torch.mean(torch.abs(x_diff)) + torch.mean(torch.abs(y_diff))


def original_strokes_loss(
    current_strokes: List[torch.Tensor],
    original_strokes: List[torch.Tensor],
) -> torch.Tensor:
    """
    Penalize deviation from original stroke positions.
    
    This is crucial for REFINEMENT mode where we want to improve
    a sketch's CLIP alignment while preserving its original character.
    Without this, the optimizer will destroy the sketch trying to match
    a generic text concept.
    
    Args:
        current_strokes: Current stroke positions (being optimized).
        original_strokes: Original stroke positions (fixed reference).
    
    Returns:
        Mean squared deviation from original positions.
    """
    if len(current_strokes) != len(original_strokes):
        raise ValueError(
            f"Stroke count mismatch: {len(current_strokes)} vs {len(original_strokes)}"
        )
    
    total_loss = torch.tensor(0.0, device=current_strokes[0].device)
    n_points = 0
    
    for curr, orig in zip(current_strokes, original_strokes):
        if curr.shape != orig.shape:
            # Skip mismatched strokes
            continue
        
        # MSE between current and original positions
        diff = curr - orig.to(curr.device)
        total_loss = total_loss + (diff ** 2).sum()
        n_points += curr.numel()
    
    if n_points > 0:
        total_loss = total_loss / n_points
    
    return total_loss


def exemplar_loss(
    current_embedding: torch.Tensor,
    exemplar_embeddings: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Guide the sketch toward "good examples" from a reference set (e.g., QuickDraw).
    
    This is the KEY for refinement: instead of optimizing toward a generic
    text concept like "a fish", we optimize toward what actual good fish
    sketches look like in CLIP space.
    
    Args:
        current_embedding: CLIP embedding of current sketch [1, D] or [B, D].
        exemplar_embeddings: CLIP embeddings of reference sketches [N, D].
        mode: How to aggregate:
            - "mean": Pull toward centroid of exemplars (stable, less specific)
            - "min": Pull toward nearest exemplar (more specific, can be noisy)
            - "softmin": Soft minimum (smooth approximation of min)
    
    Returns:
        Loss value (lower = more similar to exemplars).
    """
    # Compute cosine similarity to all exemplars
    # current_embedding: [B, D], exemplar_embeddings: [N, D]
    # Result: [B, N] similarities
    similarities = torch.nn.functional.cosine_similarity(
        current_embedding.unsqueeze(1),  # [B, 1, D]
        exemplar_embeddings.unsqueeze(0),  # [1, N, D]
        dim=2
    )  # [B, N]
    
    if mode == "mean":
        # Pull toward the centroid (average similarity)
        return 1.0 - similarities.mean(dim=1)
    
    elif mode == "min":
        # Pull toward the nearest exemplar
        return 1.0 - similarities.max(dim=1).values
    
    elif mode == "softmin":
        # Soft maximum (like softmin of distances)
        # Higher temperature = more like mean, lower = more like max
        temperature = 0.1
        weights = torch.softmax(similarities / temperature, dim=1)
        weighted_sim = (weights * similarities).sum(dim=1)
        return 1.0 - weighted_sim
    
    else:
        raise ValueError(f"Unknown exemplar loss mode: {mode}")


def aggregate_losses(loss_terms: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    """
    Combine weighted loss terms.
    """

    total = torch.tensor(0.0, device=next(iter(loss_terms.values())).device)
    for name, value in loss_terms.items():
        total = total + weights.get(name, 1.0) * value
    return total
