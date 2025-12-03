"""
Loss functions for the inverse CLIPasso optimization.
"""

from __future__ import annotations

from typing import Dict

import torch


def semantic_loss(image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
    """
    Encourage alignment between sketch and label semantics.
    """

    return 1.0 - torch.nn.functional.cosine_similarity(image_embed, text_embed)


def style_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviations from a reference style image.
    """

    raise NotImplementedError("Implement style loss (e.g., Gram matrices).")


def stroke_regularizer(strokes: torch.Tensor) -> torch.Tensor:
    """
    Encourage smooth, simple strokes.
    """

    raise NotImplementedError("Implement curvature / length regularizer.")


def total_variation(image: torch.Tensor) -> torch.Tensor:
    """
    Total variation over rendered sketch.
    """

    x_diff = image[:, :, 1:] - image[:, :, :-1]
    y_diff = image[:, 1:, :] - image[:, :-1, :]
    return torch.mean(torch.abs(x_diff)) + torch.mean(torch.abs(y_diff))


def aggregate_losses(loss_terms: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    """
    Combine weighted loss terms.
    """

    total = torch.tensor(0.0, device=next(iter(loss_terms.values())).device)
    for name, value in loss_terms.items():
        total = total + weights.get(name, 1.0) * value
    return total
