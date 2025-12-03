"""
Evaluation metrics for inverse CLIPasso outputs.
"""

from __future__ import annotations

from typing import Tuple

import torch


def clip_cosine(image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between CLIP embeddings.
    """

    return torch.nn.functional.cosine_similarity(image_embed, text_embed)


def ssim_gradient(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Structural similarity of gradient fields as a proxy for edges.
    """

    raise NotImplementedError("Implement SSIM on Sobel gradients.")


def stroke_distance(pred_strokes: torch.Tensor, gt_strokes: torch.Tensor) -> torch.Tensor:
    """
    Distance between predicted and ground-truth stroke sequences.
    """

    raise NotImplementedError("Implement stroke distance metric.")
