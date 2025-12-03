"""
Helpers for encoding sketches and text prompts with CLIP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import clip # open ai clip model


@dataclass
class ClipOutputs:
    image_embeds: torch.Tensor
    text_embeds: torch.Tensor


class ClipModel:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.model.eval()

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode images into CLIP latent space.
        """
        with torch.no_grad():
            emb = self.model.encode_image(image.to(self.device))
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text prompts into CLIP latent space.
        """
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        return emb / emb.norm(dim=-1, keepdim=True)

def compute_similarity(image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity helper.
    """

    return torch.nn.functional.cosine_similarity(image_embed, text_embed)
