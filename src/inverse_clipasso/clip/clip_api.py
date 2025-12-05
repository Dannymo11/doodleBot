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
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda", use_float32: bool = False):
        """
        Initialize CLIP model.
        
        Args:
            model_name: CLIP model variant (e.g., "ViT-B/32", "ViT-L/14").
            device: Device to load model on.
            use_float32: If True, converts model to float32 for gradient-based optimization.
                        This is slower but necessary for backprop through the model.
        """
        self.model_name = model_name
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.model.eval()

        # CLIP uses float16 by default on GPU, which can break gradient flow.
        # Convert to float32 if we need gradients for optimization.
        if use_float32:
            self.model = self.model.float()
            self.dtype = torch.float32
        else:
            # Keep original dtype (float16 on GPU, float32 on CPU)
            self.dtype = torch.float16 if device != "cpu" else torch.float32

    def encode_image(self, image: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """
        Encode images into CLIP latent space.
        
        Args:
            image: Image tensor [B, 3, H, W], should be CLIP-normalized.
            requires_grad: If True, allows gradients to flow through for optimization.
                          If False (default), uses no_grad for efficiency.
        
        Returns:
            Normalized image embeddings [B, D] in float32.
        """
        # Convert to model's expected dtype while preserving gradient info
        img_input = image.to(device=self.device, dtype=self.dtype)
        
        if requires_grad:
            # Allow gradients for optimization
            emb = self.model.encode_image(img_input)
        else:
            # No gradients for inference (faster, less memory)
        with torch.no_grad():
                emb = self.model.encode_image(img_input)
        
        # Always return float32 for consistent downstream operations
        emb = emb.float()
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text prompts into CLIP latent space.
        
        Text embeddings don't need gradients since we optimize images, not text.
        """
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        # Return float32 for consistency
        emb = emb.float()
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_text_multi(self, prompts: list[str]) -> torch.Tensor:
        """
        Encode multiple text prompts and average their embeddings.
        
        This gives a more robust target for optimization by averaging
        different phrasings of the same concept. For example:
        ["a cat", "a drawing of a cat", "a sketch of a cat"]
        
        Args:
            prompts: List of text prompts to encode and average.
        
        Returns:
            Averaged and normalized text embedding [1, D].
        """
        if not prompts:
            raise ValueError("Must provide at least one prompt")
        
        # Encode all prompts at once for efficiency
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)  # [N, D]
        
        # Average embeddings
        embeddings = embeddings.float()
        avg_emb = embeddings.mean(dim=0, keepdim=True)  # [1, D]
        
        # Normalize the averaged embedding
        return avg_emb / avg_emb.norm(dim=-1, keepdim=True)

def compute_similarity(image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity helper.
    """

    return torch.nn.functional.cosine_similarity(image_embed, text_embed)
