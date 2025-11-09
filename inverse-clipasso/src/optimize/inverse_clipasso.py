"""
Optimization driver that refines sketches toward semantic targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch

from inverse_clipasso.src.clip.clip_api import ClipModel
from inverse_clipasso.src.render.diffvg_wrapper import render_svg
from inverse_clipasso.src.optimize import losses


@dataclass
class OptimizationConfig:
    label: str
    steps: int = 500
    lr: float = 1e-2
    canvas_size: int = 224
    checkpoint_dir: Path | None = None
    loss_weights: Dict[str, float] | None = None


class InverseClipassoOptimizer:
    def __init__(self, clip_model: ClipModel, config: OptimizationConfig):
        self.clip = clip_model
        self.config = config
        self.loss_weights = config.loss_weights or {
            "semantic": 1.0,
            "stroke": 0.1,
            "tv": 0.01,
        }

    def optimize(self, init_strokes: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        Run the main optimization loop; yields intermediate sketches.
        """

        raise NotImplementedError("Implement inverse CLIPasso optimizer.")
