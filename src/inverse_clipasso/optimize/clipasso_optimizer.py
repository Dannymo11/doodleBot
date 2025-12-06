"""
CLIPasso-style optimizer with Bézier curves, augmentations, and perceptual loss.

This implements the full CLIPasso optimization pipeline:
1. Bézier curve representation for smooth, natural strokes
2. Random augmentations during CLIP encoding for robustness
3. Multi-scale perceptual loss using intermediate CLIP features
4. Separate learning rates for different parameter types
5. Progressive stroke addition with initialization options

Reference: "CLIPasso: Semantically-Aware Object Sketching" (Vinker et al., 2022)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.inverse_clipasso.clip.clip_api import ClipModel
from src.inverse_clipasso.render.bezier import (
    BezierSketch,
    BezierStroke,
    render_bezier_strokes,
    initialize_bezier_from_edges,
)
from src.inverse_clipasso.render.diffvg_wrapper import render_strokes_rgb
from src.inverse_clipasso.optimize.augmentations import (
    CLIPAugmentations,
    DifferentiableAugmentations,
    get_augmenter,
)
from src.inverse_clipasso.optimize.perceptual_loss import (
    MultiScalePerceptualLoss,
    CLIPFeatureExtractor,
    compute_clip_loss_with_augmentations,
)
from src.inverse_clipasso.optimize.losses import (
    stroke_regularizer,
    total_variation,
    aggregate_losses,
)
from src.inverse_clipasso.optimize.common import (
    CLIP_MEAN,
    CLIP_STD,
    get_lr_multiplier,
    get_scheduled_weight,
    add_gradient_noise,
)


@dataclass
class CLIPassoConfig:
    """Configuration for CLIPasso-style optimization."""
    
    # Target
    target_label: Optional[str] = None
    """Text target for optimization (e.g., 'a fish', 'a cat')."""
    
    target_image: Optional[torch.Tensor] = None
    """Image target for optimization (alternative to text)."""
    
    # Stroke representation
    use_bezier: bool = True
    """Use Bézier curves instead of polylines."""
    
    num_strokes: int = 8
    """Initial number of strokes."""
    
    bezier_samples: int = 50
    """Points to sample per Bézier curve when rendering."""
    
    # Optimization parameters
    steps: int = 2000
    """Number of optimization steps."""
    
    lr_points: float = 1.0
    """Learning rate for control points."""
    
    lr_width: float = 0.1
    """Learning rate for stroke widths."""
    
    lr_opacity: float = 0.01
    """Learning rate for stroke opacities."""
    
    lr_schedule: str = "cosine"
    """Learning rate schedule: 'constant', 'cosine', 'warmup_cosine'."""
    
    warmup_steps: int = 100
    """Warmup steps for lr scheduling."""
    
    min_lr_factor: float = 0.01
    """Minimum LR as fraction of base LR."""
    
    # Canvas
    canvas_size: int = 224
    """Canvas size (224 for CLIP)."""
    
    stroke_width: float = 3.0
    """Default stroke width."""
    
    # Augmentations (KEY for CLIPasso!)
    use_augmentations: bool = True
    """Apply augmentations during CLIP encoding."""
    
    num_augments: int = 4
    """Number of augmented views."""
    
    augment_type: str = "standard"
    """Augmentation type: 'standard' or 'differentiable'."""
    
    crop_scale: Tuple[float, float] = (0.7, 1.0)
    """Scale range for random crops."""
    
    perspective_scale: float = 0.2
    """Maximum perspective distortion."""
    
    rotation_degrees: float = 15.0
    """Maximum rotation in degrees."""
    
    # Multi-scale perceptual loss
    use_perceptual_loss: bool = True
    """Use intermediate CLIP features for perceptual loss."""
    
    perceptual_weight: float = 0.5
    """Weight for perceptual loss."""
    
    # Semantic loss
    semantic_weight: float = 1.0
    """Weight for final CLIP embedding similarity."""
    
    use_multi_prompt: bool = True
    """Average multiple text prompts for robustness."""
    
    prompt_templates: List[str] = field(default_factory=lambda: [
        "a {label}",
        "a sketch of a {label}",
        "a drawing of a {label}",
        "a simple drawing of a {label}",
        "a black and white sketch of a {label}",
    ])
    """Templates for multi-prompt encoding."""
    
    # Regularization
    stroke_reg_weight: float = 0.01
    """Weight for stroke regularization."""
    
    tv_weight: float = 0.001
    """Weight for total variation."""
    
    # Progressive stroke addition
    progressive_strokes: bool = False
    """Add strokes progressively during optimization."""
    
    stroke_add_interval: int = 500
    """Steps between adding new strokes."""
    
    max_strokes: int = 16
    """Maximum number of strokes."""
    
    # Initialization
    init_from_edges: bool = False
    """Initialize strokes from image edges (requires target_image)."""
    
    # Refinement mode
    refinement_mode: bool = False
    """Preserve original stroke structure."""
    
    original_weight: float = 0.3
    """Weight for original structure preservation."""
    
    original_weight_schedule: str = "cosine_up"
    """Schedule for original weight: 'constant', 'linear_up', 'cosine_up'."""
    
    original_weight_start: float = 0.05
    """Starting weight for scheduled original weight."""
    
    # Noise for exploration
    noise_scale: float = 0.0
    """Gradient noise scale."""
    
    noise_decay: float = 0.99
    """Noise decay per step."""
    
    # Gradient clipping
    clip_grad_norm: Optional[float] = 1.0
    """Max gradient norm."""
    
    # Reference image guidance
    reference_image: Optional[torch.Tensor] = None
    """Reference image to guide optimization."""
    
    reference_weight: float = 0.5
    """Weight for reference image loss."""
    
    # Misc
    use_tqdm: bool = True
    """Show progress bar."""
    
    checkpoint_every: int = 0
    """Checkpoint interval (0 to disable)."""
    
    checkpoint_dir: Optional[Path] = None
    """Checkpoint directory."""


@dataclass
class CLIPassoResult:
    """Result of CLIPasso optimization."""
    
    sketch: BezierSketch
    """Final optimized sketch."""
    
    final_strokes: List[torch.Tensor]
    """Final stroke polylines (for compatibility)."""
    
    final_loss: float
    """Final total loss."""
    
    final_similarity: float
    """Final CLIP similarity."""
    
    loss_history: Dict[str, List[float]]
    """Loss history over optimization."""


class CLIPassoOptimizer:
    """
    CLIPasso-style optimizer with full feature set.
    
    Key features over basic optimizer:
    1. Bézier curve strokes for smoother results
    2. Augmentations for robust CLIP guidance
    3. Multi-scale perceptual loss
    4. Separate learning rates for points/widths/opacities
    5. Progressive stroke addition
    
    Example:
        >>> clip_model = ClipModel(device="cuda", use_float32=True)
        >>> config = CLIPassoConfig(
        ...     target_label="a fish",
        ...     use_bezier=True,
        ...     use_augmentations=True,
        ...     use_perceptual_loss=True,
        ... )
        >>> optimizer = CLIPassoOptimizer(clip_model, config)
        >>> result = optimizer.optimize()
    """
    
    def __init__(self, clip_model: ClipModel, config: CLIPassoConfig):
        self.clip = clip_model
        self.config = config
        self.device = clip_model.device
        
        # Setup CLIP normalization
        self._clip_mean = CLIP_MEAN.view(1, 3, 1, 1).to(self.device)
        self._clip_std = CLIP_STD.view(1, 3, 1, 1).to(self.device)
        
        # Setup augmenter
        if config.use_augmentations:
            self.augmenter = get_augmenter(
                config.augment_type,
                num_augments=config.num_augments,
                crop_scale=config.crop_scale,
                perspective_scale=config.perspective_scale,
                rotation_degrees=config.rotation_degrees,
            ).to(self.device)
        else:
            self.augmenter = None
        
        # Setup perceptual loss
        if config.use_perceptual_loss and config.target_image is not None:
            self.perceptual_loss = MultiScalePerceptualLoss(clip_model.model)
            # Pre-compute target features
            with torch.no_grad():
                target_normalized = self._normalize_for_clip(config.target_image.to(self.device))
                self._target_features = self.perceptual_loss.extract_features(target_normalized)
        else:
            self.perceptual_loss = None
            self._target_features = None
        
        # Cache for target embedding
        self._target_embedding: Optional[torch.Tensor] = None
        self._reference_embedding: Optional[torch.Tensor] = None
    
    def _normalize_for_clip(self, image: torch.Tensor) -> torch.Tensor:
        """Apply CLIP normalization."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return (image - self._clip_mean) / self._clip_std
    
    @property
    def target_embedding(self) -> torch.Tensor:
        """Get or compute target embedding."""
        if self._target_embedding is None:
            if self.config.target_label:
                if self.config.use_multi_prompt:
                    prompts = [
                        t.format(label=self.config.target_label)
                        for t in self.config.prompt_templates
                    ]
                    self._target_embedding = self.clip.encode_text_multi(prompts)
                else:
                    self._target_embedding = self.clip.encode_text(self.config.target_label)
            elif self.config.target_image is not None:
                img = self.config.target_image.to(self.device)
                self._target_embedding = self.clip.encode_image(
                    self._normalize_for_clip(img)
                )
            else:
                raise ValueError("Must provide either target_label or target_image")
        return self._target_embedding
    
    @property
    def reference_embedding(self) -> Optional[torch.Tensor]:
        """Get or compute reference image embedding."""
        if self._reference_embedding is None and self.config.reference_image is not None:
            img = self.config.reference_image.to(self.device)
            self._reference_embedding = self.clip.encode_image(
                self._normalize_for_clip(img)
            )
        return self._reference_embedding
    
    def _get_lr_multiplier(self, step: int) -> float:
        """Get learning rate multiplier for current step."""
        return get_lr_multiplier(
            step=step,
            total_steps=self.config.steps,
            schedule=self.config.lr_schedule,
            warmup_steps=self.config.warmup_steps,
            min_lr_factor=self.config.min_lr_factor,
        )
    
    def _get_original_weight(self, step: int) -> float:
        """Get original structure weight for current step."""
        if not self.config.refinement_mode:
            return 0.0
        
        return get_scheduled_weight(
            step=step,
            total_steps=self.config.steps,
            start_weight=self.config.original_weight_start,
            end_weight=self.config.original_weight,
            schedule=self.config.original_weight_schedule,
        )
    
    def _compute_semantic_loss(
        self,
        rendered: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute semantic loss with augmentations.
        
        Returns:
            (loss, similarity) tuple
        """
        normalized = self._normalize_for_clip(rendered)
        
        if self.augmenter is not None:
            augmented = self.augmenter(normalized)
            embeddings = self.clip.encode_image(augmented, requires_grad=True)
            # Average over augmentations
            embedding = embeddings.mean(dim=0, keepdim=True)
        else:
            embedding = self.clip.encode_image(normalized, requires_grad=True)
        
        similarity = F.cosine_similarity(embedding, self.target_embedding).mean()
        loss = 1.0 - similarity
        
        return loss, similarity.item()
    
    def _compute_perceptual_loss(self, rendered: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale perceptual loss."""
        if self.perceptual_loss is None or self._target_features is None:
            return torch.tensor(0.0, device=self.device)
        
        normalized = self._normalize_for_clip(rendered)
        return self.perceptual_loss(normalized, self._target_features)
    
    def _compute_reference_loss(self, rendered: torch.Tensor) -> torch.Tensor:
        """Compute reference image guidance loss."""
        if self.reference_embedding is None:
            return torch.tensor(0.0, device=self.device)
        
        normalized = self._normalize_for_clip(rendered)
        embedding = self.clip.encode_image(normalized, requires_grad=True)
        similarity = F.cosine_similarity(embedding, self.reference_embedding)
        return 1.0 - similarity.mean()
    
    def _compute_original_loss(
        self,
        sketch: BezierSketch,
        original_points: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for deviation from original structure."""
        if not original_points:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        n_points = 0
        
        for i, (stroke, orig) in enumerate(zip(sketch.strokes, original_points)):
            if i >= len(original_points):
                break  # Don't penalize newly added strokes
            
            diff = stroke.control_points - orig.to(self.device)
            total_loss = total_loss + (diff ** 2).sum()
            n_points += stroke.control_points.numel()
        
        if n_points == 0:
            return torch.tensor(0.0, device=self.device)
        
        return total_loss / n_points
    
    def optimize(
        self,
        init_strokes: Optional[List[torch.Tensor]] = None,
        callback: Optional[Callable[[int, Dict[str, float], BezierSketch], None]] = None,
    ) -> CLIPassoResult:
        """
        Run CLIPasso-style optimization.
        
        Args:
            init_strokes: Optional initial strokes (polylines). If None, random init.
            callback: Optional callback(step, losses, sketch) called each step.
        
        Returns:
            CLIPassoResult with optimized sketch.
        """
        # Initialize sketch
        if self.config.init_from_edges and self.config.target_image is not None:
            sketch = initialize_bezier_from_edges(
                self.config.target_image,
                num_strokes=self.config.num_strokes,
                canvas_size=self.config.canvas_size,
                device=self.device,
            )
        elif init_strokes is not None:
            sketch = BezierSketch(
                num_strokes=0,
                canvas_size=self.config.canvas_size,
                stroke_width=self.config.stroke_width,
                device=self.device,
                init_strokes=init_strokes,
            )
        else:
            sketch = BezierSketch(
                num_strokes=self.config.num_strokes,
                canvas_size=self.config.canvas_size,
                stroke_width=self.config.stroke_width,
                device=self.device,
            )
        
        sketch = sketch.to(self.device)
        
        # Save original points for refinement mode
        original_points: List[torch.Tensor] = []
        if self.config.refinement_mode:
            original_points = [s.control_points.detach().clone() for s in sketch.strokes]
        
        # Setup optimizer with separate learning rates
        param_groups = sketch.get_parameters_grouped()
        optimizer = torch.optim.Adam([
            {'params': param_groups['points'], 'lr': self.config.lr_points},
            {'params': param_groups['widths'], 'lr': self.config.lr_width},
            {'params': param_groups['opacities'], 'lr': self.config.lr_opacity},
        ])
        
        # Loss history
        history: Dict[str, List[float]] = {
            'total': [],
            'semantic': [],
            'perceptual': [],
            'reference': [],
            'original': [],
            'stroke_reg': [],
            'tv': [],
            'similarity': [],
            'lr': [],
            'num_strokes': [],
        }
        
        # Noise tracking
        current_noise = self.config.noise_scale
        
        # Progress bar
        steps_iter = range(self.config.steps)
        if self.config.use_tqdm:
            steps_iter = tqdm(
                steps_iter,
                desc=f"CLIPasso → '{self.config.target_label or 'image'}'",
            )
        
        # Main optimization loop
        for step in steps_iter:
            # Update learning rates
            lr_mult = self._get_lr_multiplier(step)
            for i, group in enumerate(optimizer.param_groups):
                # Param groups are: [points0, widths0, opacities0, points1, widths1, ...]
                # After progressive addition: [original_points, original_widths, original_opacities, new_points, new_widths, ...]
                param_type = i % 3  # 0=points, 1=widths, 2=opacities
                base_lr = [self.config.lr_points, self.config.lr_width, self.config.lr_opacity][param_type]
                group['lr'] = base_lr * lr_mult
            
            optimizer.zero_grad()
            
            # Render sketch
            if self.config.use_bezier:
                rendered = render_bezier_strokes(
                    sketch,
                    canvas_size=self.config.canvas_size,
                    samples_per_stroke=self.config.bezier_samples,
                )
            else:
                polylines = sketch.to_polylines()
                rendered = render_strokes_rgb(
                    polylines,
                    canvas_size=self.config.canvas_size,
                    stroke_width=self.config.stroke_width,
                )
            
            # Compute losses
            losses = {}
            
            # Semantic loss (with augmentations)
            semantic_loss, similarity = self._compute_semantic_loss(rendered)
            losses['semantic'] = semantic_loss * self.config.semantic_weight
            
            # Perceptual loss (multi-scale)
            if self.config.use_perceptual_loss:
                perceptual = self._compute_perceptual_loss(rendered)
                losses['perceptual'] = perceptual * self.config.perceptual_weight
            
            # Reference image loss
            if self.config.reference_image is not None:
                ref_loss = self._compute_reference_loss(rendered)
                losses['reference'] = ref_loss * self.config.reference_weight
            
            # Original structure loss (refinement mode)
            if self.config.refinement_mode and original_points:
                orig_weight = self._get_original_weight(step)
                orig_loss = self._compute_original_loss(sketch, original_points)
                losses['original'] = orig_loss * orig_weight
            
            # Regularization
            polylines = sketch.to_polylines()
            if self.config.stroke_reg_weight > 0:
                stroke_reg = stroke_regularizer(polylines)
                losses['stroke_reg'] = stroke_reg * self.config.stroke_reg_weight
            
            if self.config.tv_weight > 0:
                tv = total_variation(rendered)
                losses['tv'] = tv * self.config.tv_weight
            
            # Total loss
            total_loss = sum(losses.values())
            
            # Backward
            total_loss.backward()
            
            # Add noise for exploration
            if current_noise > 0:
                add_gradient_noise(
                    [stroke.control_points for stroke in sketch.strokes],
                    current_noise
                )
                current_noise *= self.config.noise_decay
            
            # Gradient clipping
            if self.config.clip_grad_norm:
                params = [s.control_points for s in sketch.strokes]
                torch.nn.utils.clip_grad_norm_(params, self.config.clip_grad_norm)
            
            # Update
            optimizer.step()
            
            # Clamp to canvas
            sketch.clamp_to_canvas()
            
            # Progressive stroke addition
            if (self.config.progressive_strokes and
                step > 0 and
                step % self.config.stroke_add_interval == 0 and
                len(sketch) < self.config.max_strokes):
                
                sketch.add_stroke()
                # Add new stroke to optimizer
                new_stroke = sketch.strokes[-1]
                optimizer.add_param_group({
                    'params': [new_stroke.control_points],
                    'lr': self.config.lr_points * lr_mult,
                })
                optimizer.add_param_group({
                    'params': [new_stroke.stroke_width],
                    'lr': self.config.lr_width * lr_mult,
                })
                optimizer.add_param_group({
                    'params': [new_stroke.opacity],
                    'lr': self.config.lr_opacity * lr_mult,
                })
                
                if self.config.use_tqdm and hasattr(steps_iter, 'set_description'):
                    steps_iter.set_description(
                        f"CLIPasso → '{self.config.target_label or 'image'}' ({len(sketch)} strokes)"
                    )
            
            # Record history
            history['total'].append(total_loss.item())
            history['semantic'].append(losses.get('semantic', torch.tensor(0.0)).item())
            history['perceptual'].append(losses.get('perceptual', torch.tensor(0.0)).item())
            history['reference'].append(losses.get('reference', torch.tensor(0.0)).item())
            history['original'].append(losses.get('original', torch.tensor(0.0)).item())
            history['stroke_reg'].append(losses.get('stroke_reg', torch.tensor(0.0)).item())
            history['tv'].append(losses.get('tv', torch.tensor(0.0)).item())
            history['similarity'].append(similarity)
            history['lr'].append(self.config.lr_points * lr_mult)
            history['num_strokes'].append(len(sketch))
            
            # Update progress
            if self.config.use_tqdm and hasattr(steps_iter, 'set_postfix'):
                steps_iter.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'sim': f'{similarity:.4f}',
                    'strokes': len(sketch),
                })
            
            # Callback
            if callback is not None:
                callback(step, {k: v[-1] for k, v in history.items()}, sketch)
            
            # Checkpoint
            if (self.config.checkpoint_every > 0 and
                self.config.checkpoint_dir is not None and
                (step + 1) % self.config.checkpoint_every == 0):
                self._save_checkpoint(step, sketch, history)
        
        # Final result
        final_strokes = [s.detach() for s in sketch.to_polylines()]
        
        return CLIPassoResult(
            sketch=sketch,
            final_strokes=final_strokes,
            final_loss=history['total'][-1],
            final_similarity=history['similarity'][-1],
            loss_history=history,
        )
    
    def _save_checkpoint(
        self,
        step: int,
        sketch: BezierSketch,
        history: Dict[str, List[float]],
    ):
        """Save checkpoint."""
        if self.config.checkpoint_dir is None:
            return
        
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'control_points': [s.control_points.detach().cpu() for s in sketch.strokes],
            'widths': [s.stroke_width.detach().cpu() for s in sketch.strokes],
            'opacities': [s.opacity.detach().cpu() for s in sketch.strokes],
            'history': history,
        }
        
        path = self.config.checkpoint_dir / f"clipasso_step{step:04d}.pt"
        torch.save(checkpoint, path)

