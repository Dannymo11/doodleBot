"""
Optimization driver that refines sketches toward semantic targets.

This implements the core "inverse CLIPasso" loop:
1. Render strokes → image
2. Encode image with CLIP
3. Compute semantic loss against target text
4. Backpropagate to stroke parameters
5. Update strokes via gradient descent

Now with CLIPasso-style features:
- Bézier curve representation (smoother strokes)
- Augmentations during CLIP encoding (more robust optimization)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Callable, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.inverse_clipasso.clip.clip_api import ClipModel
from src.inverse_clipasso.render.diffvg_wrapper import render_strokes_rgb
from src.inverse_clipasso.optimize.losses import (
    semantic_loss,
    stroke_regularizer,
    total_variation,
    aggregate_losses,
    original_strokes_loss,
    exemplar_loss,
    reference_image_loss,
)
from src.inverse_clipasso.optimize.common import (
    CLIP_MEAN,
    CLIP_STD,
    STROKE_RADIUS_MIN,
    STROKE_RADIUS_MAX,
    SINGLE_POINT_OFFSET,
    STROKE_INIT_PADDING,
    get_lr_multiplier,
    get_scheduled_weight,
    sample_bezier_curve,
    polyline_to_bezier,
    polyline_to_bezier_segments,
    create_random_stroke,
    add_gradient_noise,
)


@dataclass
class OptimizationConfig:
    """Configuration for sketch optimization."""
    
    label: str
    """Target semantic label (e.g., 'a house', 'a cat')."""
    
    steps: int = 200
    """Number of optimization steps."""
    
    lr: float = 1.0
    """Learning rate for stroke parameters."""
    
    lr_schedule: str = "cosine"
    """Learning rate schedule: 'constant', 'cosine', 'linear', 'warmup_cosine'."""
    
    warmup_steps: int = 50
    """Number of warmup steps (for warmup_cosine schedule)."""
    
    min_lr: float = 0.01
    """Minimum learning rate for scheduling."""
    
    canvas_size: int = 224
    """Canvas size (square). 224 is standard for CLIP."""
    
    stroke_width: float = 3.0
    """Width of rendered strokes."""
    
    noise_scale: float = 0.0
    """Scale of noise to add to gradients (helps escape local minima)."""
    
    noise_decay: float = 0.99
    """Decay factor for noise scale each step."""
    
    checkpoint_every: int = 50
    """Save checkpoint every N steps (0 to disable)."""
    
    checkpoint_dir: Optional[Path] = None
    """Directory for saving checkpoints."""
    
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic": 1.0,
        "stroke": 0.01,
        "tv": 0.001,
    })
    """Weights for combining loss terms."""
    
    use_tqdm: bool = True
    """Show progress bar during optimization."""
    
    clip_grad_norm: Optional[float] = 1.0
    """Max gradient norm for clipping (None to disable)."""
    
    # Multi-prompt support
    use_multi_prompt: bool = False
    """If True, uses multiple prompts averaged together for more robust targeting."""
    
    prompt_templates: List[str] = field(default_factory=lambda: [
        "a {label}",
        "a drawing of a {label}",
        "a sketch of a {label}",
        "a simple {label}",
        "a {label} drawing",
    ])
    """Template prompts for multi-prompt mode. {label} is replaced with the target label."""
    
    # Refinement mode
    refinement_mode: bool = False
    """If True, preserves original stroke positions (for same-category refinement)."""
    
    original_weight: float = 0.5
    """Weight for original strokes preservation loss (only used in refinement mode)."""
    
    original_weight_schedule: str = "constant"
    """Schedule for original_weight: 'constant', 'linear_up', 'cosine_up'.
    - constant: use original_weight throughout
    - linear_up: start at original_weight_start, linearly increase to original_weight
    - cosine_up: start at original_weight_start, cosine increase to original_weight
    This allows more CLIP exploration early, then locks in structure later."""
    
    original_weight_start: float = 0.1
    """Starting weight for original strokes (only for linear_up/cosine_up schedules)."""
    
    # Exemplar-based guidance (QuickDraw references)
    use_exemplar_loss: bool = False
    """If True, guides optimization toward reference sketch examples."""
    
    exemplar_embeddings: Optional[torch.Tensor] = None
    """Pre-computed CLIP embeddings of reference sketches [N, D]."""
    
    exemplar_weight: float = 0.5
    """Weight for exemplar loss."""
    
    exemplar_mode: str = "mean"
    """How to aggregate exemplar similarities: 'mean', 'min', 'softmin'."""
    
    # Reference image guidance (use a perfect sketch as target)
    use_reference_image: bool = True
    """If True, guides optimization toward a reference image embedding."""
    
    reference_image_embedding: Optional[torch.Tensor] = None
    """Pre-computed CLIP embedding of the reference image [1, D]."""
    
    reference_image_weight: float = 2.0
    """Weight for reference image loss (how strongly to pull toward the reference)."""
    
    blend_text_and_reference: bool = False
    """If True, blends text embedding with reference image embedding."""
    
    text_reference_blend: float = 0.5
    """Blend ratio: 0.0 = pure text, 1.0 = pure reference image."""
    
    # Style loss (VGG-based, separate from CLIP)
    use_style_loss: bool = False
    """If True, adds VGG-based style loss using Gram matrices."""
    
    style_image: Optional[torch.Tensor] = None
    """Style reference image [1, 3, H, W] in range [0, 1]. Texture/style will be matched."""
    
    style_weight: float = 1.0
    """Weight for style loss. Higher = stronger texture matching."""
    
    style_layers: Optional[List[str]] = None
    """VGG layers for style extraction. Default: ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']"""
    
    # Stroke addition during optimization
    allow_stroke_addition: bool = False
    """If True, allows adding new strokes during optimization."""
    
    stroke_add_interval: int = 200
    """Add a new stroke every N steps (only if allow_stroke_addition is True)."""
    
    max_strokes: int = 20
    """Maximum number of strokes to have (won't add beyond this)."""
    
    new_stroke_points: int = 4
    """Number of control points for newly added strokes."""
    
    stroke_init_mode: str = "random"
    """How to initialize new strokes: 'random', 'center', 'edge'."""
    
    # ===== CLIPasso-style features =====
    
    # Bézier curves
    use_bezier: bool = False
    """If True, converts strokes to Bézier curves for smoother optimization."""
    
    bezier_samples: int = 50
    """Number of points to sample per Bézier curve when rendering."""
    
    bezier_multi_segment: bool = True
    """If True, splits long strokes into multiple Bézier curves for better fidelity."""
    
    bezier_max_points_per_segment: int = 6
    """Max points per Bézier segment when using multi-segment mode.
    Smaller = more segments = higher fidelity but more parameters to optimize."""
    
    # Augmentations (KEY for robust CLIP guidance)
    use_augmentations: bool = False
    """If True, applies random augmentations during CLIP encoding."""
    
    num_augments: int = 4
    """Number of augmented views to generate (including original)."""
    
    augment_crop_scale: Tuple[float, float] = (0.7, 1.0)
    """Scale range for random crops."""
    
    augment_perspective: float = 0.15
    """Maximum perspective distortion."""
    
    augment_rotation: float = 10.0
    """Maximum rotation in degrees."""


@dataclass
class OptimizationResult:
    """Result of optimization run."""
    
    final_strokes: List[torch.Tensor]
    """Optimized stroke parameters."""
    
    final_loss: float
    """Final total loss value."""
    
    loss_history: Dict[str, List[float]]
    """History of each loss term over optimization."""
    
    final_similarity: float
    """Final CLIP similarity to target."""


class InverseClipassoOptimizer:
    """
    Optimizer that refines sketches toward semantic targets using CLIP guidance.
    
    Example:
        >>> clip_model = ClipModel(device="cuda", use_float32=True)
        >>> config = OptimizationConfig(label="a cat", steps=100)
        >>> optimizer = InverseClipassoOptimizer(clip_model, config)
        >>> 
        >>> # Start from some initial strokes
        >>> init_strokes = [torch.randn(10, 2, requires_grad=True)]
        >>> result = optimizer.optimize(init_strokes)
        >>> print(f"Final similarity: {result.final_similarity:.4f}")
    """
    
    def __init__(self, clip_model: ClipModel, config: OptimizationConfig):
        """
        Initialize optimizer.
        
        Args:
            clip_model: CLIP model for computing semantic loss.
                       Should be initialized with use_float32=True for gradients.
            config: Optimization configuration.
        """
        self.clip = clip_model
        self.config = config
        self.device = clip_model.device
        
        # Pre-compute target text embedding (doesn't change during optimization)
        self._target_embedding: Optional[torch.Tensor] = None
        
        # Setup CLIP normalization tensors
        self._clip_mean = CLIP_MEAN.view(1, 3, 1, 1).to(self.device)
        self._clip_std = CLIP_STD.view(1, 3, 1, 1).to(self.device)
        
        # Setup augmenter if enabled
        self.augmenter = None
        if config.use_augmentations:
            from src.inverse_clipasso.optimize.augmentations import CLIPAugmentations
            self.augmenter = CLIPAugmentations(
                num_augments=config.num_augments,
                crop_scale=config.augment_crop_scale,
                perspective_scale=config.augment_perspective,
                rotation_degrees=config.augment_rotation,
            )
        
        # Setup VGG style loss if enabled (separate from CLIP)
        self.style_loss_module = None
        if config.use_style_loss and config.style_image is not None:
            from src.inverse_clipasso.optimize.style_loss import VGGStyleLoss
            self.style_loss_module = VGGStyleLoss(
                layers=config.style_layers,
                device=str(self.device),
            )
            # Pre-compute style Gram matrices from reference image
            style_img = config.style_image.to(self.device)
            self.style_loss_module.set_style_target(style_img)
            print(f"  ✅ VGG style loss enabled (layers: {self.style_loss_module.layers})")
        
        # Setup Bézier sketch if enabled
        self._bezier_sketch = None
    
    @property
    def target_embedding(self) -> torch.Tensor:
        """Lazily compute and cache target text embedding."""
        if self._target_embedding is None:
            if self.config.use_multi_prompt:
                # Generate prompts from templates
                prompts = [
                    template.format(label=self.config.label)
                    for template in self.config.prompt_templates
                ]
                self._target_embedding = self.clip.encode_text_multi(prompts)
            else:
                self._target_embedding = self.clip.encode_text(self.config.label)
        return self._target_embedding
    
    def _normalize_for_clip(self, image: torch.Tensor) -> torch.Tensor:
        """Apply CLIP normalization to rendered image."""
        return (image - self._clip_mean) / self._clip_std
    
    def _get_lr(self, step: int) -> float:
        """Compute learning rate for current step based on schedule."""
        multiplier = get_lr_multiplier(
            step=step,
            total_steps=self.config.steps,
            schedule=self.config.lr_schedule,
            warmup_steps=self.config.warmup_steps,
            min_lr_factor=self.config.min_lr / self.config.lr,
        )
        return self.config.lr * multiplier
    
    def _get_original_weight(self, step: int) -> float:
        """Compute original strokes weight for current step based on schedule.
        
        This allows more CLIP-guided exploration early in training,
        then gradually increases structure preservation to lock in results.
        """
        if not self.config.refinement_mode:
            return 0.0
        
        return get_scheduled_weight(
            step=step,
            total_steps=self.config.steps,
            start_weight=self.config.original_weight_start,
            end_weight=self.config.original_weight,
            schedule=self.config.original_weight_schedule,
        )
    
    def _create_new_stroke(
        self,
        existing_strokes: List[torch.Tensor],
        canvas_size: int = 224,
    ) -> torch.Tensor:
        """
        Create a new stroke to add to the sketch.
        
        Args:
            existing_strokes: Current strokes (used to find empty areas).
            canvas_size: Size of the canvas.
        
        Returns:
            New stroke tensor with shape [num_points, 2].
        """
        return create_random_stroke(
            canvas_size=canvas_size,
            num_points=self.config.new_stroke_points,
            mode=self.config.stroke_init_mode,
            device=str(self.device),
            padding=STROKE_INIT_PADDING,
        )
    
    def _polyline_to_bezier_points(self, polyline: torch.Tensor) -> torch.Tensor:
        """
        Convert a polyline to cubic Bézier control points.
        
        Takes a polyline [N, 2] and returns 4 control points [4, 2] for a cubic Bézier.
        """
        return polyline_to_bezier(polyline)
    
    def _sample_bezier_curve(self, control_points: torch.Tensor, num_samples: int = 50) -> torch.Tensor:
        """
        Sample points along a cubic Bézier curve.
        
        Args:
            control_points: [4, 2] tensor of control points P0, P1, P2, P3
            num_samples: Number of points to sample
        
        Returns:
            [num_samples, 2] tensor of sampled points
        """
        return sample_bezier_curve(control_points, num_samples)
    
    def _render_with_bezier(self, bezier_controls: List[torch.Tensor]) -> torch.Tensor:
        """
        Render Bézier control points by sampling curves and rendering as polylines.
        
        Args:
            bezier_controls: List of [4, 2] Bézier control point tensors
        
        Returns:
            Rendered image [1, 3, H, W]
        """
        # Sample each Bézier curve to get polylines
        polylines = []
        for controls in bezier_controls:
            sampled = self._sample_bezier_curve(controls, self.config.bezier_samples)
            polylines.append(sampled)
        
        # Render the sampled polylines
        return render_strokes_rgb(
            polylines,
            canvas_size=self.config.canvas_size,
            stroke_width=self.config.stroke_width,
        )
    
    def _compute_losses(
        self, 
        strokes: List[torch.Tensor],
        rendered_image: torch.Tensor,
        original_strokes: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss terms.
        
        Args:
            strokes: List of stroke parameter tensors (polylines or Bézier controls).
            rendered_image: Rendered RGB image [1, 3, H, W].
            original_strokes: Original stroke positions (for refinement mode).
        
        Returns:
            Dictionary of loss name → loss value.
        """
        losses = {}
        
        # Semantic loss (CLIP alignment) - with optional augmentations
        img_normalized = self._normalize_for_clip(rendered_image)
        
        if self.augmenter is not None:
            # Apply augmentations for more robust CLIP guidance
            augmented = self.augmenter(img_normalized)  # [N, 3, H, W]
            img_embeddings = self.clip.encode_image(augmented, requires_grad=True)
            # Average embeddings across augmentations
            img_embedding = img_embeddings.mean(dim=0, keepdim=True)
        else:
            img_embedding = self.clip.encode_image(img_normalized, requires_grad=True)
        
        losses["semantic"] = semantic_loss(img_embedding, self.target_embedding).mean()
        
        # Exemplar loss (guide toward good reference sketches)
        if self.config.use_exemplar_loss and self.config.exemplar_embeddings is not None:
            losses["exemplar"] = exemplar_loss(
                img_embedding,
                self.config.exemplar_embeddings.to(self.device),
                mode=self.config.exemplar_mode,
            ).mean()
        
        # Reference image loss (guide toward a perfect sketch)
        if self.config.use_reference_image and self.config.reference_image_embedding is not None:
            losses["reference"] = reference_image_loss(
                img_embedding,
                self.config.reference_image_embedding.to(self.device),
            ).mean()
        
        # Style loss (VGG-based, separate from CLIP)
        if self.config.use_style_loss and self.style_loss_module is not None:
            # Style loss uses raw image (not CLIP-normalized) 
            losses["style"] = self.style_loss_module(rendered_image)
        
        # Stroke regularization (smoothness + length)
        # For Bézier, we regularize the control points directly
        if self.config.loss_weights.get("stroke", 0) > 0:
            if self.config.use_bezier:
                # For Bézier, sample curves first then regularize
                polylines = [self._sample_bezier_curve(s, 20) for s in strokes]
                losses["stroke"] = stroke_regularizer(polylines)
            else:
                losses["stroke"] = stroke_regularizer(strokes)
        
        # Total variation (image smoothness)
        if self.config.loss_weights.get("tv", 0) > 0:
            losses["tv"] = total_variation(rendered_image)
        
        # Original strokes preservation (refinement mode)
        if self.config.refinement_mode and original_strokes is not None:
            losses["original"] = original_strokes_loss(strokes, original_strokes)
        
        return losses
    
    def optimize(
        self, 
        init_strokes: List[torch.Tensor],
        callback: Optional[Callable[[int, Dict[str, float], List[torch.Tensor]], None]] = None,
    ) -> OptimizationResult:
        """
        Run the main optimization loop.
        
        Args:
            init_strokes: Initial stroke parameters. Each tensor should be [N_i, 2]
                         representing control points. Will be cloned and optimized.
            callback: Optional function called each step with (step, losses, strokes).
        
        Returns:
            OptimizationResult with final strokes and loss history.
        """
        # Convert to Bézier control points if enabled
        # Track original stroke boundaries for multi-segment mode
        stroke_segment_counts: List[int] = []  # Number of Bézier segments per original stroke
        
        if self.config.use_bezier:
            strokes = []
            total_segments = 0
            
            for s in init_strokes:
                stroke = s.clone().detach().to(self.device)
                
                if self.config.bezier_multi_segment:
                    # Convert to multiple Bézier curves for better fidelity
                    bezier_list = polyline_to_bezier_segments(
                        stroke, 
                        max_points_per_segment=self.config.bezier_max_points_per_segment
                    )
                    stroke_segment_counts.append(len(bezier_list))
                    for bezier_controls in bezier_list:
                        bezier_controls = bezier_controls.clone().detach()
                        bezier_controls.requires_grad_(True)
                        strokes.append(bezier_controls)
                    total_segments += len(bezier_list)
                else:
                    # Single Bézier per stroke (original behavior)
                    bezier_controls = self._polyline_to_bezier_points(stroke)
                    bezier_controls.requires_grad_(True)
                    strokes.append(bezier_controls)
                    stroke_segment_counts.append(1)
                    total_segments += 1
            
            if self.config.bezier_multi_segment:
                print(f"  ✅ Converted {len(init_strokes)} strokes to {total_segments} Bézier segments")
            else:
                print(f"  ✅ Converted {len(strokes)} strokes to Bézier curves")
        else:
            # Clone strokes and ensure they require gradients
            strokes = []
            for s in init_strokes:
                stroke = s.clone().detach().to(self.device)
                stroke.requires_grad_(True)
                strokes.append(stroke)
                stroke_segment_counts.append(1)
        
        # Save original strokes for refinement mode (detached, no grad)
        original_strokes: Optional[List[torch.Tensor]] = None
        if self.config.refinement_mode:
            original_strokes = [s.clone().detach() for s in strokes]
        
        # Create local copy of loss weights to avoid mutating config
        # This ensures the optimizer can be run multiple times with consistent behavior
        loss_weights = self.config.loss_weights.copy()
        
        # Apply specific weight configs for optional losses
        if self.config.use_exemplar_loss:
            loss_weights["exemplar"] = self.config.exemplar_weight
        if self.config.use_reference_image:
            loss_weights["reference"] = self.config.reference_image_weight
        if self.config.use_style_loss:
            loss_weights["style"] = self.config.style_weight
        
        # Setup optimizer (use SGD with momentum for better exploration)
        optimizer = torch.optim.Adam(strokes, lr=self.config.lr)
        
        # Loss history tracking
        loss_history: Dict[str, List[float]] = {
            "total": [],
            "semantic": [],
            "exemplar": [],  # For exemplar-guided refinement
            "reference": [],  # For reference image guidance
            "style": [],  # For VGG-based style loss
            "stroke": [],
            "tv": [],
            "original": [],  # For refinement mode
            "original_weight": [],  # Track scheduled weight
            "similarity": [],
            "lr": [],
            "num_strokes": [],  # Track stroke count over time
        }
        
        # Noise tracking
        current_noise_scale = self.config.noise_scale
        
        # Log CLIPasso features
        if self.config.use_augmentations:
            print(f"  ✅ Augmentations enabled ({self.config.num_augments} views)")
        
        # Progress bar
        desc = f"Optimizing → '{self.config.label}'"
        if self.config.use_bezier:
            desc += " [Bézier]"
        if self.config.use_augmentations:
            desc += " [Aug]"
        
        steps_iter = range(self.config.steps)
        if self.config.use_tqdm:
            steps_iter = tqdm(steps_iter, desc=desc)
        
        # Main optimization loop
        for step in steps_iter:
            # Update learning rate based on schedule
            current_lr = self._get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Update original weight based on schedule (refinement mode)
            current_original_weight = 0.0
            if self.config.refinement_mode:
                current_original_weight = self._get_original_weight(step)
                loss_weights["original"] = current_original_weight
            
            optimizer.zero_grad()
            
            # Render strokes to image
            if self.config.use_bezier:
                rendered = self._render_with_bezier(strokes)
            else:
                rendered = render_strokes_rgb(
                    strokes,
                    canvas_size=self.config.canvas_size,
                    stroke_width=self.config.stroke_width,
                    device=self.device,
                )
            
            # Compute losses
            losses = self._compute_losses(strokes, rendered, original_strokes)
            
            # Aggregate weighted loss
            total_loss = aggregate_losses(losses, loss_weights)
            
            # Backward pass
            total_loss.backward()
            
            # Add noise to gradients (helps escape local minima)
            if current_noise_scale > 0:
                add_gradient_noise(strokes, current_noise_scale)
                current_noise_scale *= self.config.noise_decay
            
            # Gradient clipping
            if self.config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(strokes, self.config.clip_grad_norm)
            
            # Update strokes
            optimizer.step()
            
            # Clamp strokes to canvas bounds
            with torch.no_grad():
                for stroke in strokes:
                    stroke.clamp_(0, self.config.canvas_size)
            
            # Add new strokes at intervals (if enabled)
            if self.config.allow_stroke_addition:
                if (step > 0 and 
                    step % self.config.stroke_add_interval == 0 and 
                    len(strokes) < self.config.max_strokes):
                    
                    # Create new stroke
                    new_stroke = self._create_new_stroke(strokes, self.config.canvas_size)
                    strokes.append(new_stroke)
                    
                    # Add new stroke to optimizer
                    optimizer.add_param_group({'params': [new_stroke], 'lr': current_lr})
                    
                    if self.config.use_tqdm and hasattr(steps_iter, 'set_description'):
                        steps_iter.set_description(
                            f"Optimizing → '{self.config.label}' ({len(strokes)} strokes)"
                        )
            
            # Record history
            loss_history["total"].append(total_loss.item())
            loss_history["semantic"].append(losses.get("semantic", torch.tensor(0.0)).item())
            loss_history["exemplar"].append(losses.get("exemplar", torch.tensor(0.0)).item())
            loss_history["reference"].append(losses.get("reference", torch.tensor(0.0)).item())
            loss_history["style"].append(losses.get("style", torch.tensor(0.0)).item())
            loss_history["stroke"].append(losses.get("stroke", torch.tensor(0.0)).item())
            loss_history["tv"].append(losses.get("tv", torch.tensor(0.0)).item())
            loss_history["original"].append(losses.get("original", torch.tensor(0.0)).item())
            loss_history["original_weight"].append(current_original_weight)
            loss_history["lr"].append(current_lr)
            loss_history["num_strokes"].append(len(strokes))
            
            # Compute similarity for tracking
            with torch.no_grad():
                similarity = 1.0 - losses["semantic"].item()
            loss_history["similarity"].append(similarity)
            
            # Update progress bar
            if self.config.use_tqdm and hasattr(steps_iter, 'set_postfix'):
                postfix = {
                    'loss': f'{total_loss.item():.4f}',
                    'sim': f'{similarity:.4f}',
                    'lr': f'{current_lr:.4f}',
                }
                if self.config.refinement_mode:
                    postfix['ow'] = f'{current_original_weight:.3f}'
                if self.config.allow_stroke_addition:
                    postfix['strokes'] = len(strokes)
                steps_iter.set_postfix(postfix)
            
            # Callback
            if callback is not None:
                callback(step, {k: v[-1] for k, v in loss_history.items()}, strokes)
            
            # Checkpoint
            if (self.config.checkpoint_every > 0 and 
                self.config.checkpoint_dir is not None and
                (step + 1) % self.config.checkpoint_every == 0):
                self._save_checkpoint(step, strokes, loss_history)
        
        # Detach final strokes
        if self.config.use_bezier:
            # Convert Bézier control points back to polylines for consistent output
            final_strokes = []
            for bezier_controls in strokes:
                polyline = self._sample_bezier_curve(
                    bezier_controls.detach(), 
                    self.config.bezier_samples
                )
                final_strokes.append(polyline.clone())
        else:
            final_strokes = [s.detach().clone() for s in strokes]
        
        return OptimizationResult(
            final_strokes=final_strokes,
            final_loss=loss_history["total"][-1],
            loss_history=loss_history,
            final_similarity=loss_history["similarity"][-1],
        )
    
    def optimize_iter(
        self, 
        init_strokes: List[torch.Tensor],
    ) -> Iterator[tuple[int, Dict[str, float], List[torch.Tensor]]]:
        """
        Generator version of optimize that yields intermediate results.
        
        Yields:
            Tuple of (step, losses_dict, current_strokes) at each step.
        """
        results = []
        
        def collect(step, losses, strokes):
            results.append((step, losses.copy(), [s.detach().clone() for s in strokes]))
        
        # Run optimization with callback
        self.optimize(init_strokes, callback=collect)
        
        # Yield collected results
        yield from results
    
    def _save_checkpoint(
        self, 
        step: int, 
        strokes: List[torch.Tensor],
        loss_history: Dict[str, List[float]],
    ) -> None:
        """Save optimization checkpoint."""
        if self.config.checkpoint_dir is None:
            return
        
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'strokes': [s.detach().cpu() for s in strokes],
            'loss_history': loss_history,
            'config': {
                'label': self.config.label,
                'steps': self.config.steps,
                'lr': self.config.lr,
            },
        }
        
        path = self.config.checkpoint_dir / f"checkpoint_step{step:04d}.pt"
        torch.save(checkpoint, path)
