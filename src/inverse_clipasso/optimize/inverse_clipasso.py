"""
Optimization driver that refines sketches toward semantic targets.

This implements the core "inverse CLIPasso" loop:
1. Render strokes → image
2. Encode image with CLIP
3. Compute semantic loss against target text
4. Backpropagate to stroke parameters
5. Update strokes via gradient descent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Callable

import torch
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


# CLIP normalization constants
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


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
    use_reference_image: bool = False
    """If True, guides optimization toward a reference image embedding."""
    
    reference_image_embedding: Optional[torch.Tensor] = None
    """Pre-computed CLIP embedding of the reference image [1, D]."""
    
    reference_image_weight: float = 1.0
    """Weight for reference image loss (how strongly to pull toward the reference)."""
    
    blend_text_and_reference: bool = False
    """If True, blends text embedding with reference image embedding."""
    
    text_reference_blend: float = 0.5
    """Blend ratio: 0.0 = pure text, 1.0 = pure reference image."""
    
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
        total_steps = self.config.steps
        base_lr = self.config.lr
        min_lr = self.config.min_lr
        warmup_steps = self.config.warmup_steps
        schedule = self.config.lr_schedule
        
        if schedule == "constant":
            return base_lr
        
        elif schedule == "linear":
            # Linear decay from base_lr to min_lr
            progress = step / max(total_steps - 1, 1)
            return base_lr + (min_lr - base_lr) * progress
        
        elif schedule == "cosine":
            # Cosine annealing from base_lr to min_lr
            import math
            progress = step / max(total_steps - 1, 1)
            return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        elif schedule == "warmup_cosine":
            # Warmup then cosine decay
            import math
            if step < warmup_steps:
                # Linear warmup
                return min_lr + (base_lr - min_lr) * (step / warmup_steps)
            else:
                # Cosine decay after warmup
                progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
                return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        else:
            raise ValueError(f"Unknown lr_schedule: {schedule}")
    
    def _get_original_weight(self, step: int) -> float:
        """Compute original strokes weight for current step based on schedule.
        
        This allows more CLIP-guided exploration early in training,
        then gradually increases structure preservation to lock in results.
        """
        if not self.config.refinement_mode:
            return 0.0
        
        total_steps = self.config.steps
        start_weight = self.config.original_weight_start
        end_weight = self.config.original_weight
        schedule = self.config.original_weight_schedule
        
        if schedule == "constant":
            return end_weight
        
        elif schedule == "linear_up":
            # Linear increase from start to end
            progress = step / max(total_steps - 1, 1)
            return start_weight + (end_weight - start_weight) * progress
        
        elif schedule == "cosine_up":
            # Cosine increase from start to end (slower initially, faster later)
            import math
            progress = step / max(total_steps - 1, 1)
            # Cosine from 1 to 0, so we use (1 - cos) / 2 to go from 0 to 1
            cosine_progress = (1 - math.cos(math.pi * progress)) / 2
            return start_weight + (end_weight - start_weight) * cosine_progress
        
        else:
            return end_weight
    
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
        num_points = self.config.new_stroke_points
        mode = self.config.stroke_init_mode
        padding = 20  # Keep strokes away from edges
        
        if mode == "random":
            # Random position and shape
            center_x = torch.rand(1).item() * (canvas_size - 2 * padding) + padding
            center_y = torch.rand(1).item() * (canvas_size - 2 * padding) + padding
            
            # Create a small stroke around the center
            angles = torch.linspace(0, 2 * 3.14159, num_points + 1)[:-1]
            radius = 15 + torch.rand(1).item() * 20  # Random radius 15-35
            
            points = torch.zeros(num_points, 2)
            for i, angle in enumerate(angles):
                # Add some randomness to each point
                r = radius * (0.8 + 0.4 * torch.rand(1).item())
                points[i, 0] = center_x + r * torch.cos(angle)
                points[i, 1] = center_y + r * torch.sin(angle)
            
        elif mode == "center":
            # Start from center of canvas
            center_x = canvas_size / 2
            center_y = canvas_size / 2
            
            # Small stroke near center
            points = torch.zeros(num_points, 2)
            for i in range(num_points):
                points[i, 0] = center_x + (i - num_points/2) * 10 + torch.rand(1).item() * 5
                points[i, 1] = center_y + torch.rand(1).item() * 20 - 10
            
        elif mode == "edge":
            # Start from a random edge
            edge = torch.randint(0, 4, (1,)).item()  # 0=top, 1=right, 2=bottom, 3=left
            
            if edge == 0:  # Top
                start_x = torch.rand(1).item() * canvas_size
                start_y = padding
            elif edge == 1:  # Right
                start_x = canvas_size - padding
                start_y = torch.rand(1).item() * canvas_size
            elif edge == 2:  # Bottom
                start_x = torch.rand(1).item() * canvas_size
                start_y = canvas_size - padding
            else:  # Left
                start_x = padding
                start_y = torch.rand(1).item() * canvas_size
            
            # Create stroke moving inward
            points = torch.zeros(num_points, 2)
            direction = torch.rand(2) - 0.5
            direction = direction / direction.norm() * 15  # Normalize to length 15
            
            for i in range(num_points):
                points[i, 0] = start_x + direction[0] * i
                points[i, 1] = start_y + direction[1] * i
        
        else:
            # Default to random
            points = torch.rand(num_points, 2) * (canvas_size - 2 * padding) + padding
        
        # Ensure points are within bounds
        points = points.clamp(padding, canvas_size - padding)
        
        return points.to(self.device).requires_grad_(True)
    
    def _compute_losses(
        self, 
        strokes: List[torch.Tensor],
        rendered_image: torch.Tensor,
        original_strokes: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss terms.
        
        Args:
            strokes: List of stroke parameter tensors.
            rendered_image: Rendered RGB image [1, 3, H, W].
            original_strokes: Original stroke positions (for refinement mode).
        
        Returns:
            Dictionary of loss name → loss value.
        """
        losses = {}
        
        # Semantic loss (CLIP alignment)
        img_normalized = self._normalize_for_clip(rendered_image)
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
        
        # Stroke regularization (smoothness + length)
        if self.config.loss_weights.get("stroke", 0) > 0:
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
        # Clone strokes and ensure they require gradients
        strokes = []
        for s in init_strokes:
            stroke = s.clone().detach().to(self.device)
            stroke.requires_grad_(True)
            strokes.append(stroke)
        
        # Save original strokes for refinement mode (detached, no grad)
        original_strokes: Optional[List[torch.Tensor]] = None
        if self.config.refinement_mode:
            original_strokes = [s.clone().detach() for s in strokes]
            # Initialize original weight in loss_weights (will be updated each step)
            if "original" not in self.config.loss_weights:
                self.config.loss_weights["original"] = 0.0  # Will be updated per-step
        
        # Setup optimizer (use SGD with momentum for better exploration)
        optimizer = torch.optim.Adam(strokes, lr=self.config.lr)
        
        # Loss history tracking
        loss_history: Dict[str, List[float]] = {
            "total": [],
            "semantic": [],
            "exemplar": [],  # For exemplar-guided refinement
            "reference": [],  # For reference image guidance
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
        
        # Progress bar
        steps_iter = range(self.config.steps)
        if self.config.use_tqdm:
            steps_iter = tqdm(steps_iter, desc=f"Optimizing → '{self.config.label}'")
        
        # Main optimization loop
        for step in steps_iter:
            # Update learning rate based on schedule
            current_lr = self._get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Update original weight based on schedule (refinement mode)
            if self.config.refinement_mode:
                current_original_weight = self._get_original_weight(step)
                self.config.loss_weights["original"] = current_original_weight
            
            optimizer.zero_grad()
            
            # Render strokes to image
            rendered = render_strokes_rgb(
                strokes,
                canvas_size=self.config.canvas_size,
                stroke_width=self.config.stroke_width,
            )
            
            # Compute losses
            losses = self._compute_losses(strokes, rendered, original_strokes)
            
            # Aggregate weighted loss
            total_loss = aggregate_losses(losses, self.config.loss_weights)
            
            # Backward pass
            total_loss.backward()
            
            # Add noise to gradients (helps escape local minima)
            if current_noise_scale > 0:
                for stroke in strokes:
                    if stroke.grad is not None:
                        noise = torch.randn_like(stroke.grad) * current_noise_scale
                        stroke.grad.add_(noise)
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
            loss_history["stroke"].append(losses.get("stroke", torch.tensor(0.0)).item())
            loss_history["tv"].append(losses.get("tv", torch.tensor(0.0)).item())
            loss_history["original"].append(losses.get("original", torch.tensor(0.0)).item())
            loss_history["original_weight"].append(
                self.config.loss_weights.get("original", 0.0) if self.config.refinement_mode else 0.0
            )
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
                    postfix['ow'] = f'{self.config.loss_weights.get("original", 0):.3f}'
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
