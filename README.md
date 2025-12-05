# doodleBot

**Sketch refinement using CLIP guidance** — Transform rough doodles into better drawings through differentiable optimization.

![Fish Refinement](outputs/long_optimization/fish_to_A%20perfect%20sketch%20of%20a%20fish_progression.png)

## Overview

doodleBot implements **inverse CLIPasso**: instead of converting photos to sketches (like CLIPasso), we refine sketches toward semantic targets using CLIP's understanding of visual concepts.

### Key Features

- 🎨 **Semantic Optimization**: Guide sketches toward text descriptions ("a perfect fish", "a detailed bird")
- 🖼️ **Reference Image Guidance**: Pull sketches toward a "perfect" reference image
- 📷 **Photo Exemplars**: Use real photos to provide rich semantic guidance  
- ✏️ **Stroke Addition**: Dynamically add strokes during optimization
- 🔄 **Refinement Mode**: Improve sketches while preserving their original character
- 📊 **Multi-prompt Averaging**: Robust optimization using multiple text prompts

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  QuickDraw      │     │  Differentiable │     │  CLIP           │
│  Sketch         │ ──▶ │  Renderer       │ ──▶ │  Encoder        │
│  (strokes)      │     │  (diffvg)       │     │  (ViT-B/32)     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
        ▲                                                │
        │                                                ▼
        │                                       ┌─────────────────┐
        │         Gradient Descent              │  Semantic Loss  │
        └────────────────────────────────────── │  (cosine sim)   │
                                                └─────────────────┘
```

1. **Render** stroke parameters to a raster image using diffvg
2. **Encode** the image with CLIP's vision encoder
3. **Compare** to target text/image embeddings
4. **Backpropagate** gradients to stroke control points
5. **Update** stroke positions via gradient descent

## Installation

### Option 1: pip (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: Conda
```bash
conda env create -f env.yml
conda activate doodlebot
```

### Requirements
- Python 3.10+
- PyTorch with CUDA/MPS support (optional but recommended)
- diffvg for differentiable rendering

## Quick Start

### 1. Prepare QuickDraw Data
Place QuickDraw NDJSON files in `data/quickdraw/raw/strokes/`:
```
data/quickdraw/raw/strokes/
├── fish.ndjson
├── cat.ndjson
├── bird.ndjson
└── ...
```

### 2. Run Optimization
```bash
python tests/test_long_optimization.py
```

This will:
- Load a random fish sketch from QuickDraw
- Optimize it toward "A perfect sketch of a fish"
- Generate progression grid, animation GIF, and summary plots

## Usage

### Basic Text-Guided Optimization

```python
from src.inverse_clipasso.clip import ClipModel
from src.inverse_clipasso.optimize import InverseClipassoOptimizer, OptimizationConfig

# Load CLIP
clip = ClipModel(device="cuda", use_float32=True)

# Configure optimization
config = OptimizationConfig(
    label="a beautiful cat",
    steps=1000,
    lr=2.5,
    use_multi_prompt=True,  # Average multiple prompt variants
)

# Run optimization
optimizer = InverseClipassoOptimizer(clip, config)
result = optimizer.optimize(init_strokes)

print(f"Final similarity: {result.final_similarity:.1%}")
```

### Reference Image Guidance

Use a "perfect" sketch as the optimization target:

```python
config = OptimizationConfig(
    label="fish",
    use_reference_image=True,
    reference_image_embedding=clip.encode_image(perfect_fish_image),
    reference_image_weight=2.0,
)
```

### Photo Exemplars

Guide sketches using real photos for richer semantics:

```python
# Load photos from a directory
photo_embeddings = load_photo_exemplars(
    image_dir="data/photos/fish/",
    clip_model=clip,
)

config = OptimizationConfig(
    label="fish",
    use_exemplar_loss=True,
    exemplar_embeddings=photo_embeddings,
    exemplar_weight=0.5,
)
```

### Refinement Mode

Improve sketches while preserving their original structure:

```python
config = OptimizationConfig(
    label="a perfect fish",
    refinement_mode=True,
    original_weight=0.3,
    original_weight_schedule="cosine_up",  # Start flexible, lock structure later
)
```

### Stroke Addition

Dynamically add strokes during optimization:

```python
config = OptimizationConfig(
    label="a detailed bird",
    allow_stroke_addition=True,
    stroke_add_interval=200,  # Add stroke every 200 steps
    max_strokes=15,
    new_stroke_points=5,
)
```

## Project Structure

```
doodleBot/
├── src/inverse_clipasso/
│   ├── clip/              # CLIP model wrapper
│   │   └── clip_api.py    # ClipModel class
│   ├── render/            # Differentiable rendering
│   │   └── diffvg_wrapper.py
│   ├── optimize/          # Core optimization
│   │   ├── inverse_clipasso.py  # InverseClipassoOptimizer
│   │   └── losses.py      # Loss functions
│   ├── data/              # Data loading
│   └── eval/              # Evaluation metrics
├── tests/
│   └── test_long_optimization.py  # Main experiment script
├── data/
│   ├── quickdraw/raw/strokes/     # QuickDraw NDJSON files
│   └── photos/                     # Reference photos for exemplars
├── outputs/               # Generated outputs
└── scripts/               # Utility scripts
```

## Loss Functions

| Loss | Purpose | Weight |
|------|---------|--------|
| `semantic_loss` | CLIP similarity to target text | 1.0 |
| `reference_image_loss` | Similarity to reference image | 1.0-2.0 |
| `exemplar_loss` | Pull toward exemplar embeddings | 0.5 |
| `original_strokes_loss` | Preserve original structure | 0.3 |
| `stroke_regularizer` | Encourage smooth, simple strokes | 0.05 |
| `total_variation` | Image smoothness | 0.001 |

## Configuration Options

### OptimizationConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `label` | - | Target semantic label |
| `steps` | 200 | Optimization steps |
| `lr` | 1.0 | Learning rate |
| `lr_schedule` | "cosine" | LR schedule: constant, cosine, warmup_cosine |
| `noise_scale` | 0.0 | Gradient noise for exploration |
| `use_multi_prompt` | False | Average multiple text prompts |
| `refinement_mode` | False | Preserve original stroke positions |
| `use_exemplar_loss` | False | Guide toward exemplar embeddings |
| `use_reference_image` | False | Guide toward reference image |
| `allow_stroke_addition` | False | Add strokes during optimization |

## Notes

- **Device Selection**: Automatically uses CUDA if available, falls back to CPU
- **MPS (Apple Silicon)**: Currently disabled due to performance overhead with small tensors
- **Stroke Count**: Not differentiable — use scheduled addition or soft masking

## License

MIT

## Acknowledgments

- [CLIPasso](https://clipasso.github.io/clipasso/) — Inspiration for CLIP-guided sketch optimization
- [OpenAI CLIP](https://github.com/openai/CLIP) — Vision-language model
- [diffvg](https://github.com/BachiLi/diffvg) — Differentiable vector graphics
- [Quick, Draw!](https://quickdraw.withgoogle.com/data) — Sketch dataset
