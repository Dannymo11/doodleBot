# inverse-clipasso

Evaluating CLIP's zero-shot classification performance on QuickDraw sketches, with infrastructure for inverse CLIPasso (sketch refinement toward semantic targets).

## Overview

This codebase evaluates how well CLIP (Contrastive Language-Image Pre-training) can classify hand-drawn sketches from the QuickDraw dataset. It includes:

- **Zero-shot classification evaluation**: Test CLIP's ability to classify QuickDraw sketches without fine-tuning
- **Prompt engineering experiments**: Compare different text prompts (e.g., "car" vs "sketch of a car") for better classification
- **Per-class accuracy analysis**: Detailed breakdown of classification performance across different object categories
- **Error analysis**: Automatic saving of misclassified images for visual inspection

## Project Structure

```
inverse-clipasso/
├── data/
│   └── quickdraw/
│       └── strokes/          # QuickDraw NDJSON files (one per class)
├── src/                       # Core library code
│   ├── clip/                  # CLIP model integration
│   ├── data/                  # Data loading utilities
│   ├── eval/                  # Evaluation metrics
│   ├── optimize/              # Inverse CLIPasso optimization (in development)
│   ├── render/                # Rendering utilities
│   └── utils/                 # General utilities
├── experiments/
│   └── configs/               # YAML configuration files
├── notebooks/                 # Exploratory research notebooks
├── outputs/                   # Generated outputs (PNGs, SVGs, metrics)
├── errors/                    # Misclassified images (auto-generated)
└── *.py                       # Main evaluation scripts
```

## Main Scripts

### Evaluation Scripts

- **`qd_zeroshot_eval.py`**: Batch evaluation script that:
  - Tests CLIP on multiple samples per class (default: 500)
  - Computes overall and per-class accuracy
  - Saves misclassified images to `errors/` directory
  - Supports multiple classes: cat, car, apple, tree, house, dog, bird, flower, fish, chair
  - Uses prompt ensemble averaging for better classification

- **`qd_zeroshot_one.py`**: Single sample evaluation:
  - Randomly selects one sketch from a random class
  - Shows prediction and similarity scores for all classes
  - Useful for quick testing and debugging

- **`qd_random_zeroshot.py`**: Interactive single sample testing:
  - Tests a specific class (configurable)
  - Compares plain class names vs "sketch of [class]" prompts
  - Displays detailed similarity scores

- **`qd_random_sample.py`**: Simple rendering test:
  - Renders a random QuickDraw sample to PNG
  - Basic CLIP encoding test

- **`qd_clip_one.py`**: Minimal CLIP encoding example:
  - Loads and encodes one QuickDraw sample
  - Basic functionality demonstration

## Dependencies

### Python Version
- Python 3.10

### Core Dependencies

**Deep Learning:**
- `torch` - PyTorch for tensor operations
- `torchvision` - Computer vision utilities
- `clip-anytorch` - CLIP model implementation

**Data Processing:**
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `pandas` - Data manipulation
- `scikit-image` - Image processing

**Visualization & Rendering:**
- `matplotlib` - Plotting and visualization
- `PIL` (Pillow) - Image processing (included with Python)
- `svgwrite` - SVG generation
- `diffvg` - Differentiable vector graphics rendering

**Utilities:**
- `tqdm` - Progress bars
- `hydra-core` - Configuration management
- `wandb` - Experiment tracking (optional)

### Installation

#### Option 1: Conda (Recommended)
```bash
conda env create -f env.yml
conda activate inverse-clipasso
```

#### Option 2: pip
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note**: For Apple Silicon (M1/M2), the code automatically uses MPS acceleration if available. For CUDA support, ensure you have the appropriate CUDA toolkit installed.

## Quick Start

1. **Set up the environment** (see Installation above)

2. **Prepare QuickDraw data**:
   - Place QuickDraw NDJSON files in `data/quickdraw/strokes/`
   - Each file should be named `{class_name}.ndjson` (e.g., `cat.ndjson`, `car.ndjson`)

3. **Run evaluation**:
   ```bash
   python qd_zeroshot_eval.py
   ```
   This will:
   - Evaluate 500 samples per class
   - Print overall and per-class accuracy
   - Save misclassified images to `errors/{class_name}/`

4. **Test single samples**:
   ```bash
   python qd_zeroshot_one.py      # Random class, random sample
   python qd_random_zeroshot.py   # Specific class (edit label variable)
   ```

## Configuration

### Classes
Edit the `classes` list in the evaluation scripts to test different object categories:
```python
classes = ["cat", "car", "apple", "tree", "house", "dog", "bird", "flower", "fish", "chair"]
```

### Samples per Class
In `qd_zeroshot_eval.py`, adjust `K` to change the number of samples evaluated per class:
```python
K = 500  # samples per class
```

### Prompt Templates
Modify the `templates` list to experiment with different text prompts:
```python
templates = [
    "{}",                      # Plain class name
    "a sketch of a {}",         # Descriptive
    "a line drawing of a {}",  # Alternative description
    "a simple doodle of a {}", # Casual description
    "an outline of a {}",      # Minimal description
]
```

## Output

- **Console output**: Overall accuracy, per-class accuracy, runtime statistics
- **Error images**: Misclassified samples saved to `errors/{class_name}/{class}_{index}_pred-{predicted_class}.png`
- **Sample images**: Test renders saved as `sample_{class}_random*.png`

## Notes

- The `src/` directory contains infrastructure for inverse CLIPasso optimization (sketch refinement), which is currently in development
- Generated outputs should stay inside `outputs/` or `errors/` directories to keep the repo clean
- The code automatically handles device selection (CUDA, MPS, or CPU)
