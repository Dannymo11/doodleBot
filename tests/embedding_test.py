import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import json, random, numpy as np, torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from src.inverse_clipasso.clip.clip_api import ClipModel
from src.inverse_clipasso.eval.metrics import clip_cosine

# ---- render (same as before) ----
def render_strokes(drawing, size=256, stroke_width=3, padding=10):
    xs = np.concatenate([np.array(s[0]) for s in drawing])
    ys = np.concatenate([np.array(s[1]) for s in drawing])
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    w, h = max_x - min_x + 1e-6, max_y - min_y + 1e-6
    scale = (size - 2 * padding) / max(w, h)
    ox = (size - scale * w) / 2
    oy = (size - scale * h) / 2

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    for s in drawing:
        x, y = np.array(s[0]), np.array(s[1])
        X, Y = (x - min_x) * scale + ox, (y - min_y) * scale + oy
        pts = list(zip(X, Y))
        if len(pts) >= 2:
            draw.line(pts, fill="black", width=stroke_width, joint="curve")
    return img

# ---- config ----
classes = ["cat", "car", "apple", "tree", "house"]
templates = [
    "{}", "a sketch of a {}", "a line drawing of a {}",
    "a simple doodle of a {}", "an outline of a {}"
]

def stroke_to_parameter(x_coords, y_coords, min_x, min_y, scale, ox, oy, device="cuda"):
    """
    Convert a single stroke's coordinates to a differentiable [N_i, 2] tensor parameter.
    
    Args:
        x_coords: numpy array of x coordinates [N]
        y_coords: numpy array of y coordinates [N]
        min_x, min_y: bounding box minimums for translation
        scale: scaling factor for normalization
        ox, oy: offset for centering
        device: device to create tensor on
    
    Returns:
        torch.nn.Parameter of shape [N_i, 2] with requires_grad=True
    """
    # Transform to canvas coordinates
    X = (x_coords - min_x) * scale + ox
    Y = (y_coords - min_y) * scale + oy
    
    # Stack into [N_i, 2] shape
    pts = np.stack([X, Y], axis=-1)  # [N_i, 2]
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
    
    # Make learnable parameter
    return torch.nn.Parameter(pts_t, requires_grad=True)


def drawing_to_param_strokes(drawing, size=256, padding=10, device="cuda"):
    """
    Convert a QuickDraw 'drawing' (list of strokes with [x, y, t]) into
    a differentiable set of stroke parameters.

    Each stroke in `drawing` looks like:
        stroke = [x_list, y_list, t_list]

    We:
      1. Compute a global bounding box over all x,y.
      2. Scale + center the whole drawing into a square [0, size] canvas.
      3. For each stroke, create a torch.nn.Parameter of shape [N_i, 2].

    Returns:
        stroke_params: list[torch.nn.Parameter], each with shape [N_i, 2].
    """

    # ---- 1. Flatten all x,y values across all strokes to compute bbox ----
    xs_all = []
    ys_all = []
    for stroke in drawing:
        x_coords = np.array(stroke[0], dtype=np.float32)  # [N]
        y_coords = np.array(stroke[1], dtype=np.float32)  # [N]
        xs_all.append(x_coords)
        ys_all.append(y_coords)

    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    w = max_x - min_x + 1e-6
    h = max_y - min_y + 1e-6

    # ---- 2. Scale + center into a size x size canvas, with padding ----
    scale = (size - 2 * padding) / max(w, h)
    ox = (size - scale * w) / 2.0
    oy = (size - scale * h) / 2.0

    # ---- 3. Convert each stroke to a [N_i, 2] differentiable tensor ----
    stroke_params = []
    for stroke in drawing:
        x_coords = np.array(stroke[0], dtype=np.float32)  # ignore t_coords for now
        y_coords = np.array(stroke[1], dtype=np.float32)
        
        param = stroke_to_parameter(x_coords, y_coords, min_x, min_y, scale, ox, oy, device)
        stroke_params.append(param)

    return stroke_params


def embed_label_with_templates(clip_model: ClipModel, label: str, templates) -> torch.Tensor:
    """
    Use the same idea as your old script:
    multiple templates per class -> average their CLIP text embeddings.
    Returns a single [1, D] embedding for the label.
    """
    embs = []
    for t in templates:
        prompt = t.format(label)
        emb = clip_model.encode_text(prompt)  # [1, D]
        embs.append(emb)
    embs = torch.vstack(embs)                # [T, D]
    embs = embs / embs.norm(dim=-1, keepdim=True)
    avg = embs.mean(dim=0, keepdim=True)     # [1, D]
    avg = avg / avg.norm(dim=-1, keepdim=True)
    return avg

def main():
    # ---- pick random sample from a random class ----
    label = random.choice(classes)
    path = os.path.join(
        PROJECT_ROOT,
        "data", "quickdraw", "raw", "strokes",
        f"{label}.ndjson"
    )
    with open(path) as f:
        sample = json.loads(random.choice(f.readlines()))
    drawing = sample["drawing"]   # this is the [x, y, t] stroke list

    # ---- device + differentiable stroke params ----
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stroke_params = drawing_to_param_strokes(
        drawing,
        size=256,
        padding=10,
        device=device,
    )

    print(f"\nCreated differentiable strokes for label='{label}':")
    print(f"  Number of strokes: {len(stroke_params)}")
    if len(stroke_params) > 0:
        print(f"  First stroke shape: {stroke_params[0].shape}")  # e.g. torch.Size([N, 2])

    # ---- raster render (same as before, for CLIP test) ----
    img = render_strokes(drawing).convert("RGB")
    img.save(f"sample_{label}_random.png")
    
    # Display the image
    print(f"\nTesting with QuickDraw sample: {label}")
    print("=" * 60)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"QuickDraw Sample: {label}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)  # non-blocking so script continues
    plt.pause(0.1)  # brief pause to ensure window opens

    # ---- CLIP via ClipModel ----
    clip_model = ClipModel(device=device)

    # image emb (use clip_model.preprocess)
    x = clip_model.preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]
    img_emb = clip_model.encode_image(x)                    # [1, D]

    # text embs (average variants per class, using ClipModel.encode_text)
    per_class = []
    for c in classes:
        e = embed_label_with_templates(clip_model, c, templates)  # [1, D]
        per_class.append(e)
    text_embs = torch.vstack(per_class)  # [C, D]

    # Calculate cosine similarities using clip_cosine from metrics.py
    # Expand img_emb to match text_embs shape for batch computation
    img_emb_expanded = img_emb.expand(text_embs.shape[0], -1)  # [C, D]
    sims = clip_cosine(img_emb_expanded, text_embs).squeeze()  # [C]
    
    # predict
    pred_idx = sims.argmax().item()
    
    # Calculate semantic loss: 1 - cosine_similarity
    semantic_losses = 1.0 - sims  # [C]
    true_label_idx = classes.index(label)
    true_loss = semantic_losses[true_label_idx].item()
    pred_loss = semantic_losses[pred_idx].item()

    print("\n" + "=" * 60)
    print("CLIP Prediction Results")
    print("=" * 60)
    print(f"True label:  {label}")
    print(f"Predicted:   {classes[pred_idx]}")
    print(f"Correct:     {'✓' if label == classes[pred_idx] else '✗'}")
    print(f"\nSemantic Loss (1 - cosine_similarity):")
    print(f"  True label '{label}':  {true_loss:.4f}")
    print(f"  Predicted '{classes[pred_idx]}':  {pred_loss:.4f}")
    print("\nSimilarity scores & Semantic Loss (lower loss = better match):")
    for i, (c, s, loss) in enumerate(
        sorted(
            zip(classes, sims.tolist(), semantic_losses.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    ):
        marker = "←" if c == classes[pred_idx] else " "
        true_marker = "★" if c == label else " "
        print(f"  {c:>7s} -> similarity: {s:.4f} | loss: {loss:.4f} {marker}{true_marker}")
    print("=" * 60)
    
    # Keep the image window open until user closes it
    print("\nClose the image window to exit...")
    plt.show(block=True)


if __name__ == "__main__":
    main()
