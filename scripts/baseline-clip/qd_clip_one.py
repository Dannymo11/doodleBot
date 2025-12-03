import json, torch, clip
import numpy as np
from PIL import Image, ImageDraw

# Render one QuickDraw sketch to an image (NumPy version)
def render_strokes(drawing, size=256, stroke_width=3, padding=10):
    # Flatten all x and y coordinates across strokes
    xs = np.concatenate([np.array(s[0]) for s in drawing])
    ys = np.concatenate([np.array(s[1]) for s in drawing])

    # Find bounding box
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    w = max_x - min_x + 1e-6
    h = max_y - min_y + 1e-6

    # Scale so drawing fits within (size - 2*padding)
    scale = (size - 2 * padding) / max(w, h)
    ox = (size - scale * w) / 2
    oy = (size - scale * h) / 2

    # Create white canvas
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # Draw each stroke (transforming coordinates with NumPy math)
    for s in drawing:
        x = np.array(s[0])
        y = np.array(s[1])
        X = (x - min_x) * scale + ox
        Y = (y - min_y) * scale + oy
        pts = list(zip(X, Y))
        if len(pts) >= 2:
            draw.line(pts, fill="black", width=stroke_width, joint="curve")

    return img


# ---- CLIP setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

# ---- Load one QuickDraw sample ----
path = "data/quickdraw/strokes/cat.ndjson"  # change label if you want
with open(path) as f:
    first = json.loads(next(f))

img = render_strokes(first["drawing"]).convert("RGB")
img.save("sample_cat.png")  # visual check

# ---- Encode with CLIP ----
x = preprocess(img).unsqueeze(0).to(device)
with torch.no_grad():
    emb = model.encode_image(x)
    emb = emb / emb.norm(dim=-1, keepdim=True)

print("Embedding shape:", emb.shape)
