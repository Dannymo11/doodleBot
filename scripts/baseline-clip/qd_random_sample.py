import json
import random
import torch
import clip
import numpy as np
from PIL import Image, ImageDraw

# ---------- render function ----------
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

# ---------- pick random sample ----------
label = "apple"  # change this to any of our labels
path = f"data/quickdraw/strokes/{label}.ndjson"

with open(path) as f:
    lines = f.readlines()
sample = json.loads(random.choice(lines))

img = render_strokes(sample["drawing"]).convert("RGB")
img.save(f"sample_{label}_random.png")
print(f"Saved sample_{label}_random.png")

# ---------- CLIP encoding ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

x = preprocess(img).unsqueeze(0).to(device)
with torch.no_grad():
    emb = model.encode_image(x)
    emb = emb / emb.norm(dim=-1, keepdim=True)

print("Embedding shape:", emb.shape)
