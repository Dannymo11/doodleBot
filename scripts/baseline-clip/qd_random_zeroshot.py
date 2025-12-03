import json, random, numpy as np, torch, clip
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

# ---------- choose your class list ----------
classes = ["cat", "car", "apple", "tree", "house"]  # edit as you like``
classes_descriptive = [f"sketch of a {c}" for c in classes]

# ---------- pick a random sample from ONE label ----------
label = "tree"  # change to any label in `classes` if you want
path = f"data/quickdraw/strokes/{label}.ndjson"

with open(path) as f:
    lines = f.readlines()
sample = json.loads(random.choice(lines))

img = render_strokes(sample["drawing"]).convert("RGB")
img.save(f"sample_{label}_random_zeroshot.png")
print(f"Saved sample_{label}_random_zeroshot.png")

# ---------- CLIP setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

# ---------- image embedding ----------
x = preprocess(img).unsqueeze(0).to(device)
with torch.no_grad():
    img_emb = model.encode_image(x)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # L2 normalize

# ---------- build text embeddings for combined prompts ----------
prompts = [p for pair in zip(classes, classes_descriptive) for p in pair]
with torch.no_grad():
    toks = clip.tokenize(prompts).to(device)            # (2*C,)
    e = model.encode_text(toks)                         # (2*C, D)
    text_embs = e / e.norm(dim=-1, keepdim=True)        # L2 normalize each

# ---------- cosine similarities & prediction ----------
sims = (img_emb @ text_embs.T).squeeze(0)             # (2*C,)
best_idx = sims.argmax().item()
print("\nSimilarities (higher = closer):")
for c, s in sorted(zip(prompts, sims.tolist()), key=lambda x: x[1], reverse=True):
    print(f"{c:>7s}  ->  {s:.3f}")

print(f"\nPredicted: {prompts[best_idx]}")
