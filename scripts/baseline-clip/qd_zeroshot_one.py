import json, random, numpy as np, torch, clip
from PIL import Image, ImageDraw

# ---- render (your version) ----
def render_strokes(drawing, size=256, stroke_width=3, padding=10):
    xs = np.concatenate([np.array(s[0]) for s in drawing])
    ys = np.concatenate([np.array(s[1]) for s in drawing])
    min_x, max_x = xs.min(), xs.max(); min_y, max_y = ys.min(), ys.max()
    w, h = max_x - min_x + 1e-6, max_y - min_y + 1e-6
    scale = (size - 2*padding) / max(w, h); ox = (size - scale*w)/2; oy = (size - scale*h)/2
    img = Image.new("RGB", (size, size), "white"); draw = ImageDraw.Draw(img)
    for s in drawing:
        x, y = np.array(s[0]), np.array(s[1])
        X, Y = (x - min_x) * scale + ox, (y - min_y) * scale + oy
        pts = list(zip(X, Y))
        if len(pts) >= 2: draw.line(pts, fill="black", width=3, joint="curve")
    return img

# ---- config ----
classes = ["cat", "car", "apple", "tree", "house"]
templates = [
    "{}", "a sketch of a {}", "a line drawing of a {}",
    "a simple doodle of a {}", "an outline of a {}"
]

# ---- pick random sample from a random class ----
label = random.choice(classes)
path = f"data/quickdraw/strokes/{label}.ndjson"
with open(path) as f:
    sample = json.loads(random.choice(f.readlines()))
img = render_strokes(sample["drawing"]).convert("RGB")
img.save(f"sample_{label}_random.png")

# ---- CLIP ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False); model.eval()

# image emb
with torch.no_grad():
    x = preprocess(img).unsqueeze(0).to(device)
    img_emb = model.encode_image(x); img_emb /= img_emb.norm(dim=-1, keepdim=True)

# text embs (average variants per class)
with torch.no_grad():
    per_class = []
    for c in classes:
        toks = clip.tokenize([t.format(c) for t in templates]).to(device)
        e = model.encode_text(toks); e /= e.norm(dim=-1, keepdim=True)
        e = e.mean(dim=0, keepdim=True); e /= e.norm(dim=-1, keepdim=True)
        per_class.append(e)
    text_embs = torch.vstack(per_class)  # (C, D)

# predict
sims = (img_emb @ text_embs.T).squeeze(0)  # (C,)
pred_idx = sims.argmax().item()
print("True:", label, "| Pred:", classes[pred_idx])
print("Scores:")
for c, s in sorted(zip(classes, sims.tolist()), key=lambda x: x[1], reverse=True):
    print(f"{c:>7s} -> {s:.3f}")
