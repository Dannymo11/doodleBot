import os, json, random
import numpy as np
import torch, clip
import time
from PIL import Image, ImageDraw
from tqdm import tqdm

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

# ---------- config ----------
classes = ["cat","car","apple","tree","house","dog","bird","flower","fish","chair"]   # change if needed
K = 500                                              # samples per class (adjust)

templates = [
    "{}", "a sketch of a {}", "a line drawing of a {}",
    "a simple doodle of a {}", "an outline of a {}",
]

# ---------- CLIP setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu" and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon acceleration if available

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

# ---------- build one text embedding per class (average prompt variants) ----------
with torch.no_grad():
    per_class_text = []
    for c in classes:
        toks = clip.tokenize([t.format(c) for t in templates]).to(device)
        e = model.encode_text(toks)
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.mean(dim=0, keepdim=True)
        e = e / e.norm(dim=-1, keepdim=True)
        per_class_text.append(e)
    text_embs = torch.vstack(per_class_text)  # (C, D)

# timing
start = time.time()

# ---------- evaluate ----------
correct, total = 0, 0
per_class_correct = {c: 0 for c in classes}
per_class_total = {c: 0 for c in classes}

for ci, c in enumerate(classes):
    path = f"data/quickdraw/strokes/{c}.ndjson"
    with open(path) as f:
        lines = f.readlines()

    n = min(K, len(lines))
    idxs = random.sample(range(len(lines)), k=n)  # random without replacement

    for i in tqdm(idxs, desc=f"{c:>6s}", unit="img"):
        sample = json.loads(lines[i])
        img = render_strokes(sample["drawing"]).convert("RGB")

        with torch.no_grad():
            x = preprocess(img).unsqueeze(0).to(device)
            v = model.encode_image(x)
            v = v / v.norm(dim=-1, keepdim=True)
            pred = (v @ text_embs.T).argmax(dim=1).item()
            
            # Save misclassified images
            if pred != ci:
                os.makedirs(f"errors/{c}", exist_ok=True)
                img.save(f"errors/{c}/{c}_{i}_pred-{classes[pred]}.png")

        # Stats
        is_correct = int(pred == ci)
        correct += is_correct
        total += 1
        per_class_correct[c] += is_correct
        per_class_total[c] += 1

end = time.time()
runtime = end - start
acc = correct / total if total else 0.0
print(f"\nZero-shot Top-1: {acc:.4f}  ({correct}/{total})")
print(f"Runtime: {runtime/60:.2f} minutes ({runtime:.1f} seconds)")

# ---------- per-class accuracy ----------
print("\nPer-class accuracy:")
for c in classes:
    class_acc = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0.0
    print(f"  {c:>6s}: {class_acc:.4f}  ({per_class_correct[c]}/{per_class_total[c]})")
