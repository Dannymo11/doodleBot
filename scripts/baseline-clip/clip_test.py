import torch, clip
from PIL import Image
from urllib.request import urlopen

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# just test with a sample image
url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
img = Image.open(urlopen(url)).convert("RGB")
image_input = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
print("Image embedding shape:", image_features.shape)
