import clip
import torch
from PIL import Image
from io import BytesIO
from typing import Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def classify_with_clip(image_bytes: bytes, keyword: str, threshold: float = 50.0) -> Tuple[bool, float]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([keyword]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).item() * 100
    return similarity > threshold, round(similarity, 2)
