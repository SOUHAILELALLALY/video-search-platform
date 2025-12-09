import os
import io
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# initialize CLIP once; CPU/GPU will be used automatically by torch
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None

def init_clip(model_name: str = "openai/clip-vit-base-patch32"):
    global _model, _processor
    if _model is None or _processor is None:
        _model = CLIPModel.from_pretrained(model_name).to(_device)
        _processor = CLIPProcessor.from_pretrained(model_name)
    return _model, _processor


def embed_image(img: Image.Image):
    """Return a normalized numpy embedding for a PIL image."""
    model, processor = init_clip()
    inputs = processor(images=img, return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy().astype(np.float32)[0]


def embed_text(text: str):
    model, processor = init_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)[0]


def load_image_from_path(path: str):
    img = Image.open(path).convert("RGB")
    return img


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
