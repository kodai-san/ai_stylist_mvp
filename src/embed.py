# src/embed.py
import os, io, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import requests
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "raw" / "products.csv"
OUT_VEC = ROOT / "data" / "processed" / "product_vectors.npy"
OUT_META = ROOT / "data" / "processed" / "product_meta.json"

# ---- device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def map_static_to_local(path_or_url: str) -> Path:
    s = str(path_or_url or "")
    if s.startswith("/static/"):
        return ROOT / "data" / "raw" / s[len("/static/"):]
    return ROOT / s.lstrip("\\/")

def fetch_image(path_or_url: str):
    if not path_or_url:
        return None
    try:
        s = str(path_or_url)
        # local (preferred)
        p = map_static_to_local(s)
        if p.is_file():
            return Image.open(p).convert("RGB")
        # explicit path
        p2 = Path(s)
        if p2.is_file():
            return Image.open(p2).convert("RGB")
        # url
        if s.startswith("http://") or s.startswith("https://"):
            r = requests.get(s, timeout=15)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    return None

# --- add near the top (optional, for clarity)
MAX_TOKENS = 77

def build_prompt(row: pd.Series) -> str:
    name = str(row.get("product_name", "") or "")
    desc = str(row.get("description", "") or row.get("desc","") or "")
    element = str(row.get("element_type", "") or "")
    color = str(row.get("color","") or "")
    material = str(row.get("material","") or "")
    tags = str(row.get("style_tags","") or "")
    bits = []
    if element: bits.append(f"{element}")
    if color: bits.append(color)
    if material: bits.append(material)
    if tags: bits.append(tags.replace(",", " "))
    head = " ".join(bits).strip()
    tail = (desc[:180] + "…") if desc and len(desc) > 180 else desc
    prompt = f"{name}. {head}. {tail}".strip()
    return prompt

print("Loading product CSV…", DATA_CSV)
df = pd.read_csv(DATA_CSV, dtype={"product_id":str}).fillna("")
ids, vecs = [], []

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = str(row["product_id"])
        url = row.get("local_image") or row.get("image_url") or ""
        img = fetch_image(url)
        if img is None:
            # print only a few to avoid spam
            if len(ids) < 3:
                print("WARN: missing image for", pid, url)
            continue

        inputs = processor(images=img, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

        prompt = build_prompt(row)
        text_inputs = processor(
        text=[prompt],
        return_tensors="pt",
        padding="max_length",   # pad to max_length
        truncation=True,        # truncate anything longer
        max_length=MAX_TOKENS,  # CLIP text limit
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        text_features = model.get_text_features(**text_inputs)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

        alpha = 0.3  # image-heavy blend
        fused = torch.nn.functional.normalize((1 - alpha) * image_features + alpha * text_features, p=2, dim=-1)

        ids.append(pid)
        vecs.append(fused.squeeze(0).cpu().numpy())

if not vecs:
    print("No vectors created. Check images in CSV (local_image / image_url).")
    raise SystemExit(2)

vecs = np.stack(vecs, axis=0)
OUT_VEC.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT_VEC, vecs)
with open(OUT_META, "w", encoding="utf-8") as f:
    json.dump({"ids": ids}, f, ensure_ascii=False, indent=2)

print(f"Saved vectors: {vecs.shape} → {OUT_VEC}")
print(f"Saved meta → {OUT_META}")