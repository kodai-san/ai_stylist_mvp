# src/infer_types_quick.py
import re, io
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import requests, torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "data" / "raw" / "products.csv"

TYPE_LABELS = [
    "sofa","armchair","chair","stool",
    "bed","nightstand","bedside table",
    "table","coffee table","desk",
    "lamp","floor lamp","table lamp",
    "rug","cabinet","dresser","drawer","shelf","bookcase","storage",
]
CANON = {"armchair":"chair","stool":"chair","bedside table":"nightstand",
         "coffee table":"table","drawer":"storage","bookcase":"shelf"}

KEYWORD_RULES = [
    (r"\bsofa|couch|loveseat\b","sofa"),
    (r"\barmchair\b","armchair"),
    (r"\bchair\b","chair"),
    (r"\bstool\b","stool"),
    (r"\bbed\b","bed"),
    (r"\bnightstand|bedside\b","nightstand"),
    (r"\bdesk\b","desk"),
    (r"\bcoffee table\b","coffee table"),
    (r"\btable\b","table"),
    (r"\blamp|light\b","lamp"),
    (r"\brug\b","rug"),
    (r"\bdresser|drawer|cabinet|chest\b","storage"),
    (r"\bshelf|bookcase|shelving\b","shelf"),
]

def unify(label: str) -> str:
    lab = (label or "").strip().lower()
    return CANON.get(lab, lab) or ""

def kw_guess(text: str) -> str:
    t = (text or "").lower()
    for pat, lab in KEYWORD_RULES:
        if re.search(pat, t):
            return lab
    return ""

def fetch_image(path_or_url: str):
    if not path_or_url:
        return None
    try:
        if path_or_url.startswith("/static/"):
            p = ROOT / path_or_url.lstrip("/")
            return Image.open(p).convert("RGB")
        if path_or_url.startswith("http"):
            r = requests.get(path_or_url, timeout=15); r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        return Image.open(ROOT / path_or_url).convert("RGB")
    except Exception:
        return None

def clip_guess(pil, model, proc, device):
    if pil is None:
        return "", 0.0, 0.0
    prompts = [f"a photo of a {lab}" for lab in TYPE_LABELS]
    with torch.no_grad():
        inputs = proc(text=prompts, images=pil, return_tensors="pt", padding=True).to(device)
        logits = model(**inputs).logits_per_image.squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    i_top = int(np.argmax(probs))
    conf = float(probs[i_top])
    second = float(np.partition(probs, -2)[-2]) if probs.size >= 2 else 0.0
    return TYPE_LABELS[i_top], conf, conf - second

def main():
    df = pd.read_csv(CSV, dtype={"product_id": str})
    for col in ["category_auto","category_conf","category_margin"]:
        if col not in df.columns: df[col] = ""

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()
              else torch.device("cpu"))
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = f"{row.get('product_name','')} {row.get('description','')}"
        label = unify(kw_guess(text))
        conf, margin = 1.0, 1.0
        if not label:
            pil = fetch_image(str(row.get("local_image","") or row.get("image_url","")))
            lab2, conf, margin = clip_guess(pil, model, proc, device)
            label = unify(lab2)
        df.at[i, "category_auto"] = label
        df.at[i, "category_conf"] = conf
        df.at[i, "category_margin"] = margin

    df.to_csv(CSV, index=False)
    print("Wrote inferred types to:", CSV)

if __name__ == "__main__":
    main()