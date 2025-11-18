# src/fix_missing_vectors.py
import os, io, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import requests, torch, faiss
from transformers import CLIPModel, CLIPProcessor

CSV = "data/raw/products.csv"
VEC = "data/processed/product_vectors.npy"
META = "data/processed/product_meta.json"
IDX = "data/processed/product_index.faiss"

def load_df():
    return pd.read_csv(CSV, dtype={"product_id": str}).set_index("product_id")

def open_local(path: str):
    if not path: return None
    if path.startswith("/static/"):
        path = path.replace("/static/", "data/raw/")
    try:
        im = Image.open(path).convert("RGB")
        if im.width < 100 or im.height < 100:  # avoid tiny placeholders
            return None
        return im
    except Exception:
        return None

def open_http(url: str):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        if im.width < 100 or im.height < 100:
            return None
        return im
    except Exception:
        return None

def embed_missing():
    df = load_df()

    ids = []
    vecs = None
    if Path(META).exists() and Path(VEC).exists():
        ids = json.load(open(META, "r", encoding="utf-8"))["ids"]
        vecs = np.load(VEC)
    else:
        ids, vecs = [], np.empty((0,512), dtype="float32")

    have = set(ids)
    want = set(df.index.astype(str))
    missing = [pid for pid in want - have]
    if not missing:
        print("No missing IDs. Nothing to do.")
        return

    print(f"Embedding {len(missing)} missing items…")

    # Device & model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    new_ids = []
    new_vecs = []

    with torch.no_grad():
        for pid in missing:
            row = df.loc[pid]
            img = open_local(str(row.get("local_image","")))
            if img is None:
                img = open_http(str(row.get("image_url","")))
            if img is None:
                continue

            inputs = proc(images=img, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**inputs)
            img_feat = torch.nn.functional.normalize(img_feat, p=2, dim=-1)

            tags = str(row.get("style_tags","")).strip()
            name = str(row.get("product_name",""))
            prompt = f"A product that is {tags}. {name}." if tags else f"A product. {name}."
            tokens = proc(text=[prompt], return_tensors="pt", padding=True).to(device)
            txt_feat = model.get_text_features(**tokens)
            txt_feat = torch.nn.functional.normalize(txt_feat, p=2, dim=-1)

            alpha = 0.3
            fused = torch.nn.functional.normalize((1-alpha)*img_feat + alpha*txt_feat, p=2, dim=-1)
            new_ids.append(pid)
            new_vecs.append(fused.squeeze(0).cpu().numpy().astype("float32"))

    if not new_ids:
        print("No new vectors created (images may be missing).")
        return

    # Append and persist
    vecs = np.vstack([vecs, np.stack(new_vecs, axis=0)])
    ids.extend(new_ids)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    np.save(VEC, vecs)
    json.dump({"ids": ids}, open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Rebuild FAISS (FlatIP)
    row_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(row_norms, 1e-9, None)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype("float32"))
    faiss.write_index(index, IDX)

    print(f"Done. ids={len(ids)} vectors={vecs.shape} -> {IDX}")

if __name__ == "__main__":
    embed_missing()
