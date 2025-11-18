# src/debug_vector.py
import os, io, json, argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import requests, torch, faiss
from transformers import CLIPModel, CLIPProcessor

CSV = "data/raw/products.csv"
VEC = "data/processed/product_vectors.npy"
META = "data/processed/product_meta.json"
IDX = "data/processed/product_index.faiss"

USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; MVPBot/0.3)")
MIN_W = int(os.getenv("MIN_IMG_W", "50"))   # relaxed thresholds for single fix
MIN_H = int(os.getenv("MIN_IMG_H", "50"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "768"))
ALPHA = float(os.getenv("EMBED_BLEND_ALPHA", "0.3"))

def map_local(p: str) -> str:
    if not isinstance(p, str) or not p.strip():
        return ""
    p = p.strip()
    if p.startswith("/static/"):
        p = p.replace("/static/", "data/raw/").lstrip("/")
    return p

def open_local(p: str) -> Optional[Image.Image]:
    if not p: return None
    try:
        im = Image.open(p).convert("RGB")
        if im.width < MIN_W or im.height < MIN_H: return None
        return im
    except Exception:
        return None

def open_http(url: str) -> Optional[Image.Image]:
    if not isinstance(url, str) or not url.strip(): return None
    try:
        r = requests.get(url.strip(), timeout=20, headers={"User-Agent": USER_AGENT, "Accept":"image/*"})
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        if im.width < MIN_W or im.height < MIN_H: return None
        return im
    except Exception:
        return None

def downscale(im: Image.Image) -> Image.Image:
    if MAX_SIDE and max(im.size) > MAX_SIDE:
        im = im.copy()
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    return im

def tags_prompt(tags: str, name: str) -> str:
    tags = ", ".join([t.strip() for t in str(tags or "").split(",") if t.strip()])
    return f"A product that is {tags}. {name}." if tags else f"A product. {name}."

def embed_one(im: Image.Image, prompt: str, device, model, proc) -> np.ndarray:
    im = downscale(im)
    with torch.no_grad():
        x = proc(images=im, return_tensors="pt").to(device)
        img = model.get_image_features(**x)
        img = torch.nn.functional.normalize(img, p=2, dim=-1)
        t = proc(text=[prompt], return_tensors="pt", padding=True).to(device)
        txt = model.get_text_features(**t)
        txt = torch.nn.functional.normalize(txt, p=2, dim=-1)
        fused = torch.nn.functional.normalize((1-ALPHA)*img + ALPHA*txt, p=2, dim=-1)
        return fused.squeeze(0).cpu().numpy().astype("float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="product_id to fix")
    ap.add_argument("--replace", action="store_true", help="replace if exists (otherwise append if missing)")
    args = ap.parse_args()
    pid = str(args.id).strip()

    df = pd.read_csv(CSV, dtype={"product_id": str}).set_index("product_id")
    if pid not in df.index:
        raise SystemExit(f"{pid} not found in CSV")

    row = df.loc[pid]
    name = row.get("product_name", "") or ""
    tags = row.get("style_tags", "") or ""
    local = map_local(row.get("local_image", "") or "")
    url = row.get("image_url", "") or ""

    # Try images
    im = open_local(local) or open_http(url)
    if im is None:
        raise SystemExit(f"Image not usable for {pid}. local={local} url={url}")

    print(f"Image OK: {im.size}  local={bool(local)} url={bool(url)}")

    # Device & model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    vec = embed_one(im, tags_prompt(tags, name), device, model, proc)
    print("New vec dim:", vec.shape[0], "norm≈", float((vec**2).sum()**0.5))

    # Load existing vectors/meta (create if missing)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    if Path(META).exists() and Path(VEC).exists():
        ids = json.load(open(META, "r", encoding="utf-8"))["ids"]
        V = np.load(VEC)
    else:
        ids, V = [], np.empty((0, vec.shape[0]), dtype="float32")

    if pid in ids:
        i = ids.index(pid)
        if args.replace:
            print(f"Replacing existing vector at idx {i}")
            V[i] = vec
        else:
            print(f"{pid} already present. Use --replace to overwrite.")
            # still rebuild index below so FAISS is in sync
    else:
        print("Appending new vector")
        ids.append(pid)
        V = np.vstack([V, vec[None, :]])

    # Persist
    np.save(VEC, V.astype("float32"))
    json.dump({"ids": ids}, open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Rebuild FAISS (FlatIP on normalized vectors)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    Vn = V / np.clip(norms, 1e-9, None)
    index = faiss.IndexFlatIP(Vn.shape[1])
    index.add(Vn.astype("float32"))
    faiss.write_index(index, IDX)
    print(f"Done. ids={len(ids)} vectors={Vn.shape} -> {IDX}")

if __name__ == "__main__":
    main()
