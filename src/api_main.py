from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict
import json
import logging
import re

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel

# Paths
ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_CSV = ROOT / "data" / "raw" / "products.csv"
VEC_PATH     = ROOT / "data" / "processed" / "product_vectors.npy"
META_PATH    = ROOT / "data" / "processed" / "product_meta.json"
INDEX_PATH   = ROOT / "data" / "processed" / "product_index.faiss"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

app = FastAPI(title="AI Stylist MVP")

# CORS (dev-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (serve cached /static/images/*)
app.mount("/static", StaticFiles(directory=str(ROOT / "data" / "raw")), name="static")

# Public UI (served at /app)
public_dir = ROOT / "public"
if public_dir.exists():
    app.mount("/app", StaticFiles(directory=str(public_dir), html=True), name="app")

# --------------------------------------------------------------------------------------
# Data & vector index
# --------------------------------------------------------------------------------------
def _s(x) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def _price_to_float(s: str) -> Optional[float]:
    s = _s(s)
    if not s:
        return None
    m = re.search(r"[\d]+(?:[.,]\d+)?", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def _tag_set(s: str) -> set[str]:
    s = _s(s).lower()
    if not s:
        return set()
    return {t.strip() for t in s.replace(";", ",").split(",") if t.strip()}

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

COMPLEMENT_MAP: Dict[str, List[str]] = {
    "table":   ["chair", "lamp", "rug", "storage"],
    "desk":    ["chair", "lamp", "storage"],
    "sofa":    ["side_table", "lamp", "rug", "storage", "chair"],
    "bed":     ["nightstand", "lamp", "bedding", "storage", "rug"],
    "chair":   ["table", "desk", "lamp", "rug"],
    "storage": ["lamp", "chair", "table", "rug"],
    "lamp":    ["table", "sofa", "bed", "desk", "chair"],
    "rug":     ["table", "sofa", "chair", "bed"],
}

def _rule_bonus(base_cat: str, cand_cat: str) -> float:
    base_cat = _s(base_cat).lower()
    cand_cat = _s(cand_cat).lower()
    allowed = COMPLEMENT_MAP.get(base_cat, [])
    return 1.0 if cand_cat in allowed else 0.0

# Load CSV
try:
    prod_df = pd.read_csv(PRODUCTS_CSV, dtype={"product_id": str}).fillna("")
    prod_df = prod_df.set_index("product_id")
except Exception as e:
    logging.error(f"Failed to load CSV: {e}")
    raise

# Load vectors + FAISS
try:
    vecs = np.load(str(VEC_PATH)).astype("float32")
    with open(META_PATH, "r", encoding="utf-8") as f:
        ids = json.load(f)["ids"]
    index = faiss.read_index(str(INDEX_PATH))
except Exception as e:
    logging.error(f"Failed to load vectors/index: {e}")
    raise

if len(ids) != vecs.shape[0] or index.ntotal != vecs.shape[0]:
    logging.error(
        f"Mismatch: ids={len(ids)} vecs={vecs.shape} index={index.ntotal}. "
        "Rebuild with: python -m src.embed && python -m src.build_index"
    )
id_to_row = {str(pid): i for i, pid in enumerate(ids)}

def category_of(pid: str) -> str:
    try:
        return _s(prod_df.loc[pid, "category"]).lower()
    except Exception:
        return "unknown"

class Product(BaseModel):
    product_id: str
    product_name: str
    image_url: str
    category: str
    price: str
    page_url: str
    style_tags: str
    description: Optional[str] = ""

class RecResponse(BaseModel):
    base: Product
    recommendations: List[Product]

# --------------------------------------------------------------------------------------
# CLIP text (lazy)
# --------------------------------------------------------------------------------------
_clip_model = None
_clip_proc = None
_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

def ensure_clip():
    global _clip_model, _clip_proc
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def _text_vec(prompt: str) -> np.ndarray:
    ensure_clip()
    with torch.no_grad():
        toks = _clip_proc(
            text=[prompt],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).to(_device)
        feats = _clip_model.get_text_features(**toks)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats.cpu().numpy().astype("float32")

def _build_complement_prompt(pid: str) -> str:
    if pid not in prod_df.index:
        return "items that go well together"
    row = prod_df.loc[pid]
    cat  = _s(row.get("category", ""))
    tags = _s(row.get("style_tags", "")).replace(",", " ")
    desc = _s(row.get("description", ""))[:120]
    bits = []
    if cat: bits.append(cat)
    if tags: bits.append(tags)
    head = " ".join(bits).strip()
    if head:
        return f"items that go well with a {head} for a cohesive room. {desc}"
    return f"items that go well together. {desc}"

# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "items": int(len(prod_df))}

@app.get("/debug/status")
def debug_status(n: int = 10):
    return {
        "csv_rows": int(len(prod_df)),
        "vectors": int(len(ids)),
        "index_size": int(index.ntotal),
        "sample_ids": [str(x) for x in ids[:n]],
        "paths": {
            "csv": str(PRODUCTS_CSV),
            "vec": str(VEC_PATH),
            "meta": str(META_PATH),
            "faiss": str(INDEX_PATH),
        },
    }

@app.get("/debug/has_vector/{product_id}")
def debug_has_vector(product_id: str):
    return {"product_id": product_id, "in_vectors": product_id in id_to_row}

@app.get("/debug/row/{product_id}")
def debug_row(product_id: str):
    if product_id not in prod_df.index:
        return {"found": False}
    row = prod_df.loc[product_id].to_dict()
    return {"found": True, **{k: _s(v) for k, v in row.items()}}

@app.get("/search")
def search(q: str, k: int = 8):
    """Text → product search via CLIP text embedding."""
    tvec = _text_vec(q)  # (1, d)
    D, I = index.search(tvec.astype("float32"), max(1, k))
    hits = [ids[i] for i in I[0] if i >= 0]
    results = []
    for pid in hits:
        row = prod_df.loc[pid]
        results.append({
            "product_id": pid,
            "product_name": _s(row.get("product_name", "")),
            "image_url": _s(row.get("local_image", "")) or _s(row.get("image_url", "")),
            "category": _s(row.get("category", "")),
            "price": _s(row.get("price", "")),
            "page_url": _s(row.get("page_url", "")),
            "description": _s(row.get("description", "")),
        })
    return {"query": q, "results": results}

@app.get("/recommendations/{product_id}", response_model=RecResponse)
def recommendations(
    product_id: str,
    k: int = 2,
    exclude_same_category: int = 1,
    beta: float = 0.55,  # blend: 0=image-only, 1=text-only
    strict_allow: int = 0,
):
    if len(ids) != vecs.shape[0] or index.ntotal != vecs.shape[0]:
        raise HTTPException(status_code=500, detail="Vectors/index out of sync. Rebuild embeddings & index.")

    product_id = _s(product_id).strip()
    if product_id not in id_to_row:
        raise HTTPException(status_code=404, detail="product_id not found in vectors")

    base_idx = id_to_row[product_id]
    base_vec = vecs[base_idx:base_idx+1]

    prompt = _build_complement_prompt(product_id)
    tvec = _text_vec(prompt)

    q = (1.0 - beta) * base_vec + beta * tvec
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-9, None)

    over = min(max(k + 50, 64), len(ids))
    D, I = index.search(q.astype("float32"), over)

    base_row = prod_df.loc[product_id]
    base_cat = _s(base_row.get("category", "")).lower()
    base_tags = _tag_set(base_row.get("style_tags", ""))
    base_price = _price_to_float(base_row.get("price", ""))

    allowed_set = set(COMPLEMENT_MAP.get(base_cat, []))

    candidates: list[tuple[str, float]] = []
    seen = {product_id}
    for idx, sim in zip(I[0], D[0]):
        if idx < 0:
            continue
        pid = ids[idx]
        if pid in seen:
            continue
        seen.add(pid)
        if exclude_same_category and base_cat and category_of(pid) == base_cat:
            continue
        if strict_allow and allowed_set and category_of(pid) not in allowed_set:
            continue
        candidates.append((pid, float(sim)))

    def score(pid: str, base_sim: float) -> float:
        row = prod_df.loc[pid]
        cand_cat = _s(row.get("category", "")).lower()
        cand_tags = _tag_set(row.get("style_tags", ""))

        j = _jaccard(base_tags, cand_tags)
        rb = _rule_bonus(base_cat, cand_cat)

        s_price = 0.0
        bp = base_price
        cp = _price_to_float(row.get("price", ""))
        if bp and cp and bp > 0 and cp > 0:
            lo = min(bp, cp); hi = max(bp, cp)
            s_price = lo / hi

        return 0.60 * base_sim + 0.25 * j + 0.10 * s_price + 0.15 * rb

    scored = [(pid, score(pid, sim)) for pid, sim in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    topk = [pid for pid, _ in scored[:k]]

    def to_product(pid: str) -> Product:
        row = prod_df.loc[pid]
        img = _s(row.get("local_image", "")) or _s(row.get("image_url", ""))
        return Product(
            product_id=_s(pid),
            product_name=_s(row.get("product_name", "")),
            image_url=_s(img),
            category=_s(row.get("category", "")),
            price=_s(row.get("price", "")),
            page_url=_s(row.get("page_url", "")),
            style_tags=_s(row.get("style_tags", "")),
            description=_s(row.get("description", "")),
        )

    base_prod = to_product(product_id)
    recs = [to_product(pid) for pid in topk]
    return RecResponse(base=base_prod, recommendations=recs)

# --- Compose endpoints ---
from src.compose import (
    compose_pair,
    compose_adjacent_pair,
    HAS_REMBG,
    REMBG_ERROR,
)

@app.get("/compose/adjacent/{base_id}/{rec_id}")
def compose_adjacent_endpoint(
    base_id: str,
    rec_id: str,
    cutout: int = 1,
    bg: str = "white",
    order: str = "base_left",       # "base_left" | "rec_left"
    floor: float = 0.82,
    h: int = 560,
    gap: int = 28,
    pad: int = 6,
    auto: int = 0,
    auto_min: float = 0.88,
    auto_max: float = 0.98,
):
    try:
        buf = compose_adjacent_pair(
            base_id, rec_id,
            cutout=bool(cutout),
            bg_kind=bg,
            floor_ratio=float(floor),
            target_h=int(h),
            min_gap_px=int(gap),
            avoid_pad_px=int(pad),
            auto_perspective=bool(auto),
            auto_strength_min=float(auto_min),
            auto_strength_max=float(auto_max),
            order=order,
        )
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# (Kept) simple card-style side-by-side
@app.get("/compose/{base_id}/{rec_id}")
def compose_cards_endpoint(
    base_id: str, rec_id: str,
    cutout: int = 0,
    bg: str = "white",
    auto: int = 0,
    auto_min: float = 0.88,
    auto_max: float = 0.98,
):
    try:
        img = compose_pair(
            base_id, rec_id,
            cutout=bool(cutout),
            bg_kind=bg,
            auto_perspective=bool(auto),
            auto_strength_min=float(auto_min),
            auto_strength_max=float(auto_max),
        )
        return StreamingResponse(img, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/debug/rembg_status")
def rembg_status():
    return {"has_rembg": bool(HAS_REMBG), "error": REMBG_ERROR}

# Mount an extra UI alias at /demo if you like
app.mount("/demo", StaticFiles(directory=str(ROOT / "public"), html=True), name="site")
