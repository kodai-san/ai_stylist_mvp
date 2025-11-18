# src/enrich.py
import argparse, re
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from PIL import Image
from slugify import slugify

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "raw" / "products.csv"
RAW_DIR = ROOT / "data" / "raw"           # /static maps here
IMG_DIR = RAW_DIR / "images"              # we expect /static/images/<pid>.jpg

KW_TO_ELEMENT = [
    ("sofa", "sofa"), ("couch", "sofa"), ("sectional", "sofa"),
    ("armchair", "chair"), ("chair", "chair"), ("stool", "chair"),
    ("bench", "bench"),
    ("bed", "bed"), ("headboard", "bed"), ("mattress", "bed"),
    ("table", "table"), ("coffee table", "table"), ("dining", "table"),
    ("desk", "desk"),
    ("cabinet", "storage"), ("drawer", "storage"), ("dresser", "storage"),
    ("wardrobe", "storage"), ("closet", "storage"), ("shelf", "storage"),
    ("bookshelf", "storage"), ("tv stand", "storage"), ("sideboard", "storage"),
    ("lamp", "lamp"), ("light", "lamp"), ("pendant", "lamp"), ("ceiling", "lamp"),
    ("rug", "rug"), ("carpet", "rug"),
    ("mirror", "mirror"),
    ("sink", "bathroom"), ("vanity", "bathroom"),
]

MATERIAL_WORDS = [
    "wood","oak","walnut","beech","pine","bamboo","rattan",
    "metal","steel","iron","aluminum","glass","marble",
    "fabric","leather","linen","cotton","wool","velvet","plastic","acrylic"
]

COLOR_WORDS = [
    "white","black","gray","grey","silver","charcoal",
    "brown","beige","cream","ivory","tan","oak","walnut",
    "red","maroon","pink","orange","yellow","gold",
    "green","olive","teal","mint",
    "blue","navy","indigo","sky",
    "purple","violet","lavender"
]

NAMED_COLORS = {
    "white": (245,245,245), "black": (20,20,20), "gray": (128,128,128), "grey": (128,128,128),
    "brown": (125,94,66), "beige": (222,209,183), "cream": (240,234,214), "ivory": (255,255,240), "tan": (210,180,140),
    "oak": (186,140,99), "walnut": (120,84,60),
    "red": (200,60,60), "maroon": (128,0,0), "pink": (236,142,180), "orange": (245,140,60),
    "yellow": (245,210,60), "gold": (212,175,55),
    "green": (70,160,100), "olive": (128,128,0), "teal": (0,128,128), "mint": (170,240,200),
    "blue": (80,140,220), "navy": (0,0,128), "indigo": (75,0,130), "sky": (135,206,235),
    "purple": (130,80,180), "violet": (148,0,211), "lavender": (181,126,220),
    "silver": (192,192,192), "charcoal": (54,69,79),
}

def text_of(row: pd.Series) -> str:
    name = str(row.get("product_name", "") or "")
    desc = str(row.get("description", "") or row.get("desc", "") or "")
    return f"{name} {desc}".strip().lower()

def infer_element(text: str) -> str:
    t = (text or "").lower()
    for kw, cat in KW_TO_ELEMENT:
        if kw in t:
            return cat
    return "furniture"

def infer_materials(text: str) -> str:
    t = (text or "").lower()
    mats = [w for w in MATERIAL_WORDS if w in t]
    # merge similar: "oak"/"walnut" imply "wood"
    if any(x in mats for x in ["oak","walnut","beech","pine","bamboo","rattan"]):
        if "wood" not in mats: mats.append("wood")
    return ", ".join(sorted(set(mats)))

def nearest_color_name(rgb: Tuple[int,int,int]) -> str:
    import math
    r,g,b = rgb
    best, bestd = None, 1e9
    for name, (R,G,B) in NAMED_COLORS.items():
        d = (r-R)**2 + (g-G)**2 + (b-B)**2
        if d < bestd: best, bestd = name, d
    return best or "unknown"

def infer_color_from_image(local_path: Path) -> Optional[str]:
    if not local_path or not local_path.is_file():
        return None
    try:
        img = Image.open(local_path).convert("RGB")
        img = img.resize((64,64))
        colors = img.getcolors(maxcolors=64*64)
        if not colors: return None
        colors.sort(reverse=True, key=lambda x: x[0])  # by count
        # skip near-white background dominance if exists
        for _, (r,g,b) in colors[:8]:
            name = nearest_color_name((r,g,b))
            if name not in ("white","cream","ivory"):  # try to avoid background
                return name
        return nearest_color_name(colors[0][1])
    except Exception:
        return None

def map_static_to_local(static_path: str) -> Path:
    # "/static/images/abc.jpg" -> ROOT/data/raw/images/abc.jpg
    s = str(static_path or "")
    if s.startswith("/static/"):
        return RAW_DIR / s[len("/static/"):]
    p = RAW_DIR / s.lstrip("\\/")
    return p

def normalize_tags(tags: str) -> str:
    if not tags: return ""
    parts = [t.strip().lower() for t in str(tags).split(",") if t.strip()]
    # de-duplicate & sort for stability
    return ", ".join(sorted(set(parts)))

def build_series_key(name: str) -> str:
    slug = slugify(name or "")
    if not slug: return ""
    parts = slug.split("-")
    return "-".join(parts[:2])  # first two tokens

def main(limit: Optional[int] = None, recompute_colors: bool = False):
    print("Enriching:", CSV)
    df = pd.read_csv(CSV, dtype={"product_id": str}).fillna("")
    if limit: df = df.head(limit)

    out_rows = []
    for _, row in df.iterrows():
        pid = str(row.get("product_id", ""))
        name = str(row.get("product_name", "") or f"Product {pid}")
        text = text_of(row)

        # element/material from text
        element = infer_element(text)
        materials = infer_materials(text)

        # color: prefer explicit word in text, else image
        found_color = None
        for w in COLOR_WORDS:
            if f" {w} " in f" {text} ":
                found_color = "grey" if w == "gray" else w
                break

        local_image = str(row.get("local_image", "") or "")
        color = found_color
        if not color or recompute_colors:
            local_path = map_static_to_local(local_image)
            col = infer_color_from_image(local_path)
            color = color or col or ""

        # series key & tidy tags
        series_key = build_series_key(name)
        style_tags = normalize_tags(row.get("style_tags", ""))

        enriched = dict(row)
        enriched["product_id"] = pid
        enriched["product_name"] = name
        enriched["element_type"] = element
        enriched["material"] = materials
        enriched["color"] = color
        enriched["series_key"] = series_key
        enriched["style_tags"] = style_tags
        out_rows.append(enriched)

    out = pd.DataFrame(out_rows)
    out.to_csv(CSV, index=False)
    print(f"Saved enriched CSV → {CSV}")
    print("Columns now include: element_type, material, color, series_key, style_tags (normalized)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--recompute-colors", action="store_true", help="force color inference from image")
    args = ap.parse_args()
    main(args.limit, args.recompute_colors)