# src/prepare_ikea_hf.py
import argparse, csv, os, time, tempfile, shutil
from pathlib import Path
from datasets import load_dataset
from slugify import slugify
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "raw" / "products.csv"
IMG_DIR = ROOT / "data" / "raw" / "images"

# simple heuristics to infer category and tags from text
KW_TO_CAT = [
    ("sofa", "sofa"), ("armchair", "chair"), ("chair", "chair"), ("stool", "chair"),
    ("bed", "bed"), ("nightstand", "nightstand"), ("table", "table"), ("desk", "desk"),
    ("cabinet", "storage"), ("drawer", "storage"), ("dresser", "storage"), ("shelf", "storage"),
    ("lamp", "lamp"), ("light", "lamp"), ("rug", "rug"), ("sink", "bathroom"),
]
TAG_WORDS = ["wood", "oak", "walnut", "metal", "steel", "glass",
             "white", "black", "beige", "grey", "gray", "brown",
             "minimalist", "modern", "classic", "scandinavian", "natural"]

# IMPORTANT: prefer "desc" (HF key) first
TEXT_FIELDS = ("desc", "description", "caption", "text", "prompt", "title")

def infer_category(text: str) -> str:
    t = (text or "").lower()
    for kw, cat in KW_TO_CAT:
        if kw in t:
            return cat
    return "furniture"

def infer_tags(text: str) -> str:
    t = (text or "").lower()
    tags = [w for w in TAG_WORDS if w in t]
    return ", ".join(sorted(set(tags)))

def to_str(x): return "" if x is None else str(x)

def first_present(d: dict, keys=TEXT_FIELDS) -> str:
    for k in keys:
        if k in d and d[k]:
            v = d[k]
            return "" if v is None else str(v)
    return ""

def safe_write_csv(rows, out_path: Path, fieldnames):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="products_", suffix=".csv", dir=out_path.parent)
    os.close(fd)
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    for _ in range(6):
        try:
            os.replace(tmp, out_path)  # atomic on Windows
            return True
        except PermissionError:
            time.sleep(0.8)
    fallback = out_path.parent / f"products_{int(time.time())}.csv"
    shutil.move(tmp, fallback)
    print(f"[WARN] Could not replace {out_path}; wrote {fallback} instead.")
    return False

def main(limit: int | None, split: str):
    print(f"Loading HF dataset Sam1995/ikea_dataset_3.0 split={split} …")
    ds = load_dataset("Sam1995/ikea_dataset_3.0", split=split)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()

    for i, item in enumerate(ds):
        if limit and i >= limit:
            break

        img = item.get("image", None)
        if img is None:
            continue

        name = to_str(item.get("title") or item.get("name"))
        desc = first_present(item)  # <- grabs "desc" if present
        url  = to_str(item.get("url"))
        cat0 = to_str(item.get("category") or item.get("label"))

        # stable product_id
        base = slugify(name) if name else f"ikea-{i}"
        pid = base or f"ikea-{i}"
        j = 1
        while pid in seen:
            pid = f"{base}-{j}"; j += 1
        seen.add(pid)

        # save image locally
        if isinstance(img, Image.Image):
            pil = img.convert("RGB")
        else:
            try:
                pil = Image.fromarray(img).convert("RGB")
            except Exception:
                continue
        (IMG_DIR / f"{pid}.jpg").parent.mkdir(parents=True, exist_ok=True)
        pil.save(IMG_DIR / f"{pid}.jpg", "JPEG", quality=90)

        text_blob = f"{name} {desc}".strip()
        category = (cat0 or infer_category(text_blob)).lower()
        style = infer_tags(text_blob)
        fallback_name = (name or (desc[:80] if desc else "") or f"IKEA item {i}").strip() or f"IKEA item {i}"

        rows.append({
            "product_id": pid,
            "product_name": fallback_name,
            "image_url": "",  # we use local_image
            "category": category,
            "price": "",
            "page_url": url,
            "style_tags": style,
            "local_image": f"/static/images/{pid}.jpg",
            "description": desc,  # map HF 'desc' -> CSV 'description'
        })

    if rows:
        cols = ["product_id","product_name","image_url","category","price","page_url",
                "style_tags","local_image","description"]
        ok = safe_write_csv(rows, OUT_CSV, cols)
        if ok:
            print(f"Wrote {len(rows)} rows -> {OUT_CSV}")
        else:
            print(f"Close apps locking {OUT_CSV}; a fallback copy was written.")
        print(f"Images saved under -> {IMG_DIR}")
    else:
        print("No rows written; check dataset split and fields.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", type=str, default="train", help='e.g. "train" or "train+validation+test"')
    args = ap.parse_args()
    main(args.limit, args.split)