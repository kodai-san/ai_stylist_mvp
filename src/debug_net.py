import os, csv, time, argparse, re
from pathlib import Path
from typing import List, Dict
import requests

BASE = "https://dummyjson.com"
UA = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; MVPBot/0.2)")
HEADERS = {"User-Agent": UA, "Accept": "application/json"}

def fetch_category(category: str, limit: int = 200, per_page: int = 100) -> List[Dict]:
    """Fetch up to `limit` products from DummyJSON category with pagination."""
    items: List[Dict] = []
    skip = 0
    while len(items) < limit:
        batch = min(per_page, limit - len(items))
        url = f"{BASE}/products/category/{category}?limit={batch}&skip={skip}"
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        j = r.json()
        prods = j.get("products", [])
        if not prods:
            break
        items.extend(prods)
        skip += batch
        time.sleep(0.2)  # be polite
    return items[:limit]

# --- optional: very lightweight auto-tagging from title/description ---
KEYWORDS = [
    ("wood", "wood"), ("oak", "wood"), ("walnut", "wood"),
    ("leather", "leather"), ("fabric", "fabric"), ("linen", "fabric"),
    ("metal", "metal"), ("steel", "metal"), ("iron", "metal"),
    ("glass", "glass"),
    ("minimal", "minimalist"), ("modern", "modern"), ("scandinavian", "scandinavian"),
    ("rustic", "rustic"), ("industrial", "industrial"),
    ("black", "black"), ("white", "white"), ("gray", "gray"), ("beige", "beige"),
    ("compact", "compact"), ("folding", "folding"), ("adjustable", "adjustable")
]
KW_RE = [(re.compile(rf"\b{re.escape(k)}\b", re.I), tag) for k, tag in KEYWORDS]

def auto_tags_from_text(*texts: str) -> str:
    found = []
    blob = " ".join(t for t in texts if t).lower()
    for rx, tag in KW_RE:
        if rx.search(blob):
            found.append(tag)
    # de-dup preserve order
    seen, out = set(), []
    for t in found:
        if t not in seen:
            seen.add(t); out.append(t)
    return ", ".join(out)

def to_row(p: Dict, use_auto_tags: bool) -> Dict:
    # image preference: first in images, else thumbnail
    images = p.get("images") or []
    img = images[0] if images else (p.get("thumbnail") or "")
    pid = str(p.get("id"))
    name = p.get("title", "").strip()
    cat = p.get("category", "").strip()  # e.g., "furniture"
    price = str(p.get("price", ""))
    page_url = f"{BASE}/products/{pid}"  # JSON page; fine for demo

    style_tags = ""
    if use_auto_tags:
        style_tags = auto_tags_from_text(name, p.get("description", ""), cat)

    return {
        "product_id": pid,
        "product_name": name,
        "image_url": img,
        "category": cat,
        "price": price,
        "page_url": page_url,
        "style_tags": style_tags,
    }

def main():
    ap = argparse.ArgumentParser(description="Seed products.csv from DummyJSON furniture")
    ap.add_argument("--limit", type=int, default=120, help="max products to fetch")
    ap.add_argument("--auto-tags", action="bool", nargs="?", const=True, default=False,
                    help="generate simple style_tags from title/description")
    args = ap.parse_args()

    print(f"Fetching up to {args.limit} furniture products from DummyJSON…")
    prods = fetch_category("furniture", limit=args.limit, per_page=100)
    if not prods:
        raise SystemExit("No products returned from DummyJSON (network issue?)")

    rows = [to_row(p, args.auto_tags) for p in prods if p.get("title") and (p.get("images") or p.get("thumbnail"))]

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    out_csv = "data/raw/products.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["product_id","product_name","image_url","category","price","page_url","style_tags"])
        w.writeheader(); w.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out_csv}")
    # quick peek
    for r in rows[:5]:
        print("•", r["product_id"], r["product_name"], "—", r["image_url"])

if __name__ == "__main__":
    main()