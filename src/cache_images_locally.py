import os, io
from pathlib import Path
import httpx
import pandas as pd
from PIL import Image

CSV_IN  = "data/raw/products.csv"
IMG_DIR = Path("data/raw/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

def fetch_image(url: str) -> bytes | None:
    if not url: return None
    try:
        with httpx.Client(follow_redirects=True, timeout=20, headers={"User-Agent":"Mozilla/5.0"}) as c:
            r = c.get(url)
            r.raise_for_status()
            ct = r.headers.get("content-type","").lower()
            if not ct.startswith("image/"):  # avoid 1x1 gifs / html errors
                return None
            if len(r.content) < 2048:       # skip tiny “pixel” responses
                return None
            return r.content
    except Exception:
        return None

def save_as_jpg(raw: bytes, out_path: Path) -> bool:
    try:
        im = Image.open(io.BytesIO(raw))
        # basic sanity: skip microscopic placeholders
        if im.width < 100 or im.height < 100:
            return False
        if im.mode not in ("RGB","RGBA"):
            im = im.convert("RGB")
        else:
            im = im.convert("RGB")
        im.save(out_path, format="JPEG", quality=90, optimize=True)
        return True
    except Exception:
        return False

def main():
    df = pd.read_csv(CSV_IN)
    local_paths = []
    ok = 0
    for _, row in df.iterrows():
        pid = str(row["product_id"])
        url = str(row.get("image_url",""))
        out = IMG_DIR / f"{pid}.jpg"
        if out.exists():
            local_paths.append(f"/static/images/{pid}.jpg"); ok += 1; continue
        raw = fetch_image(url)
        if not raw:
            local_paths.append("")
            continue
        if save_as_jpg(raw, out):
            local_paths.append(f"/static/images/{pid}.jpg"); ok += 1
        else:
            local_paths.append("")
    df["local_image"] = local_paths
    df.to_csv(CSV_IN, index=False)
    print(f"Cached {ok}/{len(df)} images → {IMG_DIR}")

if __name__ == "__main__":
    main()