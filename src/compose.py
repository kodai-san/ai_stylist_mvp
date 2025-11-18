# src/compose.py
#imports
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import numpy as np
try:
    import cv2 as cv
    HAS_CV = True
except Exception:
    HAS_CV = False

# --- rembg (optional) ---
REMBG_SESSION = None
HAS_REMBG = False
REMBG_ERROR = ""

try:
    # NOTE: new_session comes from the rembg root package (not rembg.session)
    from rembg import remove as rembg_remove, new_session

    # Try a strong model first; switch to "u2net" if needed
    REMBG_SESSION = new_session("isnet-general-use")
    HAS_REMBG = True
except Exception as e:
    REMBG_SESSION = None
    HAS_REMBG = False
    REMBG_ERROR = repr(e)

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent.parent  # project root (parent of src/)
PRODUCTS_CSV = ROOT / "data" / "raw" / "products.csv"
IMG_DIR = ROOT / "data" / "raw" / "images"  # for info; serve via /static/images/<pid>.jpg


# ---------- helpers ----------
def _s(x) -> str:
    if x is None:
        return ""
    try:
        # handle pandas/np NaN
        import math

        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return str(x)


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(PRODUCTS_CSV, dtype={"product_id": str}).fillna("")
    return df.set_index("product_id")


def _local_path_from_csv(local_image_value: str) -> Optional[Path]:
    """
    CSV stores e.g. /static/images/<pid>.jpg. Map to real file under data/raw/images/.
    """
    p = _s(local_image_value).strip()
    if not p:
        return None
    if p.startswith("/static/"):
        rel = p.replace("/static/", "")  # images/xxx.jpg
        return (ROOT / "data" / "raw" / rel).resolve()
    # If someone stored a relative path already:
    return (ROOT / p.lstrip("/")).resolve()


def _fetch_image(url: str, timeout: int = 15) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGBA")
    except Exception:
        return None


def _load_product_image(df: pd.DataFrame, pid: str) -> Optional[Image.Image]:
    """
    Load product image as RGBA. Prefer local_image; fallback to image_url.
    """
    if pid not in df.index:
        return None
    row = df.loc[pid]
    # 1) local cached image (fastest & most reliable)
    lp = _local_path_from_csv(row.get("local_image", ""))
    if lp and lp.exists():
        try:
            return Image.open(lp).convert("RGBA")
        except Exception:
            pass
    # 2) remote image_url
    url = _s(row.get("image_url", ""))
    if url:
        return _fetch_image(url)
    return None


def _fit_into(img: Image.Image, box: Tuple[int, int]) -> Image.Image:
    """
    Contain-fit image into box (keep aspect). Letterbox with transparent padding.
    """
    w, h = img.size
    bw, bh = box
    scale = min(bw / max(1, w), bh / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img2 = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    canvas.paste(img2, ((bw - nw) // 2, (bh - nh) // 2), img2)
    return canvas


def _label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str):
    """
    Rounded label with default font.
    """
    x, y = xy
    pad = 8
    font = ImageFont.load_default()
    # textbbox returns (l, t, r, b); width = r-l, height = b-t
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    rect = (x, y, x + tw + pad * 2, y + th + pad * 2)
    draw.rounded_rectangle(rect, radius=10, fill=(255, 255, 255, 230))
    draw.text((x + pad, y + pad), text, fill=(15, 23, 42), font=font)


def _make_grid_bg(size: Tuple[int, int]) -> Image.Image:
    W, H = size
    bg = Image.new("RGBA", (W, H), (246, 249, 251, 255))  # light greenish/neutral
    tile = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rectangle((0, 0, 39, 39), outline=(226, 232, 240, 140))
    for x in range(0, W, 40):
        for y in range(0, H, 40):
            bg.alpha_composite(tile, (x, y))
    return bg


def _drop_shadow(card: Image.Image, blur_px: int = 12, offset=(12, 12)) -> Image.Image:
    """
    Simple drop shadow under a given card.
    """
    offx, offy = offset
    shadow_color = (0, 0, 0, 64)
    base = Image.new("RGBA", (card.width + offx + blur_px * 2, card.height + offy + blur_px * 2), (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", card.size, shadow_color)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur_px))
    base.paste(shadow_layer, (offx + blur_px, offy + blur_px), shadow_layer)
    base.alpha_composite(card, (0, 0))
    return base

def _pil_to_rgba_np(im: Image.Image) -> np.ndarray:
    return np.array(im.convert("RGBA"))

def _np_to_pil_rgba(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")

def _mask_from_rgba_or_white_bg(rgba: np.ndarray) -> np.ndarray:
    a = rgba[:,:,3]
    if np.any(a == 0) and np.any(a == 255):
        m = (a > 0).astype(np.uint8)*255
    else:
        bgr  = cv.cvtColor(rgba[:,:,:3], cv.COLOR_RGBA2BGR)
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        m = cv.threshold(255 - gray, 10, 255, cv.THRESH_BINARY)[1]
        m = cv.morphologyEx(m, cv.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    return m

def _largest_contour_mask(mask: np.ndarray) -> np.ndarray:
    cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return mask
    c = max(cnts, key=cv.contourArea)
    out = np.zeros_like(mask); cv.drawContours(out,[c],-1,255,-1)
    return out

def _trapezoid_warp(rgba: np.ndarray, side: str, shrink: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    if side == "right":
        dst = np.float32([[0,0],[w*shrink,0],[w*shrink,h],[0,h]])
    else:
        dst = np.float32([[w*(1-shrink),0],[w,0],[w,h],[w*(1-shrink),h]])
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(rgba, M, (w,h),
        flags=cv.INTER_LANCZOS4, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0,0))

def _width_ratio(mask: np.ndarray, top_q=0.2, bot_q=0.8) -> float:
    h = mask.shape[0]
    def width_at(q):
        y = max(0, min(h-1, int(h*q)))
        xs = np.where(mask[y] > 0)[0]
        return int(xs.max() - xs.min() + 1) if len(xs) else 1
    topw, botw = width_at(top_q), width_at(bot_q)
    return max(topw,1)/max(botw,1)

def _verticality_score(mask: np.ndarray) -> float:
    edges = cv.Canny(mask, 50, 150)
    lines = cv.HoughLines(edges, 1, np.pi/180, 60)
    if lines is None: return 1.0
    dev = []
    for rho,theta in lines[:,0]:
        dev.append(min(abs(theta-0), abs(theta-np.pi/2)))
    return float(np.mean(dev)) if dev else 1.0

def _symmetry_score(mask: np.ndarray) -> float:
    m2 = cv.flip(mask, 1)
    xor = cv.bitwise_xor(mask, m2)
    return xor.sum() / (255.0 * mask.size + 1e-6)

def auto_perspective_normalize_pil(
    pil_rgba: Image.Image,
    side_candidates=("right","left"),
    strength_guard=(0.86, 0.98),
) -> Image.Image:
    if not HAS_CV:
        return pil_rgba
    rgba = _pil_to_rgba_np(pil_rgba)
    base = _largest_contour_mask(_mask_from_rgba_or_white_bg(rgba))
    if base.sum() == 0:
        return pil_rgba
    best = {"score": 1e9, "img": rgba}
    for side in side_candidates:
        for s in np.linspace(strength_guard[0], strength_guard[1], 7):
            w = _trapezoid_warp(rgba, side, float(s))
            m = _largest_contour_mask(_mask_from_rgba_or_white_bg(w))
            wr = _width_ratio(m); vs = _verticality_score(m); ss = _symmetry_score(m)
            score = abs(1.0-wr)*2.0 + vs*0.8 + ss*0.6
            if score < best["score"]:
                best = {"score": score, "img": w}
    return _np_to_pil_rgba(best["img"])

# ---------- background removal ----------
def _rembg_cutout(pil_rgba: Image.Image) -> Image.Image:
    """
    Stronger remover with alpha matting + small post-process to clean halos.
    Returns RGBA with transparent background.
    """
    if not HAS_REMBG:
        return pil_rgba

    buf = BytesIO()
    pil_rgba.convert("RGB").save(buf, "PNG")
    data = buf.getvalue()

    out_bytes = rembg_remove(
        data,
        session=REMBG_SESSION,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=5,
        post_process_mask=True,
    )
    cut = Image.open(BytesIO(out_bytes)).convert("RGBA")

    # Optional light mask cleanup to reduce white fringes:
    a = cut.split()[3]
    a = a.point(lambda p: 0 if p < 12 else (255 if p > 242 else p))
    a = a.filter(ImageFilter.MedianFilter(size=3))
    cut.putalpha(a)
    return cut

# ---------- public API ----------
def compose_pair(
    base_pid: str,
    rec_pid: str,
    *,
    cutout: bool = False,
    bg_kind: str = "white",  # "white" | "grid"
    canvas_size: Tuple[int, int] = (1600, 900),
    auto_perspective: bool = False,
    auto_strength_min: float = 0.86,
    auto_strength_max: float = 0.98,
) -> BytesIO:
    """
    Create a side-by-side composite (base + recommendation) and return a PNG buffer.

    - base_pid / rec_pid: product_ids present in products.csv
    - cutout=True uses rembg to remove backgrounds (requires rembg installed)
    - bg_kind: "white" or "grid"
    - canvas_size: (W, H)
    """
    df = _load_df()
    base_img = _load_product_image(df, base_pid)
    rec_img = _load_product_image(df, rec_pid)
    if base_img is None or rec_img is None:
        raise ValueError("Could not load one or both product images. Check local_image / image_url.")

    if cutout and HAS_REMBG:
        try:
            base_img = _rembg_cutout(base_img)
            rec_img = _rembg_cutout(rec_img)
        except Exception:
            # If remover fails, proceed with originals
            pass

    if auto_perspective:
        try:
            base_img = auto_perspective_normalize_pil(
                base_img, strength_guard=(auto_strength_min, auto_strength_max)
            )
            rec_img = auto_perspective_normalize_pil(
                rec_img, strength_guard=(auto_strength_min, auto_strength_max)
            )
        except Exception:
            pass

    W, H = canvas_size
    bg = Image.new("RGBA", (W, H), (255, 255, 255, 255)) if bg_kind == "white" else _make_grid_bg((W, H))

    pad = 32
    col_w = (W - pad * 3) // 2
    col_h = H - pad * 2

    # cards
    card_bg = (255, 255, 255, 255)
    card1 = Image.new("RGBA", (col_w, col_h), card_bg)
    card2 = Image.new("RGBA", (col_w, col_h), card_bg)

    # fit images into "media" area (leave space at bottom for labels)
    media_h = col_h - 80
    base_fit = _fit_into(base_img, (col_w, media_h))
    rec_fit = _fit_into(rec_img, (col_w, media_h))

    card1.paste(base_fit, (0, 0), base_fit)
    card2.paste(rec_fit, (0, 0), rec_fit)

    d1 = ImageDraw.Draw(card1)
    d2 = ImageDraw.Draw(card2)
    _label(d1, (10, col_h - 60), f"Base: {base_pid}")
    _label(d2, (10, col_h - 60), f"Best match: {rec_pid}")

    # drop shadows & compose onto background
    bg.alpha_composite(_drop_shadow(card1), (pad, pad))
    bg.alpha_composite(_drop_shadow(card2), (pad * 2 + col_w, pad))

    out = BytesIO()
    bg.convert("RGB").save(out, "PNG", optimize=True)
    out.seek(0)
    return out


# ---------- CLI test ----------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compose base+recommendation into one PNG.")
    ap.add_argument("base_id", help="base product_id")
    ap.add_argument("rec_id", help="recommended product_id")
    ap.add_argument("--cutout", action="store_true", help="remove background with rembg")
    ap.add_argument("--bg", choices=["white", "grid"], default="white", help="background kind")
    ap.add_argument("--out", default="out.png", help="output file path")
    args = ap.parse_args()

    buf = compose_pair(args.base_id, args.rec_id, cutout=args.cutout, bg_kind=args.bg)
    with open(args.out, "wb") as f:
        f.write(buf.getvalue())
    print(f"Wrote {args.out} (cutout={args.cutout}, bg={args.bg})")