from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import json
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --- Optional OpenCV (used for robust masks & perspective) ---
try:
    import cv2 as cv
    HAS_CV = True
except Exception:
    HAS_CV = False

# --- Optional rembg (background removal) ---
REMBG_SESSION = None
HAS_REMBG = False
REMBG_ERROR = ""
try:
    from rembg import remove as rembg_remove, new_session
    REMBG_SESSION = new_session("isnet-general-use")
    HAS_REMBG = True
except Exception as e:
    REMBG_ERROR = repr(e)
    HAS_REMBG = False
    REMBG_SESSION = None

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PRODUCTS_CSV = ROOT / "data" / "raw" / "products.csv"

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def _s(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    return str(x)

def _load_df() -> pd.DataFrame:
    df = pd.read_csv(PRODUCTS_CSV, dtype={"product_id": str}).fillna("")
    return df.set_index("product_id")

def _local_path_from_csv(local_image_value: str) -> Optional[Path]:
    """
    CSV stores e.g. /static/images/<pid>.jpg → map to data/raw/images/<pid>.jpg
    """
    p = _s(local_image_value).strip()
    if not p:
        return None
    if p.startswith("/static/"):
        rel = p.replace("/static/", "")  # images/xxx.jpg
        return (ROOT / "data" / "raw" / rel).resolve()
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
    # 1) local cached image
    lp = _local_path_from_csv(row.get("local_image", ""))
    if lp and lp.exists():
        try:
            return Image.open(lp).convert("RGBA")
        except Exception:
            pass
    # 2) remote
    url = _s(row.get("image_url", ""))
    if url:
        return _fetch_image(url)
    return None

# --------------------------------------------------------------------------------------
# Small drawing helpers
# --------------------------------------------------------------------------------------
def _make_grid_bg(size: Tuple[int, int]) -> Image.Image:
    W, H = size
    bg = Image.new("RGBA", (W, H), (246, 249, 251, 255))
    tile = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rectangle((0, 0, 39, 39), outline=(226, 232, 240, 140))
    for x in range(0, W, 40):
        for y in range(0, H, 40):
            bg.alpha_composite(tile, (x, y))
    return bg

def _label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str):
    x, y = xy
    pad = 8
    font = ImageFont.load_default()
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    rect = (x, y, x + tw + pad * 2, y + th + pad * 2)
    draw.rounded_rectangle(rect, radius=10, fill=(255, 255, 255, 230))
    draw.text((x + pad, y + pad), text, fill=(15, 23, 42), font=font)

def _ground_shadow(canvas: Image.Image, obj: Image.Image, pos: Tuple[int,int], strength: float = 0.30):
    """
    Simple soft contact shadow under object's bottom.
    """
    x0, y0 = pos
    w, h = obj.size
    by = _robust_bottom_y(_alpha_np(obj))
    ew = int(w * 0.6)
    eh = max(6, int(h * 0.05))
    shadow = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(shadow)
    d.ellipse([w//2 - ew//2, by - eh//2, w//2 + ew//2, by + eh//2], fill=(0,0,0,int(255*strength)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
    canvas.alpha_composite(shadow, (x0, y0))
    return canvas

# --------------------------------------------------------------------------------------
# Alpha / masks / geometry
# --------------------------------------------------------------------------------------
def _alpha_np(img: Image.Image) -> np.ndarray:
    return np.array(img.split()[-1])

def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not HAS_CV:
        return mask
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        return mask
    c = max(cnts, key=cv.contourArea)
    out = np.zeros_like(mask)
    cv.drawContours(out, [c], -1, 255, -1)
    return out

def _robust_bottom_y(alpha: np.ndarray, close_k: int = 5, q: float = 0.97, min_run: int = 6) -> int:
    """
    Robust estimate of the object's bottom contact within its own coordinates.
    - keeps largest component
    - morphological closing to fill wire/mesh gaps (if cv available)
    - take 97th percentile of column bottoms with short solid run constraint
    """
    m = (alpha > 10).astype(np.uint8) * 255
    m = _largest_component(m)
    if HAS_CV:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_k, close_k))
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, k, iterations=1)

    h, w = m.shape
    bottoms = []
    for x in range(w):
        col = m[:, x]
        idx = np.where(col > 0)[0]
        if idx.size == 0: 
            continue
        y = int(idx.max())
        # check short solid run
        run = 1
        j = y - 1
        while j >= 0 and col[j] > 0 and run < 64:
            run += 1
            j -= 1
        if run >= min_run:
            bottoms.append(y)
    if not bottoms:
        return int(0.9 * h)
    return int(np.quantile(bottoms, q))

def _place_on_floor(canvas: Image.Image, obj: Image.Image, x_center: int, floor_y: int) -> Tuple[Image.Image, Tuple[int,int]]:
    """
    Composite obj so its robust bottom sits at floor_y.
    """
    yb = _robust_bottom_y(_alpha_np(obj))
    x0 = int(x_center - obj.size[0] // 2)
    y0 = int(floor_y - yb)
    layer = Image.new("RGBA", canvas.size, (0,0,0,0))
    layer.alpha_composite(obj, (x0, y0))
    canvas.alpha_composite(layer)
    return canvas, (x0, y0)

def _resize_to_height(pil_rgba: Image.Image, target_h: int) -> Image.Image:
    if target_h <= 0: 
        return pil_rgba
    w, h = pil_rgba.size
    scale = float(target_h) / max(1, h)
    return pil_rgba.resize((max(1, int(round(w*scale))), target_h), Image.LANCZOS)

def _alpha_intersection_at_positions(A: Image.Image, posA: Tuple[int,int], B: Image.Image, posB: Tuple[int,int]) -> int:
    """
    Count intersecting opaque pixels (A ∩ B) using alpha masks.
    """
    ax, ay = posA; bx, by = posB
    aw, ah = A.size; bw, bh = B.size

    left   = max(ax, bx)
    top    = max(ay, by)
    right  = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0

    Aa = _alpha_np(A)[top - ay:bottom - ay, left - ax:right - ax] > 10
    Ba = _alpha_np(B)[top - by:bottom - by, left - bx:right - bx] > 10
    return int((Aa & Ba).sum())

# --------------------------------------------------------------------------------------
# Optional: auto perspective normalize (gentle)
# --------------------------------------------------------------------------------------
def _pil_to_rgba_np(im: Image.Image) -> np.ndarray:
    return np.array(im.convert("RGBA"))

def _np_to_pil_rgba(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")

def _mask_from_rgba_or_white_bg(rgba: np.ndarray) -> np.ndarray:
    a = rgba[:,:,3]
    if np.any(a == 0) and np.any(a == 255):
        m = (a > 0).astype(np.uint8)*255
    else:
        if not HAS_CV:
            # simple fallback (assume white bg) without cv2
            gray = np.mean(rgba[:,:,:3], axis=2)
            m = ((255 - gray) > 10).astype(np.uint8)*255
            return m
        bgr  = cv.cvtColor(rgba[:,:,:3], cv.COLOR_RGBA2BGR)
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        m = cv.threshold(255 - gray, 10, 255, cv.THRESH_BINARY)[1]
        m = cv.morphologyEx(m, cv.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    return m

def _verticality_score(mask: np.ndarray) -> float:
    if not HAS_CV:
        return 1.0
    edges = cv.Canny(mask, 50, 150)
    lines = cv.HoughLines(edges, 1, np.pi/180, 60)
    if lines is None: 
        return 1.0
    dev = []
    for rho,theta in lines[:,0]:
        dev.append(min(abs(theta-0), abs(theta-np.pi/2)))
    return float(np.mean(dev)) if dev else 1.0

def _symmetry_score(mask: np.ndarray) -> float:
    if not HAS_CV:
        return 1.0
    m2 = cv.flip(mask, 1)
    xor = cv.bitwise_xor(mask, m2)
    return xor.sum() / (255.0 * mask.size + 1e-6)

def _width_ratio(mask: np.ndarray, top_q=0.2, bot_q=0.8) -> float:
    h = mask.shape[0]
    def width_at(q):
        y = max(0, min(h-1, int(h*q)))
        xs = np.where(mask[y] > 0)[0]
        return int(xs.max() - xs.min() + 1) if len(xs) else 1
    topw, botw = width_at(top_q), width_at(bot_q)
    return max(topw,1)/max(botw,1)

def _largest_contour_mask(mask: np.ndarray) -> np.ndarray:
    if not HAS_CV:
        return mask
    cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        return mask
    c = max(cnts, key=cv.contourArea)
    out = np.zeros_like(mask)
    cv.drawContours(out, [c], -1, 255, -1)
    return out

def _trapezoid_warp(rgba: np.ndarray, side: str, shrink: float) -> np.ndarray:
    if not HAS_CV: 
        return rgba
    h, w = rgba.shape[:2]
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    if side == "right":
        dst = np.float32([[0,0],[w*shrink,0],[w*shrink,h],[0,h]])
    else:
        dst = np.float32([[w*(1-shrink),0],[w,0],[w,h],[w*(1-shrink),h]])
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(rgba, M, (w,h),
        flags=cv.INTER_LANCZOS4, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0,0))

def auto_perspective_normalize_pil(
    pil_rgba: Image.Image,
    side_candidates=("right","left"),
    strength_guard=(0.88, 0.98),
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

# --------------------------------------------------------------------------------------
# Background removal
# --------------------------------------------------------------------------------------
def _rembg_cutout(pil_rgba: Image.Image) -> Image.Image:
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
    a = cut.split()[3].point(lambda p: 0 if p < 12 else (255 if p > 242 else p))
    a = a.filter(ImageFilter.MedianFilter(size=3))
    cut.putalpha(a)
    return cut

# --------------------------------------------------------------------------------------
# NEW: Side-by-side (adjacent) with collision avoidance → ONE image
# --------------------------------------------------------------------------------------
def compose_adjacent_pair(
    base_pid: str,
    rec_pid: str,
    *,
    cutout: bool = True,
    bg_kind: str = "white",               # "white" | "grid"
    canvas_size: Tuple[int,int] = (1200, 800),
    floor_ratio: float = 0.82,
    target_h: int = 560,                  # visual height for both
    min_gap_px: int = 28,                 # minimum horizontal gap
    avoid_pad_px: int = 6,                # extra safety pad (dilate masks virtually)
    auto_perspective: bool = False,
    auto_strength_min: float = 0.88,
    auto_strength_max: float = 0.98,
    order: str = "base_left",             # "base_left" | "rec_left"
) -> BytesIO:
    """
    Place two products side-by-side on a shared floor line, ensure no alpha collision,
    and return a single PNG.
    """
    df = _load_df()
    A = _load_product_image(df, base_pid)
    B = _load_product_image(df, rec_pid)
    if A is None or B is None:
        raise ValueError("Could not load one or both product images.")

    # optional cutout
    if cutout and HAS_REMBG:
        try:
            A = _rembg_cutout(A)
            B = _rembg_cutout(B)
        except Exception:
            pass

    # optional gentle perspective
    if auto_perspective and HAS_CV:
        try:
            A = auto_perspective_normalize_pil(A, strength_guard=(auto_strength_min, auto_strength_max))
            B = auto_perspective_normalize_pil(B, strength_guard=(auto_strength_min, auto_strength_max))
        except Exception:
            pass

    # scale to same visual height
    A = _resize_to_height(A.convert("RGBA"), target_h)
    B = _resize_to_height(B.convert("RGBA"), target_h)

    # canvas + floor
    W, H = canvas_size
    floor_y = int(H * float(floor_ratio))
    bg = Image.new("RGBA", (W, H), (255,255,255,255)) if bg_kind == "white" else _make_grid_bg((W, H))

    # choose who is left/right
    left_img, left_id = (A, base_pid) if order == "base_left" else (B, rec_pid)
    right_img, right_id = (B, rec_pid) if order == "base_left" else (A, base_pid)

    lw, lh = left_img.size
    rw, rh = right_img.size

    # initial centers (aim ~30/70 split)
    cx_left  = int(W * 0.36)
    cx_right = int(W * 0.64)

    # place once to get positions
    tmp = bg.copy()
    tmp, pos_left = _place_on_floor(tmp, left_img,  cx_left,  floor_y)
    tmp, pos_right = _place_on_floor(tmp, right_img, cx_right, floor_y)

    # collision avoidance: push apart symmetrically until there's a gap
    def _virt_overlap(A: Image.Image, pA: Tuple[int,int], B: Image.Image, pB: Tuple[int,int], pad: int) -> int:
        inter = _alpha_intersection_at_positions(A, pA, B, pB)
        if inter == 0 or pad <= 0 or not HAS_CV:
            return inter
        # approximate extra pad by dilating masks inside a cropped bbox
        ax, ay = pA; bx, by = pB
        aw, ah = A.size; bw, bh = B.size
        L = max(ax, bx) - pad; T = max(ay, by) - pad
        R = min(ax + aw, bx + bw) + pad; Btm = min(ay + ah, by + bh) + pad
        if R <= L or Btm <= T:
            return 0
        # crop alpha with padding (clamped)
        def crop_alpha(img, x0, y0, x1, y1, px, py):
            # clamp source region relative to img placement (px,py)
            w,h = img.size
            lx = max(x0 - px, 0); rx = min(x1 - px, w)
            ty = max(y0 - py, 0); by2 = min(y1 - py, h)
            if rx <= lx or by2 <= ty: 
                return None
            return (_alpha_np(img)[ty:by2, lx:rx] > 10).astype(np.uint8)*255

        Am = crop_alpha(A, L, T, R, Btm, ax, ay)
        Bm = crop_alpha(B, L, T, R, Btm, bx, by)
        if Am is None or Bm is None: 
            return 0
        if HAS_CV:
            k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (pad*2+1, pad*2+1))
            Am = cv.dilate(Am, k, iterations=1)
            Bm = cv.dilate(Bm, k, iterations=1)
            return int(((Am>0) & (Bm>0)).sum())
        else:
            return int(((Am>0) & (Bm>0)).sum())

    inter = _virt_overlap(left_img, pos_left, right_img, pos_right, avoid_pad_px)

    step = 12
    max_iter = 120
    it = 0
    while inter > 0 and it < max_iter:
        it += 1
        cx_left  = max(lw//2 + 8,  cx_left - step)
        cx_right = min(W - rw//2 - 8, cx_right + step)
        # re-place and re-check
        tmp = bg.copy()
        tmp, pos_left  = _place_on_floor(tmp, left_img,  cx_left,  floor_y)
        tmp, pos_right = _place_on_floor(tmp, right_img, cx_right, floor_y)

        # maintain minimal hard gap between right edge of left and left edge of right
        hard_gap = (pos_right[0] - pos_left[0] - lw)
        if hard_gap < min_gap_px:
            cx_left  = max(lw//2 + 8,  cx_left - (min_gap_px - hard_gap)//2)
            cx_right = min(W - rw//2 - 8, cx_right + (min_gap_px - hard_gap)//2)
            tmp = bg.copy()
            tmp, pos_left  = _place_on_floor(tmp, left_img,  cx_left,  floor_y)
            tmp, pos_right = _place_on_floor(tmp, right_img, cx_right, floor_y)

        inter = _virt_overlap(left_img, pos_left, right_img, pos_right, avoid_pad_px)

        # If we hit the border and still collide, gently downscale both and restart
        hit_left_border  = (cx_left <= lw//2 + 8)
        hit_right_border = (cx_right >= W - rw//2 - 8)
        if inter > 0 and (hit_left_border or hit_right_border):
            # shrink 5% and recompute from center
            target_h = max(320, int(target_h * 0.95))
            A = _resize_to_height(A, target_h)
            B = _resize_to_height(B, target_h)
            if order == "base_left":
                left_img, right_img = A, B
            else:
                left_img, right_img = B, A
            lw, lh = left_img.size; rw, rh = right_img.size
            cx_left, cx_right = int(W*0.34), int(W*0.66)
            tmp = bg.copy()
            tmp, pos_left  = _place_on_floor(tmp, left_img,  cx_left,  floor_y)
            tmp, pos_right = _place_on_floor(tmp, right_img, cx_right, floor_y)
            inter = _virt_overlap(left_img, pos_left, right_img, pos_right, avoid_pad_px)

    # final composite
    bg, pos_left  = _place_on_floor(bg, left_img,  cx_left,  floor_y)
    bg = _ground_shadow(bg, left_img,  pos_left, 0.22)

    bg, pos_right = _place_on_floor(bg, right_img, cx_right, floor_y)
    bg = _ground_shadow(bg, right_img, pos_right, 0.28)

    # labels (optional)
    d = ImageDraw.Draw(bg)
    _label(d, (12, 12), f"left: {left_id}")
    _label(d, (12, 44), f"right: {right_id}")

    out = BytesIO()
    bg.convert("RGB").save(out, "PNG", optimize=True)
    out.seek(0)
    return out

# --------------------------------------------------------------------------------------
# (Kept) Existing simple side-by-side cards, if you still use it elsewhere
# --------------------------------------------------------------------------------------
def compose_pair(
    base_pid: str,
    rec_pid: str,
    *,
    cutout: bool = False,
    bg_kind: str = "white",
    canvas_size: Tuple[int, int] = (1600, 900),
    auto_perspective: bool = False,
    auto_strength_min: float = 0.88,
    auto_strength_max: float = 0.98,
) -> BytesIO:
    """
    Original card-style side-by-side (kept for compatibility).
    """
    df = _load_df()
    A = _load_product_image(df, base_pid)
    B = _load_product_image(df, rec_pid)
    if A is None or B is None:
        raise ValueError("Could not load one or both product images.")

    if cutout and HAS_REMBG:
        try:
            A = _rembg_cutout(A); B = _rembg_cutout(B)
        except Exception:
            pass
    if auto_perspective and HAS_CV:
        try:
            A = auto_perspective_normalize_pil(A, strength_guard=(auto_strength_min, auto_strength_max))
            B = auto_perspective_normalize_pil(B, strength_guard=(auto_strength_min, auto_strength_max))
        except Exception:
            pass

    W, H = canvas_size
    pad = 32
    col_w = (W - pad * 3) // 2
    col_h = H - pad * 2
    media_h = col_h - 80

    def _fit_into(img: Image.Image, box: Tuple[int,int]) -> Image.Image:
        w, h = img.size; bw, bh = box
        s = min(bw/max(1,w), bh/max(1,h))
        nw, nh = max(1,int(w*s)), max(1,int(h*s))
        tmp = img.resize((nw,nh), Image.LANCZOS)
        canvas = Image.new("RGBA",(bw,bh),(0,0,0,0))
        canvas.paste(tmp,((bw-nw)//2,(bh-nh)//2),tmp)
        return canvas

    bg = Image.new("RGBA",(W,H),(255,255,255,255)) if bg_kind=="white" else _make_grid_bg((W,H))
    card1 = Image.new("RGBA",(col_w,col_h),(255,255,255,255))
    card2 = Image.new("RGBA",(col_w,col_h),(255,255,255,255))
    card1.paste(_fit_into(A,(col_w,media_h)),(0,0),_fit_into(A,(col_w,media_h)))
    card2.paste(_fit_into(B,(col_w,media_h)),(0,0),_fit_into(B,(col_w,media_h)))
    d1 = ImageDraw.Draw(card1); d2 = ImageDraw.Draw(card2)
    _label(d1,(10,col_h-60),f"Base: {base_pid}")
    _label(d2,(10,col_h-60),f"Best match: {rec_pid}")

    # light drop shadow
    def _shadow(card: Image.Image) -> Image.Image:
        offx, offy, blur = 12, 12, 12
        base = Image.new("RGBA",(card.width+offx+blur*2,card.height+offy+blur*2),(0,0,0,0))
        s = Image.new("RGBA",card.size,(0,0,0,64)).filter(ImageFilter.GaussianBlur(blur))
        base.paste(s,(offx+blur,offy+blur),s)
        base.alpha_composite(card,(0,0))
        return base

    bg.alpha_composite(_shadow(card1),(pad,pad))
    bg.alpha_composite(_shadow(card2),(pad*2+col_w,pad))
    out = BytesIO()
    bg.convert("RGB").save(out,"PNG",optimize=True)
    out.seek(0)
    return out