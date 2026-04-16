"""Input normalization for TRELLIS.2.

Goal: lift dark/cluttered/off-center captures (Quest 3 office) into the
brightness/contrast/framing band the stock photos live in.

Pipeline:
  1. CLAHE on L (LAB space)            — local contrast / shadow lift
  2. Gray-world white balance          — kill tungsten cast
  3. Conditional Lanczos upscale       — get long edge >= MIN_LONG_EDGE
  4. (post-rembg) Square auto-crop     — center subject, fill ~38% of frame

Steps 1-3 run on the raw BGR image before GFPGAN+ESRGAN.
Step 4 runs on the RGBA cutout returned by BiRefNet, before TRELLIS.
"""
import cv2
import numpy as np
from PIL import Image

# TRELLIS internally downscales to <=1024 and does its own alpha-bbox square
# crop, so the only reason to upscale here is to give GFPGAN's face restorer
# enough pixels to work on. ~1500 long edge gives GFPGAN a >150px face for any
# of our typical Quest 3 captures without wasting compute on pixels that
# trellis will throw away.
MIN_LONG_EDGE = 1500
TARGET_PERSON_FRAC = 0.62  # only used by autocrop, which is now opt-in


def clahe_lab(bgr, clip=2.5, tile=(8, 8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def white_balance_grayworld(bgr):
    res = bgr.astype(np.float32)
    means = res.reshape(-1, 3).mean(axis=0)  # B, G, R
    gray = means.mean()
    scale = gray / np.maximum(means, 1e-6)
    res *= scale
    return np.clip(res, 0, 255).astype(np.uint8)


def conditional_upscale(bgr, min_long_edge=MIN_LONG_EDGE):
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge >= min_long_edge:
        return bgr, 1.0
    factor = min_long_edge / long_edge
    new_w = int(round(w * factor))
    new_h = int(round(h * factor))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4), factor


def preprocess_raw(bgr):
    """Steps 1-3: run before GFPGAN+ESRGAN on the raw BGR input."""
    out = clahe_lab(bgr)
    out = white_balance_grayworld(out)
    out, scale = conditional_upscale(out)
    return out, scale


def autocrop_square_from_alpha(rgba_pil, target_person_frac=TARGET_PERSON_FRAC, pad_color=(127, 127, 127)):
    """Step 4: post-rembg square crop centered on the subject.

    Takes the RGBA cutout, finds the alpha bbox, and re-frames it as a
    square where the person fills `target_person_frac` of the side. Areas
    outside the original frame are padded with mid-gray.

    Returns a new RGBA PIL image (still RGBA so the alpha mask is preserved).
    """
    rgba = np.array(rgba_pil)
    if rgba.shape[2] == 3:
        # No alpha — fall back to centered square crop
        h, w = rgba.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return Image.fromarray(rgba[y0:y0+side, x0:x0+side])

    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 128)
    if len(ys) == 0:
        return rgba_pil

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_h = y1 - y0 + 1
    bbox_w = x1 - x0 + 1
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    # Side length such that the longer bbox dimension fills target_person_frac
    side = int(round(max(bbox_w, bbox_h) / target_person_frac))

    sx0 = int(round(cx - side / 2))
    sy0 = int(round(cy - side / 2))
    sx1 = sx0 + side
    sy1 = sy0 + side

    H, W = alpha.shape
    pad_left = max(0, -sx0)
    pad_top = max(0, -sy0)
    pad_right = max(0, sx1 - W)
    pad_bottom = max(0, sy1 - H)

    crop_x0 = max(0, sx0)
    crop_y0 = max(0, sy0)
    crop_x1 = min(W, sx1)
    crop_y1 = min(H, sy1)

    cropped = rgba[crop_y0:crop_y1, crop_x0:crop_x1]

    # Pad with mid-gray RGB and 0 alpha
    padded = np.zeros((side, side, 4), dtype=np.uint8)
    padded[:, :, 0] = pad_color[0]
    padded[:, :, 1] = pad_color[1]
    padded[:, :, 2] = pad_color[2]
    padded[pad_top:pad_top + cropped.shape[0],
           pad_left:pad_left + cropped.shape[1]] = cropped

    return Image.fromarray(padded)
