"""Image loading and preprocessing.

The pipeline applied to every image (reference and candidate alike):

1. Open the file and verify it is not corrupt.
2. Apply the EXIF orientation tag.
3. Extract the alpha channel (if any) and convert colour data to RGB.
4. Build a foreground mask (from alpha, or by thresholding the background
   colour, depending on ``mask_mode``).
5. Composite transparent pixels onto the configured background colour.
6. Crop (``none`` / ``center_crop`` / ``foreground_bbox``).
7. Resize to fit the square canvas while keeping the aspect ratio (never
   stretched) and paste it centred on a canvas filled with the background
   colour. The mask is transformed with the exact same geometry.

The result is a :class:`PreprocessedImage` holding an ``HxWx3`` uint8 RGB
array, an optional boolean mask and some metadata for reporting.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import PreprocessingConfig

logger = logging.getLogger(__name__)

_RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
}


class ImageLoadError(RuntimeError):
    """Raised when an image file cannot be read or is corrupt."""


@dataclass
class PreprocessedImage:
    """Result of :func:`preprocess_image`.

    Attributes:
        rgb: ``(H, W, 3)`` uint8 RGB array on the canvas.
        mask: ``(H, W)`` boolean foreground mask on the canvas, or ``None``
            when no reliable mask could be produced.
        mask_source: ``"alpha"``, ``"background_threshold"`` or ``"none"``.
        mask_reliable: whether ``mask`` should be trusted for silhouette IoU.
        meta: extra information (original size, crop box, scale, ...).
    """

    rgb: np.ndarray
    mask: np.ndarray | None
    mask_source: str
    mask_reliable: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_mask(self) -> bool:
        return self.mask is not None and self.mask_reliable

    def to_pil(self) -> Image.Image:
        return Image.fromarray(self.rgb, mode="RGB")

    def mask_to_pil(self) -> Image.Image | None:
        if self.mask is None:
            return None
        return Image.fromarray((self.mask.astype(np.uint8) * 255), mode="L")

    def meta_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("rgb")
        d.pop("mask")
        return d


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_image(path: str | Path) -> Image.Image:
    """Open an image, verify it is not corrupt and fix EXIF orientation.

    Raises:
        ImageLoadError: if the file is missing, unreadable or corrupt.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Image file not found: {path}")
    try:
        with Image.open(path) as probe:
            probe.verify()  # cheap integrity check; invalidates the handle
        img = Image.open(path)
        img.load()  # force full decode so truncated files fail here
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageLoadError(f"Cannot read image {path}: {exc}") from exc
    try:
        img = ImageOps.exif_transpose(img)
    except Exception as exc:  # pragma: no cover - extremely rare EXIF issues
        logger.warning("Could not apply EXIF orientation for %s: %s", path, exc)
    return img


def _has_transparency(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        return True
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False


def split_rgb_alpha(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert any supported PIL mode to an RGB uint8 array plus optional alpha.

    Handles RGB, RGBA, grayscale (L, LA), palette (P) and 16/32-bit integer
    or float single-channel images (scaled to 8 bit).

    Returns:
        ``(rgb, alpha)`` where ``rgb`` is ``(H, W, 3)`` uint8 and ``alpha`` is
        ``(H, W)`` uint8 or ``None`` if the image has no alpha channel.
    """
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)
        gray = Image.fromarray(arr.astype(np.uint8), mode="L")
        return np.asarray(gray.convert("RGB"), dtype=np.uint8), None

    if _has_transparency(img):
        rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
        return np.ascontiguousarray(rgba[..., :3]), np.ascontiguousarray(rgba[..., 3])

    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return rgb, None


# ---------------------------------------------------------------------------
# Mask + compositing
# ---------------------------------------------------------------------------
def build_foreground_mask(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    cfg: PreprocessingConfig,
) -> tuple[np.ndarray | None, str, bool]:
    """Build a boolean foreground mask.

    Returns:
        ``(mask, source, reliable)``. ``mask`` is ``None`` when no mask can be
        produced under the configured ``mask_mode``. Alpha-derived masks are
        always reliable; thresholded masks are reliable only when the
        foreground fraction is inside the configured range.
    """
    use_alpha = alpha is not None and cfg.mask_mode in ("alpha", "auto")
    if use_alpha:
        mask = alpha >= int(cfg.alpha_threshold)
        return mask, "alpha", True

    if cfg.mask_mode == "alpha":
        return None, "none", False

    # Threshold against the background colour.
    bg = np.asarray(cfg.background_color, dtype=np.int16).reshape(1, 1, 3)
    diff = np.abs(rgb.astype(np.int16) - bg).max(axis=-1)
    mask = diff > int(cfg.mask_background_threshold)
    frac = float(mask.mean()) if mask.size else 0.0
    reliable = float(cfg.mask_min_foreground_fraction) <= frac <= float(cfg.mask_max_foreground_fraction)
    if not reliable:
        logger.debug(
            "Background-threshold mask rejected: foreground fraction %.4f outside [%.4f, %.4f]",
            frac,
            cfg.mask_min_foreground_fraction,
            cfg.mask_max_foreground_fraction,
        )
    return mask, "background_threshold", reliable


def composite_on_background(rgb: np.ndarray, alpha: np.ndarray | None, background: list[int]) -> np.ndarray:
    """Alpha-composite ``rgb`` over a solid background colour."""
    if alpha is None:
        return rgb
    a = alpha.astype(np.float32)[..., None] / 255.0
    bg = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    out = rgb.astype(np.float32) * a + bg * (1.0 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------
def compute_crop_box(
    shape: tuple[int, int],
    mask: np.ndarray | None,
    mask_reliable: bool,
    cfg: PreprocessingConfig,
) -> tuple[int, int, int, int] | None:
    """Compute a crop box ``(left, top, right, bottom)`` for the configured mode.

    The box may extend outside the image for ``foreground_bbox``; callers must
    pad with the background colour. Returns ``None`` when no crop is applied.
    """
    h, w = shape
    if cfg.crop_mode == "none":
        return None
    if cfg.crop_mode == "center_crop":
        side = min(h, w)
        left = (w - side) // 2
        top = (h - side) // 2
        return left, top, left + side, top + side
    if cfg.crop_mode == "foreground_bbox":
        if mask is None or not mask_reliable or not mask.any():
            logger.warning("foreground_bbox requested but no reliable foreground mask; falling back to no crop")
            return None
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        bw, bh = x1 - x0, y1 - y0
        side = max(bw, bh)
        pad = int(round(side * float(cfg.foreground_padding)))
        side_padded = side + 2 * pad
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        left = int(round(cx - side_padded / 2.0))
        top = int(round(cy - side_padded / 2.0))
        return left, top, left + side_padded, top + side_padded
    raise ValueError(f"Unknown crop_mode {cfg.crop_mode!r}")


def _crop_with_padding(arr: np.ndarray, box: tuple[int, int, int, int], fill: Any) -> np.ndarray:
    """Crop ``arr`` to ``box``; areas outside the array are filled with ``fill``."""
    left, top, right, bottom = box
    h, w = arr.shape[:2]
    out_shape = (bottom - top, right - left) + arr.shape[2:]
    out = np.empty(out_shape, dtype=arr.dtype)
    out[...] = fill
    src_x0, src_y0 = max(left, 0), max(top, 0)
    src_x1, src_y1 = min(right, w), min(bottom, h)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_y0, dst_x0 = src_y0 - top, src_x0 - left
        out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = arr[
            src_y0:src_y1, src_x0:src_x1
        ]
    return out


# ---------------------------------------------------------------------------
# Fit to canvas
# ---------------------------------------------------------------------------
def fit_to_canvas(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    cfg: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Resize (aspect-preserving) and centre the image on a square canvas.

    The mask, when present, is resized with the same geometry and re-binarised.
    """
    size = int(cfg.canvas_size)
    h, w = rgb.shape[:2]
    scale = size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resample = _RESAMPLE[cfg.resample]

    pil_rgb = Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), resample=resample)
    canvas = Image.new("RGB", (size, size), tuple(int(c) for c in cfg.background_color))
    off_x = (size - new_w) // 2
    off_y = (size - new_h) // 2
    canvas.paste(pil_rgb, (off_x, off_y))
    out_rgb = np.asarray(canvas, dtype=np.uint8)

    out_mask: np.ndarray | None = None
    if mask is not None:
        pil_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
            (new_w, new_h), resample=Image.Resampling.BILINEAR
        )
        mask_canvas = Image.new("L", (size, size), 0)
        mask_canvas.paste(pil_mask, (off_x, off_y))
        out_mask = np.asarray(mask_canvas, dtype=np.uint8) >= 128

    info = {
        "scale": scale,
        "resized_size": [new_w, new_h],
        "offset": [off_x, off_y],
        "canvas_size": size,
    }
    return out_rgb, out_mask, info


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess_pil(img: Image.Image, cfg: PreprocessingConfig, name: str = "<image>") -> PreprocessedImage:
    """Run the full preprocessing pipeline on an already-opened PIL image."""
    meta: dict[str, Any] = {"name": name, "original_mode": img.mode, "original_size": list(img.size)}
    rgb, alpha = split_rgb_alpha(img)
    meta["has_alpha"] = alpha is not None

    mask, mask_source, mask_reliable = build_foreground_mask(rgb, alpha, cfg)
    rgb = composite_on_background(rgb, alpha, cfg.background_color)

    box = compute_crop_box(rgb.shape[:2], mask, mask_reliable, cfg)
    meta["crop_mode"] = cfg.crop_mode
    meta["crop_box"] = list(box) if box is not None else None
    if box is not None:
        rgb = _crop_with_padding(rgb, box, np.asarray(cfg.background_color, dtype=np.uint8))
        if mask is not None:
            mask = _crop_with_padding(mask, box, False)

    rgb, mask, fit_info = fit_to_canvas(rgb, mask, cfg)
    meta.update(fit_info)
    if mask is not None:
        meta["foreground_fraction"] = float(mask.mean())

    return PreprocessedImage(rgb=rgb, mask=mask, mask_source=mask_source, mask_reliable=mask_reliable, meta=meta)


def preprocess_image(path: str | Path, cfg: PreprocessingConfig) -> PreprocessedImage:
    """Load ``path`` and run :func:`preprocess_pil` on it."""
    path = Path(path)
    img = load_image(path)
    result = preprocess_pil(img, cfg, name=path.name)
    result.meta["path"] = str(path)
    logger.debug(
        "Preprocessed %s: %s %s -> canvas %d, mask=%s (reliable=%s)",
        path.name,
        result.meta["original_mode"],
        result.meta["original_size"],
        cfg.canvas_size,
        result.mask_source,
        result.mask_reliable,
    )
    return result


def save_preprocessed(image: PreprocessedImage, directory: str | Path, name: str) -> Path:
    """Save the preprocessed RGB image (and mask, if any) to ``directory``.

    Returns the path of the saved RGB PNG.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    rgb_path = directory / f"{stem}.png"
    image.to_pil().save(rgb_path)
    if image.mask is not None:
        mask_img = image.mask_to_pil()
        if mask_img is not None:
            mask_img.save(directory / f"{stem}_mask.png")
    return rgb_path
