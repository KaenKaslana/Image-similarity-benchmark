"""Synthetic test-image generation.

Used by the test-suite and by ``scripts/make_sample_data.py`` so the project
can be exercised without any hand-made images. Images are simple coloured
shapes on a transparent (RGBA) or white (RGB / grayscale) background.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

Shape = Literal["circle", "square", "triangle", "ellipse"]


def draw_shape(
    size: int = 256,
    shape: Shape = "circle",
    color: tuple[int, int, int] = (200, 60, 40),
    offset: tuple[int, int] = (0, 0),
    scale: float = 0.6,
    mode: str = "RGBA",
    background: tuple[int, int, int] = (255, 255, 255),
    seed: int | None = 0,
) -> Image.Image:
    """Draw a single shape on a square canvas.

    Args:
        size: canvas side in pixels.
        shape: geometry to draw.
        color: fill colour (RGB).
        offset: ``(dx, dy)`` translation of the shape centre in pixels.
        scale: shape diameter relative to ``size``.
        mode: ``"RGBA"`` (transparent background), ``"RGB"`` (solid
            ``background``) or ``"L"`` (grayscale on solid background).
        background: background colour for non-transparent modes.
        seed: when not ``None`` a little deterministic texture noise is added
            inside the shape so SSIM/LPIPS have structure to work with.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size / 2 + offset[0], size / 2 + offset[1]
    r = size * scale / 2
    box = (cx - r, cy - r, cx + r, cy + r)
    fill = (*color, 255)
    if shape == "circle":
        draw.ellipse(box, fill=fill)
    elif shape == "ellipse":
        draw.ellipse((cx - r, cy - r * 0.6, cx + r, cy + r * 0.6), fill=fill)
    elif shape == "square":
        draw.rectangle(box, fill=fill)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
    else:
        raise ValueError(f"unknown shape {shape!r}")

    if seed is not None:
        rng = np.random.default_rng(seed)
        arr = np.asarray(img).astype(np.int16)
        noise = rng.integers(-25, 26, size=(size, size, 1), dtype=np.int16)
        inside = arr[..., 3:4] > 0
        arr[..., :3] = np.clip(arr[..., :3] + noise * inside, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

    if mode == "RGBA":
        return img
    solid = Image.new("RGBA", (size, size), (*background, 255))
    solid.alpha_composite(img)
    if mode == "RGB":
        return solid.convert("RGB")
    if mode == "L":
        return solid.convert("L")
    raise ValueError(f"unsupported mode {mode!r}")


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Multiply brightness of the colour channels, preserving alpha."""
    if img.mode == "RGBA":
        rgb = ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)
        rgb.putalpha(img.getchannel("A"))
        return rgb
    return ImageEnhance.Brightness(img).enhance(factor)


def write_sample_set(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    size: int = 256,
    mode: str = "RGBA",
) -> list[str]:
    """Write a small paired dataset (identical, shifted, recoloured, reshaped).

    Returns the list of file names written to both folders.
    """
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "front.png": (draw_shape(size, "circle", mode=mode), draw_shape(size, "circle", mode=mode)),
        "side.png": (draw_shape(size, "square", mode=mode), draw_shape(size, "square", offset=(12, 6), mode=mode)),
        "top.png": (draw_shape(size, "triangle", mode=mode), draw_shape(size, "ellipse", mode=mode)),
        "back.png": (
            draw_shape(size, "circle", color=(40, 90, 200), mode=mode),
            draw_shape(size, "circle", color=(40, 180, 90), mode=mode),
        ),
    }
    for name, (ref, cand) in specs.items():
        ref.save(reference_dir / name)
        cand.save(candidate_dir / name)
    return list(specs)


# ---------------------------------------------------------------------------
# Orthographic three-view drawings of a simple object
# ---------------------------------------------------------------------------
def draw_three_views(
    size: int = 512,
    width: float = 0.50,
    depth: float = 0.30,
    height: float = 0.35,
    cyl_radius: float = 0.10,
    cyl_height: float = 0.18,
    body_color: tuple[int, int, int] = (90, 120, 200),
    cyl_color: tuple[int, int, int] = (220, 140, 60),
    mode: str = "RGBA",
    background: tuple[int, int, int] = (255, 255, 255),
    seed: int | None = 0,
) -> dict[str, Image.Image]:
    """Render front / side / top orthographic views of a box with a cylinder on top.

    All dimensions are fractions of ``size``. The object is centred in every
    view, so the three images are consistent with one another the way a
    proper three-view drawing is: front shows ``width x height``, side shows
    ``depth x height`` and top shows ``width x depth`` with the cylinder as a
    circle.

    Returns:
        ``{"front.png": img, "side.png": img, "top.png": img}``.
    """
    s = size
    cx = cy = s / 2

    def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        return img, ImageDraw.Draw(img)

    total_h = height + cyl_height
    # Front view: box (width x height) with cylinder rectangle (2r x cyl_height) on top.
    front, d = _canvas()
    top_y = cy - total_h * s / 2
    box_top = top_y + cyl_height * s
    d.rectangle((cx - width * s / 2, box_top, cx + width * s / 2, box_top + height * s), fill=(*body_color, 255))
    d.rectangle((cx - cyl_radius * s, top_y, cx + cyl_radius * s, box_top), fill=(*cyl_color, 255))

    # Side view: box (depth x height) with the same cylinder rectangle.
    side, d = _canvas()
    d.rectangle((cx - depth * s / 2, box_top, cx + depth * s / 2, box_top + height * s), fill=(*body_color, 255))
    d.rectangle((cx - cyl_radius * s, top_y, cx + cyl_radius * s, box_top), fill=(*cyl_color, 255))

    # Top view: box (width x depth) with the cylinder as a circle.
    top, d = _canvas()
    d.rectangle((cx - width * s / 2, cy - depth * s / 2, cx + width * s / 2, cy + depth * s / 2), fill=(*body_color, 255))
    d.ellipse((cx - cyl_radius * s, cy - cyl_radius * s, cx + cyl_radius * s, cy + cyl_radius * s), fill=(*cyl_color, 255))

    views = {"front.png": front, "side.png": side, "top.png": top}
    out: dict[str, Image.Image] = {}
    for name, img in views.items():
        if seed is not None:
            rng = np.random.default_rng(seed)
            arr = np.asarray(img).astype(np.int16)
            noise = rng.integers(-18, 19, size=(s, s, 1), dtype=np.int16)
            inside = arr[..., 3:4] > 0
            arr[..., :3] = np.clip(arr[..., :3] + noise * inside, 0, 255)
            img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        if mode == "RGBA":
            out[name] = img
        else:
            solid = Image.new("RGBA", (s, s), (*background, 255))
            solid.alpha_composite(img)
            out[name] = solid.convert("RGB" if mode == "RGB" else "L")
    return out


def write_three_view_set(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    size: int = 512,
    mode: str = "RGBA",
) -> list[str]:
    """Write a three-view (front/side/top) reference set and a slightly
    different candidate object (wider box, smaller and shorter cylinder).

    The geometric deviation shows up consistently across the views: the wider
    box affects front and top, the cylinder change affects all three.
    """
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    ref = draw_three_views(size, mode=mode)
    cand = draw_three_views(size, width=0.56, cyl_radius=0.08, cyl_height=0.14, mode=mode)
    for name in ref:
        ref[name].save(reference_dir / name)
        cand[name].save(candidate_dir / name)
    return list(ref)
