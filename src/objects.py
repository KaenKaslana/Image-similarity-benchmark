"""Multi-object synthetic test cases.

Several *different* objects are assembled from simple 3-D primitives and
rendered as orthographic front / side / top views. Every candidate carries a
different kind of deviation (identical, missing part, proportion change,
colour change + added part, misalignment) so each metric can be checked one
effect at a time.

Object space: ``x`` = right, ``y`` = depth (away from the front camera),
``z`` = up. All coordinates and sizes are fractions of the canvas side; an
object should fit in roughly ``[-0.45, 0.45]``.

Primitive dictionaries::

    {"kind": "box",    "center": (x, y, z), "size": (sx, sy, sz), "color": (r, g, b)}
    {"kind": "cyl_z",  "center": (x, y, z), "radius": r, "height": h, "color": ...}  # vertical cylinder
    {"kind": "cyl_x",  "center": (x, y, z), "radius": r, "length": l, "color": ...}  # cylinder along x
    {"kind": "sphere", "center": (x, y, z), "radius": r, "color": ...}
    {"kind": "prism",  "center": (x, y, z), "size": (sx, sy, sz), "color": ...}      # roof, ridge along x
    {"kind": "cone",   "center": (x, y, z), "radius": r, "top_radius": rt, "height": h, "color": ...}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

Primitive = dict[str, Any]
VIEWS: tuple[str, ...] = ("front", "side", "top")

BLUE = (90, 120, 200)
ORANGE = (220, 140, 60)
GREEN = (80, 170, 100)
RED = (200, 70, 60)
GREY = (140, 140, 150)
BROWN = (150, 100, 60)
YELLOW = (230, 200, 70)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def _extent(prim: Primitive) -> tuple[float, float, float]:
    kind = prim["kind"]
    if kind in ("box", "prism"):
        return tuple(prim["size"])  # type: ignore[return-value]
    if kind == "sphere":
        r = prim["radius"]
        return 2 * r, 2 * r, 2 * r
    if kind == "cyl_z":
        r = prim["radius"]
        return 2 * r, 2 * r, prim["height"]
    if kind == "cyl_x":
        r = prim["radius"]
        return prim["length"], 2 * r, 2 * r
    if kind == "cone":
        r = prim["radius"]
        return 2 * r, 2 * r, prim["height"]
    raise ValueError(f"unknown primitive kind {kind!r}")


def project_primitive(prim: Primitive, view: str) -> tuple[str, tuple[float, ...], float]:
    """Project one primitive into a view.

    Returns ``(shape, params, depth)``: ``shape`` is ``rect`` / ``ellipse`` /
    ``polygon``; ``params`` are 2-D object-space coordinates ``(u, v)`` with
    ``u`` to the right and ``v`` up on screen; ``depth`` is larger for
    primitives closer to the camera (painter's algorithm).
    """
    x, y, z = prim["center"]
    kind = prim["kind"]
    sx, sy, sz = _extent(prim)

    if view == "front":  # camera looks along +y
        u, v, depth, w, h = x, z, -y, sx, sz
        circle = kind == "sphere"
    elif view == "side":  # camera at +x looking along -x
        u, v, depth, w, h = y, z, x, sy, sz
        circle = kind in ("sphere", "cyl_x")
    elif view == "top":  # camera above looking down; front of the object at the bottom of the image
        u, v, depth, w, h = x, -y, z, sx, sy
        circle = kind in ("sphere", "cyl_z", "cone")
    else:
        raise ValueError(f"unknown view {view!r}")

    if circle:
        return "ellipse", (u - w / 2, v - h / 2, u + w / 2, v + h / 2), depth
    if kind == "prism" and view == "front":
        return "polygon", (u - w / 2, v - h / 2, u + w / 2, v - h / 2, u, v + h / 2), depth
    if kind == "cone" and view in ("front", "side"):
        rt = prim["top_radius"]
        return "polygon", (u - w / 2, v - h / 2, u + w / 2, v - h / 2, u + rt, v + h / 2, u - rt, v + h / 2), depth
    return "rect", (u - w / 2, v - h / 2, u + w / 2, v + h / 2), depth


def render_object_views(
    primitives: list[Primitive],
    size: int = 512,
    mode: str = "RGBA",
    background: tuple[int, int, int] = (255, 255, 255),
    seed: int | None = 0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Image.Image]:
    """Render front / side / top orthographic views of a primitive assembly.

    Args:
        primitives: list of primitive dicts (see module docstring).
        size: canvas side in pixels.
        mode: ``RGBA`` (transparent background), ``RGB`` or ``L``.
        background: fill colour for non-transparent modes.
        seed: deterministic texture noise inside the object (``None`` = flat).
        offset: ``(dx, dy)`` shift of the whole object in every view, as a
            fraction of the canvas; used to build a misaligned candidate.

    Returns:
        ``{"front.png": img, "side.png": img, "top.png": img}``.
    """
    s = size
    cx, cy = s / 2 + offset[0] * s, s / 2 + offset[1] * s
    out: dict[str, Image.Image] = {}
    for view in VIEWS:
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        projected = [(project_primitive(p, view), tuple(p["color"])) for p in primitives]
        projected.sort(key=lambda item: item[0][2])  # far first, near last
        for (shape, params, _depth), color in projected:
            px = [cx + params[i] * s if i % 2 == 0 else cy - params[i] * s for i in range(len(params))]
            fill = (*color, 255)
            if shape in ("rect", "ellipse"):
                x0, y0, x1, y1 = px
                box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                (draw.rectangle if shape == "rect" else draw.ellipse)(box, fill=fill)
            else:
                draw.polygon(list(zip(px[0::2], px[1::2])), fill=fill)
        if seed is not None:
            rng = np.random.default_rng(seed)
            arr = np.asarray(img).astype(np.int16)
            noise = rng.integers(-18, 19, size=(s, s, 1), dtype=np.int16)
            inside = arr[..., 3:4] > 0
            arr[..., :3] = np.clip(arr[..., :3] + noise * inside, 0, 255)
            img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        if mode != "RGBA":
            solid = Image.new("RGBA", (s, s), (*background, 255))
            solid.alpha_composite(img)
            img = solid.convert("RGB" if mode == "RGB" else "L")
        out[f"{view}.png"] = img
    return out


# ---------------------------------------------------------------------------
# Object definitions
# ---------------------------------------------------------------------------
def mug() -> list[Primitive]:
    return [
        {"kind": "cyl_z", "center": (0.0, 0.0, 0.0), "radius": 0.16, "height": 0.40, "color": BLUE},
        {"kind": "cyl_x", "center": (0.24, 0.0, 0.05), "radius": 0.04, "length": 0.16, "color": BLUE},  # handle
    ]


def table(top_thickness: float = 0.06, leg_height: float = 0.36) -> list[Primitive]:
    top = {"kind": "box", "center": (0.0, 0.0, 0.03), "size": (0.72, 0.40, top_thickness), "color": BROWN}
    legs = [
        {"kind": "box", "center": (sx * 0.30, sy * 0.16, -leg_height / 2), "size": (0.05, 0.05, leg_height), "color": BROWN}
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]
    return [top] + legs


def bottle(neck_radius: float = 0.05, neck_height: float = 0.14) -> list[Primitive]:
    neck_z = 0.23 + neck_height / 2
    cap_z = 0.23 + neck_height + 0.03
    return [
        {"kind": "cyl_z", "center": (0.0, 0.0, -0.10), "radius": 0.13, "height": 0.50, "color": GREEN},
        {"kind": "cone", "center": (0.0, 0.0, 0.19), "radius": 0.13, "top_radius": neck_radius, "height": 0.08, "color": GREEN},
        {"kind": "cyl_z", "center": (0.0, 0.0, neck_z), "radius": neck_radius, "height": neck_height, "color": GREEN},
        {"kind": "cyl_z", "center": (0.0, 0.0, cap_z), "radius": neck_radius + 0.01, "height": 0.06, "color": RED},
    ]


def house(roof_color: tuple[int, int, int] = RED, chimney: bool = False) -> list[Primitive]:
    prims = [
        {"kind": "box", "center": (0.0, 0.0, -0.13), "size": (0.60, 0.44, 0.40), "color": YELLOW},
        {"kind": "prism", "center": (0.0, 0.0, 0.19), "size": (0.68, 0.50, 0.24), "color": roof_color},
        {"kind": "box", "center": (-0.10, -0.23, -0.23), "size": (0.10, 0.02, 0.20), "color": BROWN},  # door
    ]
    if chimney:
        prims.append({"kind": "box", "center": (0.18, 0.0, 0.26), "size": (0.08, 0.08, 0.20), "color": GREY})
    return prims


def robot() -> list[Primitive]:
    return [
        {"kind": "box", "center": (0.0, 0.0, 0.0), "size": (0.30, 0.18, 0.34), "color": GREY},
        {"kind": "sphere", "center": (0.0, 0.0, 0.27), "radius": 0.10, "color": GREY},
        {"kind": "box", "center": (-0.21, 0.0, 0.04), "size": (0.08, 0.08, 0.28), "color": ORANGE},
        {"kind": "box", "center": (0.21, 0.0, 0.04), "size": (0.08, 0.08, 0.28), "color": ORANGE},
        {"kind": "box", "center": (-0.08, 0.0, -0.31), "size": (0.10, 0.12, 0.28), "color": ORANGE},
        {"kind": "box", "center": (0.08, 0.0, -0.31), "size": (0.10, 0.12, 0.28), "color": ORANGE},
    ]


def chair() -> list[Primitive]:
    legs = [
        {"kind": "box", "center": (sx * 0.13, sy * 0.13, -0.24), "size": (0.04, 0.04, 0.30), "color": BROWN}
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]
    return [
        {"kind": "box", "center": (0.0, 0.0, -0.07), "size": (0.34, 0.34, 0.04), "color": BROWN},  # seat
        {"kind": "box", "center": (0.0, 0.15, 0.15), "size": (0.34, 0.04, 0.40), "color": RED},  # backrest
    ] + legs


def lamp() -> list[Primitive]:
    return [
        {"kind": "cyl_z", "center": (0.0, 0.0, -0.40), "radius": 0.16, "height": 0.04, "color": GREY},
        {"kind": "cyl_z", "center": (0.0, 0.0, -0.10), "radius": 0.02, "height": 0.56, "color": GREY},
        {"kind": "cone", "center": (0.0, 0.0, 0.28), "radius": 0.22, "top_radius": 0.10, "height": 0.20, "color": YELLOW},
    ]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def object_test_cases() -> dict[str, dict[str, Any]]:
    """Built-in multi-object test cases.

    Each entry: ``{"reference": prims, "candidate": prims, "offset": (dx, dy),
    "expect": str}``. All objects differ from each other and every candidate
    carries a different kind of deviation.
    """
    return {
        "robot": {
            "reference": robot(),
            "candidate": robot(),
            "offset": (0.0, 0.0),
            "expect": "identical object -> every view scores ~100",
        },
        "mug": {
            "reference": mug(),
            "candidate": mug()[:1],
            "offset": (0.0, 0.0),
            "expect": "missing part (handle) -> front & top silhouette/edge drop, side almost unchanged",
        },
        "table": {
            "reference": table(),
            "candidate": table(top_thickness=0.12, leg_height=0.24),
            "offset": (0.0, 0.0),
            "expect": "proportion change (thick top, short legs) -> front & side drop, top unchanged",
        },
        "bottle": {
            "reference": bottle(),
            "candidate": bottle(neck_radius=0.035, neck_height=0.26),
            "offset": (0.0, 0.0),
            "expect": "taller, thinner neck -> front & side drop moderately, top nearly unchanged",
        },
        "house": {
            "reference": house(),
            "candidate": house(roof_color=BLUE, chimney=True),
            "offset": (0.0, 0.0),
            "expect": "colour change + added chimney -> LPIPS/SSIM drop everywhere, silhouette drops a little",
        },
        "lamp": {
            "reference": lamp(),
            "candidate": lamp(),
            "offset": (0.05, 0.03),
            "expect": "same object shifted by (5 %, 3 %) -> all metrics drop; --crop-mode foreground_bbox recovers it",
        },
        "mismatch": {
            "reference": chair(),
            "candidate": lamp(),
            "offset": (0.0, 0.0),
            "expect": "two completely different objects (chair vs lamp) -> low silhouette/edge in every view, clearly the worst group",
        },
    }


def write_object_set(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    size: int = 512,
    mode: str = "RGBA",
    objects: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Write ``<object>_<view>.png`` pairs for several objects.

    Returns the file names written (identical in both folders).
    """
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    objects = objects or object_test_cases()
    names: list[str] = []
    for obj_name, spec in objects.items():
        ref_views = render_object_views(spec["reference"], size=size, mode=mode)
        cand_views = render_object_views(spec["candidate"], size=size, mode=mode, offset=spec.get("offset", (0.0, 0.0)))
        for view_file, img in ref_views.items():
            name = f"{obj_name}_{view_file}"
            img.save(reference_dir / name)
            cand_views[view_file].save(candidate_dir / name)
            names.append(name)
    return names
