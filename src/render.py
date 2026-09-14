"""Orthographic multi-view rendering of 3D meshes.

The renderer is a small, dependency-light software rasteriser built on numpy.
It deliberately avoids OpenGL / pyrender / Blender so that it behaves the same
on Windows, Linux and macOS (including headless machines).

Pipeline for one model:

1. Load the file with :mod:`trimesh` and merge every part into one mesh
   (node transforms are applied).
2. Rotate it into a canonical frame (``+Y`` up, ``+Z`` front) according to
   the ``up`` / ``front`` options.
3. Centre the mesh on its bounding box and scale it so that its **largest**
   extent equals one unit. The same scale is used for every view, so the
   relative size of the views is preserved.
4. For each requested view, project orthographically onto a square canvas,
   rasterise with a z-buffer and flat headlight shading, and write an RGBA
   PNG (transparent background, so the benchmark can build an exact
   silhouette mask from the alpha channel).

Both models of a comparison must be rendered with the **same**
:class:`RenderOptions`; :func:`render_views` records them in ``views.json``
next to the images so a run can be reproduced.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".stl", ".ply", ".off", ".3mf", ".dae", ".zip")

_AXES = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}
AXIS_NAMES = tuple(_AXES)

# Camera definitions in the canonical frame (+Y up, +Z front), as
# (forward direction the camera looks along, image-up direction). They follow
# third-angle projection: the top view shows the front of the object at the
# bottom of the image, the right side view shows the front on the left.
# ``iso`` is a three-quarter view from front-right-above, useful as the
# single "hero" image handed to an image-to-3D generator.
_ISO_FORWARD = -np.array([1.0, 0.8, 1.0]) / np.linalg.norm([1.0, 0.8, 1.0])
VIEWS: dict[str, tuple[Any, Any]] = {
    "front": ("-z", "+y"),
    "back": ("+z", "+y"),
    "side": ("-x", "+y"),  # right side view
    "left": ("+x", "+y"),
    "top": ("-y", "-z"),
    "bottom": ("+y", "+z"),
    "iso": (_ISO_FORWARD, "+y"),
}
DEFAULT_VIEWS = ("front", "side", "top")
ORTHO_VIEWS = ("front", "back", "side", "left", "top", "bottom")
STYLES = ("shaded", "silhouette")


class RenderError(RuntimeError):
    """Raised when a model cannot be loaded or rendered."""


@dataclass
class RenderOptions:
    """Rendering settings shared by every model of a comparison."""

    size: int = 512
    views: tuple[str, ...] = DEFAULT_VIEWS
    up: str = "+y"
    front: str = "+z"
    fill: float = 0.85
    style: str = "shaded"
    ambient: float = 0.35
    supersample: int = 2
    gray_min: int = 60
    gray_max: int = 210

    def validate(self) -> None:
        if self.size < 16:
            raise RenderError("size must be >= 16")
        if not self.views:
            raise RenderError("at least one view is required")
        for v in self.views:
            if v not in VIEWS:
                raise RenderError(f"unknown view {v!r}; choose from {', '.join(VIEWS)}")
        if self.up not in _AXES or self.front not in _AXES:
            raise RenderError(f"up/front must be one of {', '.join(AXIS_NAMES)}")
        if abs(float(_AXES[self.up] @ _AXES[self.front])) > 1e-9:
            raise RenderError(f"up ({self.up}) and front ({self.front}) must be perpendicular")
        if not 0.1 <= self.fill <= 1.0:
            raise RenderError("fill must be in [0.1, 1.0]")
        if self.style not in STYLES:
            raise RenderError(f"style must be one of {', '.join(STYLES)}")
        if not 0.0 <= self.ambient <= 1.0:
            raise RenderError("ambient must be in [0, 1]")
        if self.supersample < 1:
            raise RenderError("supersample must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["views"] = list(self.views)
        return d


@dataclass
class LoadedMesh:
    """A mesh in the canonical frame, centred and scaled to unit max extent."""

    vertices: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int64
    face_normals: np.ndarray  # (F, 3) float64, unit length
    source: Path
    original_extents: tuple[float, float, float]
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading and normalisation
# ---------------------------------------------------------------------------
def all_orientations() -> list[tuple[str, str]]:
    """The 24 right-handed ``(up, front)`` axis pairs."""
    return [(u, f) for u in AXIS_NAMES for f in AXIS_NAMES if abs(float(_AXES[u] @ _AXES[f])) < 1e-9]


def reorient(mesh: LoadedMesh, up: str, front: str) -> LoadedMesh:
    """Return a copy of an already-canonical mesh re-interpreted with a new up/front.

    The mesh was loaded with some ``(up0, front0)``; this applies the extra
    rotation that ``(up, front)`` would have produced relative to the
    identity, then re-normalises centre and scale (the bounding box changes
    with rotation).
    """
    return rotate(mesh, canonical_rotation(up, front), {"reoriented": {"up": up, "front": front}})


def yaw_matrix(degrees: float) -> np.ndarray:
    """Rotation about the canonical up axis (+Y); positive turns the front towards +X."""
    t = np.deg2rad(degrees)
    c, s_ = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]], dtype=np.float64)


def rotate(mesh: LoadedMesh, matrix: np.ndarray, note: dict[str, Any] | None = None) -> LoadedMesh:
    """Apply a rotation matrix to a canonical mesh and re-normalise centre/scale."""
    vertices = mesh.vertices @ matrix.T
    normals = mesh.face_normals @ matrix.T
    lo, hi = vertices.min(0), vertices.max(0)
    vertices = (vertices - (lo + hi) / 2.0) / float((hi - lo).max())
    meta = dict(mesh.meta)
    if note:
        meta.update(note)
    return LoadedMesh(vertices, mesh.faces, normals, mesh.source, mesh.original_extents, meta)


def canonical_rotation(up: str, front: str) -> np.ndarray:
    """Rotation matrix ``R`` such that ``R @ p`` maps ``up`` to +Y and ``front`` to +Z."""
    y = _AXES[up]
    z = _AXES[front]
    x = np.cross(y, z)
    return np.stack([x, y, z]).astype(np.float64)


def load_mesh(path: str | Path, up: str = "+y", front: str = "+z") -> LoadedMesh:
    """Load any format trimesh understands and normalise it.

    Raises:
        RenderError: if the file is missing, cannot be parsed, or contains no
            triangles (e.g. a point cloud).
    """
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RenderError("trimesh is required for rendering: pip install trimesh") from exc

    path = Path(path)
    if not path.is_file():
        raise RenderError(f"model file not found: {path}")
    try:
        loaded = trimesh.load(str(path), force="mesh")
    except Exception as exc:  # trimesh raises many different types
        raise RenderError(f"could not load {path.name}: {exc}") from exc
    if not isinstance(loaded, trimesh.Trimesh) or loaded.faces is None or len(loaded.faces) == 0:
        raise RenderError(f"{path.name} contains no triangle geometry (point cloud or empty scene?)")

    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    original_extents = tuple(float(v) for v in (vertices.max(0) - vertices.min(0)))

    rot = canonical_rotation(up, front)
    vertices = vertices @ rot.T
    normals = np.asarray(loaded.face_normals, dtype=np.float64) @ rot.T

    lo, hi = vertices.min(0), vertices.max(0)
    centre = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    if not np.isfinite(extent) or extent <= 0.0:
        raise RenderError(f"{path.name} has a degenerate (zero-size) bounding box")
    vertices = (vertices - centre) / extent

    meta = {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "original_extents": list(original_extents),
        "scale": extent,
    }
    logger.info("Loaded %s: %d vertices, %d faces", path.name, len(vertices), len(faces))
    return LoadedMesh(vertices, faces, normals, path, original_extents, meta)


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------
def rasterize(
    tri_xy: np.ndarray, tri_depth: np.ndarray, tri_value: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer rasterisation of flat-coloured triangles.

    Args:
        tri_xy: ``(N, 3, 2)`` pixel coordinates of the triangle corners
            (pixel ``i`` covers ``[i, i+1)``, its centre is ``i + 0.5``).
        tri_depth: ``(N, 3)`` depth at each corner; **smaller is closer**.
        tri_value: ``(N,)`` value written for pixels covered by each triangle.
        size: canvas is ``size x size``.

    Returns:
        ``(depth, value)`` arrays of shape ``(size, size)``. ``depth`` is
        ``+inf`` where nothing was drawn.

    Triangles are bucketed by bounding-box size so that all triangles of a
    bucket are rasterised with one vectorised numpy pass; only the pixels
    inside each triangle survive to the z-buffer merge.
    """
    zbuf = np.full(size * size, np.inf, dtype=np.float64)
    vbuf = np.zeros(size * size, dtype=np.float32)
    if len(tri_xy) == 0:
        return zbuf.reshape(size, size), vbuf.reshape(size, size)

    xy = np.asarray(tri_xy, dtype=np.float64)
    depth = np.asarray(tri_depth, dtype=np.float64)
    value = np.asarray(tri_value, dtype=np.float32)

    ax, ay = xy[:, 0, 0], xy[:, 0, 1]
    bx, by = xy[:, 1, 0], xy[:, 1, 1]
    cx, cy = xy[:, 2, 0], xy[:, 2, 1]
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    # Pixel index range whose centres can lie inside the triangle.
    xmin = np.ceil(xy[:, :, 0].min(1) - 0.5).astype(np.int64)
    xmax = np.floor(xy[:, :, 0].max(1) - 0.5).astype(np.int64)
    ymin = np.ceil(xy[:, :, 1].min(1) - 0.5).astype(np.int64)
    ymax = np.floor(xy[:, :, 1].max(1) - 0.5).astype(np.int64)
    np.clip(xmin, 0, size - 1, out=xmin)
    np.clip(xmax, 0, size - 1, out=xmax)
    np.clip(ymin, 0, size - 1, out=ymin)
    np.clip(ymax, 0, size - 1, out=ymax)
    keep = (xmax >= xmin) & (ymax >= ymin) & (np.abs(area) > 1e-12)
    keep &= (xy[:, :, 0].max(1) >= 0) & (xy[:, :, 0].min(1) <= size)
    keep &= (xy[:, :, 1].max(1) >= 0) & (xy[:, :, 1].min(1) <= size)
    idx_all = np.nonzero(keep)[0]
    if len(idx_all) == 0:
        return zbuf.reshape(size, size), vbuf.reshape(size, size)

    span = np.maximum(xmax - xmin, ymax - ymin)[idx_all] + 1
    tier = np.ceil(np.log2(span)).astype(np.int64)  # bucket K = 2**tier >= span
    max_elements = 1 << 22  # per vectorised chunk

    for t in np.unique(tier):
        k = 1 << int(t)
        tris = idx_all[tier == t]
        chunk = max(1, max_elements // (k * k))
        offs = np.arange(k, dtype=np.int64)
        for start in range(0, len(tris), chunk):
            sel = tris[start : start + chunk]
            px = xmin[sel][:, None] + offs[None, :]  # (M, K)
            py = ymin[sel][:, None] + offs[None, :]
            px3 = px[:, None, :]  # (M, 1, K)
            py3 = py[:, :, None]  # (M, K, 1)
            fx = px3 + 0.5
            fy = py3 + 0.5
            s = np.sign(area[sel])[:, None, None]
            a_x, a_y = ax[sel][:, None, None], ay[sel][:, None, None]
            b_x, b_y = bx[sel][:, None, None], by[sel][:, None, None]
            c_x, c_y = cx[sel][:, None, None], cy[sel][:, None, None]
            w0 = ((c_x - b_x) * (fy - b_y) - (c_y - b_y) * (fx - b_x)) * s
            w1 = ((a_x - c_x) * (fy - c_y) - (a_y - c_y) * (fx - c_x)) * s
            w2 = ((b_x - a_x) * (fy - a_y) - (b_y - a_y) * (fx - a_x)) * s
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            inside &= (px3 <= xmax[sel][:, None, None]) & (py3 <= ymax[sel][:, None, None])
            if not inside.any():
                continue
            inv_area = 1.0 / np.abs(area[sel])[:, None, None]
            d = depth[sel]
            z = (w0 * d[:, 0, None, None] + w1 * d[:, 1, None, None] + w2 * d[:, 2, None, None]) * inv_area
            lin = np.broadcast_to(py3 * size + px3, inside.shape)
            val = np.broadcast_to(value[sel][:, None, None], inside.shape)

            lin = lin[inside]
            z = z[inside]
            val = val[inside]
            closer = z < zbuf[lin]
            if not closer.any():
                continue
            lin, z, val = lin[closer], z[closer], val[closer]
            order = np.lexsort((z, lin))
            lin, z, val = lin[order], z[order], val[order]
            first = np.empty(len(lin), dtype=bool)
            first[0] = True
            np.not_equal(lin[1:], lin[:-1], out=first[1:])
            lin, z, val = lin[first], z[first], val[first]
            zbuf[lin] = z
            vbuf[lin] = val

    return zbuf.reshape(size, size), vbuf.reshape(size, size)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def _axis(spec: Any) -> np.ndarray:
    return _AXES[spec] if isinstance(spec, str) else np.asarray(spec, dtype=np.float64)


def view_basis(view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(right, up, forward)`` orthonormal vectors of a named view."""
    forward_spec, up_spec = VIEWS[view]
    forward = _axis(forward_spec)
    up = _axis(up_spec)
    up = up - forward * float(up @ forward)  # orthogonalise (matters for iso)
    up = up / np.linalg.norm(up)
    right = np.cross(forward, up)
    return right, up, forward


def render_view(mesh: LoadedMesh, view: str, opts: RenderOptions) -> np.ndarray:
    """Render one orthographic view to an ``(size, size, 4)`` uint8 RGBA image."""
    right, up, forward = view_basis(view)
    ss = int(opts.supersample)
    size = int(opts.size) * ss
    scale = opts.fill * size  # unit extent -> fill * canvas

    v = mesh.vertices
    xs = size / 2.0 + (v @ right) * scale
    ys = size / 2.0 - (v @ up) * scale
    depth = v @ forward

    f = mesh.faces
    tri_xy = np.stack([np.stack([xs[f[:, i]], ys[f[:, i]]], axis=1) for i in range(3)], axis=1)
    tri_depth = depth[f]

    if opts.style == "silhouette":
        gray = np.zeros(len(f), dtype=np.float32)
    else:
        facing = np.abs(mesh.face_normals @ forward)
        intensity = opts.ambient + (1.0 - opts.ambient) * facing
        gray = (opts.gray_min + (opts.gray_max - opts.gray_min) * intensity).astype(np.float32)

    zbuf, vbuf = rasterize(tri_xy, tri_depth, gray, size)
    coverage = np.isfinite(zbuf).astype(np.float32)

    if ss > 1:
        out = int(opts.size)
        cov = cv2.resize(coverage, (out, out), interpolation=cv2.INTER_AREA)
        premul = cv2.resize(vbuf * coverage, (out, out), interpolation=cv2.INTER_AREA)
        gray_img = premul / np.maximum(cov, 1e-6)
    else:
        cov, gray_img = coverage, vbuf

    rgba = np.zeros((cov.shape[0], cov.shape[1], 4), dtype=np.uint8)
    g = np.clip(np.round(gray_img), 0, 255).astype(np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = g
    rgba[..., 3] = np.clip(np.round(cov * 255.0), 0, 255).astype(np.uint8)
    return rgba


def render_views(
    model: str | Path | LoadedMesh,
    output_dir: str | Path,
    opts: RenderOptions | None = None,
    prefix: str = "",
) -> dict[str, Path]:
    """Render every view in ``opts.views`` and write ``<prefix><view>.png``.

    Also writes ``views.json`` with the render options and mesh statistics.

    Returns:
        Mapping ``view name -> PNG path``.
    """
    opts = opts or RenderOptions()
    opts.validate()
    mesh = model if isinstance(model, LoadedMesh) else load_mesh(model, opts.up, opts.front)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for view in opts.views:
        rgba = render_view(mesh, view, opts)
        path = output_dir / f"{prefix}{view}.png"
        Image.fromarray(rgba, mode="RGBA").save(path)
        written[view] = path
        logger.info("Rendered %s -> %s", view, path)

    meta = {
        "model": str(mesh.source),
        "mesh": mesh.meta,
        "render": opts.to_dict(),
        "images": {k: p.name for k, p in written.items()},
    }
    with open(output_dir / "views.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return written


def is_mesh_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MESH_EXTENSIONS


def parse_views(text: str | Sequence[str]) -> tuple[str, ...]:
    """``"front,side,top"`` or ``"all"`` -> tuple of view names."""
    if isinstance(text, str):
        if text.strip().lower() == "all":
            return ORTHO_VIEWS
        parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in text]
    unknown = [p for p in parts if p not in VIEWS]
    if unknown:
        raise RenderError(f"unknown view(s) {unknown}; choose from {', '.join(VIEWS)} or 'all'")
    if not parts:
        raise RenderError("no views given")
    return tuple(dict.fromkeys(parts))
