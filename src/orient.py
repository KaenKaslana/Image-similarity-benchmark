"""Automatic orientation of a candidate mesh against a reference.

AI-generated or downloaded models rarely agree on which axis is "up" and
which side is the "front". Comparing views of two models that are rotated
relative to each other gives meaningless scores, so :func:`auto_orient`
tries every one of the 24 axis-aligned orientations of the candidate,
renders cheap low-resolution silhouettes of the requested views and keeps
the orientation whose silhouettes overlap the reference best (mean IoU over
the views).

After the best axis-aligned orientation is found, a second pass searches
the rotation about the up axis (yaw) in fine steps, because generated models
usually have the right up axis but face whatever direction the input image
was taken from. Tilt about other axes and mirroring are not corrected.

Every view is zoomed so that the object's projected bounding box fills the
canvas, the same as the shape-first scoring config (``crop_mode:
foreground_bbox``). Without this the search could "fix" a size mismatch by
rotation: models are normalised by their largest extent, so a candidate
whose handle sticks out a little further looks smaller in every view, and
turning the handle diagonally shrinks the bounding box and scales it back up.

Speed: silhouettes need no depth or shading. Small triangles go through the
vectorised rasteriser, large ones are filled with OpenCV, and dense meshes
(AI generators emit 1M+ faces) are reduced by vertex clustering on a grid of
``2 * size`` cells before the search. The chosen rotation is applied to the
original mesh.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .metrics import compute_silhouette_iou
from .render import LoadedMesh, all_orientations, load_mesh, rasterize, reorient, rotate, view_basis, yaw_matrix

logger = logging.getLogger(__name__)


@dataclass
class OrientResult:
    up: str
    front: str
    mean_iou: float
    yaw: float = 0.0
    axis_aligned_iou: float = 0.0
    ranking: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "up": self.up,
            "front": self.front,
            "yaw": self.yaw,
            "mean_iou": self.mean_iou,
            "axis_aligned_iou": self.axis_aligned_iou,
            "ranking": self.ranking[:8],
        }


def apply_orientation(mesh: LoadedMesh, up: str, front: str, yaw: float = 0.0) -> LoadedMesh:
    """``reorient`` then rotate about +Y by ``yaw`` degrees (as chosen by :func:`auto_orient`)."""
    out = reorient(mesh, up, front)
    if abs(yaw) > 1e-9:
        out = rotate(out, yaw_matrix(yaw), {"yaw": yaw})
    return out


def _mean_iou(ref_masks: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    ious = {v: compute_silhouette_iou(ref_masks[v], masks[v]) or 0.0 for v in ref_masks}
    return float(np.mean(list(ious.values()))), ious


_FIT_MARGIN = 0.02  # empty border around the zoomed object, as a fraction of the canvas
_BIG_TRIANGLE = 4.0  # triangles wider than this many pixels are filled with OpenCV
_CV_SHIFT = 4  # fixed-point bits for sub-pixel OpenCV drawing


def fit_silhouette(mesh: LoadedMesh, view: str, size: int) -> np.ndarray:
    """Boolean silhouette of one view, zoomed so the projected bounding box fills the canvas."""
    right, up, _ = view_basis(view)
    u = mesh.vertices @ right
    w = -(mesh.vertices @ up)
    u_lo, u_hi, w_lo, w_hi = u.min(), u.max(), w.min(), w.max()
    scale = size * (1.0 - 2.0 * _FIT_MARGIN) / max(u_hi - u_lo, w_hi - w_lo, 1e-12)
    xs = size / 2.0 + (u - (u_lo + u_hi) / 2.0) * scale
    ys = size / 2.0 + (w - (w_lo + w_hi) / 2.0) * scale
    tx, ty = xs[mesh.faces], ys[mesh.faces]
    big = np.maximum(tx.max(1) - tx.min(1), ty.max(1) - ty.min(1)) > _BIG_TRIANGLE
    small = ~big
    n_small = int(small.sum())
    zbuf, _ = rasterize(np.stack([tx[small], ty[small]], axis=2), np.zeros((n_small, 3)),
                        np.zeros(n_small, np.float32), size)
    img = np.isfinite(zbuf).astype(np.uint8)
    if big.any():
        # OpenCV puts pixel centres on integer coordinates, the rasteriser at i + 0.5.
        # fillPoly with many polygons uses even-odd filling (overlaps become holes),
        # so the (few) large triangles are filled one by one.
        pts = np.rint((np.stack([tx[big], ty[big]], axis=2) - 0.5) * (1 << _CV_SHIFT)).astype(np.int32)
        for tri in pts:
            cv2.fillConvexPoly(img, tri, 1, lineType=cv2.LINE_8, shift=_CV_SHIFT)
    return img.astype(bool)


def silhouette_masks(mesh: LoadedMesh, views: Sequence[str], size: int) -> dict[str, np.ndarray]:
    """Boolean silhouettes of ``views`` at ``size`` pixels, each zoomed to the object."""
    return {v: fit_silhouette(mesh, v, size) for v in views}


def decimate_for_silhouettes(mesh: LoadedMesh, cells: int, max_faces: int = 60000) -> LoadedMesh:
    """Vertex clustering: snap vertices to ``cells`` grid cells per unit, merge each
    cell and drop collapsed or duplicate faces. Only for silhouette search; meshes
    with at most ``max_faces`` faces are returned unchanged."""
    if len(mesh.faces) <= max_faces:
        return mesh
    grid = np.floor((mesh.vertices + 0.5) * cells).astype(np.int64)
    keys, inverse = np.unique(grid, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    sums = np.zeros((len(keys), 3))
    np.add.at(sums, inverse, mesh.vertices)
    vertices = sums / np.bincount(inverse, minlength=len(keys))[:, None]
    faces = inverse[mesh.faces]
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = np.unique(np.sort(faces[ok], axis=1), axis=0)
    logger.debug("decimated %d -> %d faces for orientation search", len(mesh.faces), len(faces))
    return LoadedMesh(vertices, faces, np.zeros((len(faces), 3)), mesh.source, mesh.original_extents, dict(mesh.meta))


def auto_orient(
    candidate: str | Path | LoadedMesh,
    reference: LoadedMesh,
    views: Sequence[str],
    size: int = 128,
    orientations: Sequence[tuple[str, str]] | None = None,
    yaw_step: float = 10.0,
    yaw_refine_step: float = 2.0,
) -> OrientResult:
    """Pick the orientation of ``candidate`` that best matches ``reference``.

    Stage 1 tries the 24 axis-aligned ``(up, front)`` pairs. Stage 2 keeps the
    winner's up axis and scans the yaw (rotation about up) in ``yaw_step``
    increments over the full circle, then refines around the best angle in
    ``yaw_refine_step`` increments. ``yaw_step <= 0`` disables stage 2.

    Args:
        candidate: mesh path (loaded in the identity frame) or an already
            loaded mesh whose frame is treated as the starting point.
        reference: reference mesh in its final frame.
        views: view names used for the comparison (e.g. front/side/top).
        size: silhouette resolution (each view zoomed to the object); 128 px is
            plenty for choosing a rotation.
        orientations: subset of ``(up, front)`` pairs to try (default: all 24).

    Returns:
        ``(up, front, yaw)`` **relative to the frame the candidate was loaded
        in** (apply with :func:`apply_orientation`), plus the ranking.
    """
    views = tuple(views)
    if not views:
        raise ValueError("auto_orient needs at least one view")
    base = candidate if isinstance(candidate, LoadedMesh) else load_mesh(candidate)
    base = decimate_for_silhouettes(base, 2 * size)
    ref_masks = silhouette_masks(decimate_for_silhouettes(reference, 2 * size), views, size)

    ranking: list[dict[str, Any]] = []
    for up, front in orientations or all_orientations():
        mean, ious = _mean_iou(ref_masks, silhouette_masks(reorient(base, up, front), views, size))
        ranking.append({"up": up, "front": front, "yaw": 0.0, "mean_iou": mean, "iou": ious})
    ranking.sort(key=lambda r: r["mean_iou"], reverse=True)
    best = ranking[0]
    logger.info(
        "auto-orient: best axis-aligned up=%s front=%s (mean IoU %.3f); runner-up up=%s front=%s (%.3f)",
        best["up"], best["front"], best["mean_iou"],
        ranking[1]["up"] if len(ranking) > 1 else "-",
        ranking[1]["front"] if len(ranking) > 1 else "-",
        ranking[1]["mean_iou"] if len(ranking) > 1 else 0.0,
    )
    result = OrientResult(best["up"], best["front"], best["mean_iou"], 0.0, best["mean_iou"], ranking)
    if yaw_step <= 0:
        return result

    aligned = reorient(base, best["up"], best["front"])

    def score(yaw: float) -> tuple[float, dict[str, float]]:
        mesh = aligned if abs(yaw) < 1e-9 else rotate(aligned, yaw_matrix(yaw))
        return _mean_iou(ref_masks, silhouette_masks(mesh, views, size))

    tried: dict[float, tuple[float, dict[str, float]]] = {0.0: (best["mean_iou"], best["iou"])}
    for yaw in np.arange(yaw_step, 360.0, yaw_step):
        tried[float(yaw)] = score(float(yaw))
    coarse_best = max(tried, key=lambda k: tried[k][0])
    if yaw_refine_step > 0:
        for yaw in np.arange(coarse_best - yaw_step + yaw_refine_step, coarse_best + yaw_step, yaw_refine_step):
            y = float(yaw) % 360.0
            if y not in tried:
                tried[y] = score(y)
    best_yaw = max(tried, key=lambda k: tried[k][0])
    best_yaw = best_yaw - 360.0 if best_yaw > 180.0 else best_yaw
    mean, ious = tried[best_yaw % 360.0 if best_yaw < 0 else best_yaw]
    if mean > result.mean_iou + 1e-6:
        logger.info("auto-orient: yaw %.1f deg improves mean IoU %.3f -> %.3f", best_yaw, result.mean_iou, mean)
        result.yaw = round(float(best_yaw), 2)
        result.mean_iou = mean
        result.ranking.insert(0, {"up": best["up"], "front": best["front"], "yaw": result.yaw, "mean_iou": mean, "iou": ious})
    return result
