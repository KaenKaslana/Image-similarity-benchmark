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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .metrics import compute_silhouette_iou
from .render import LoadedMesh, RenderOptions, all_orientations, load_mesh, render_view, reorient, rotate, yaw_matrix

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


def silhouette_masks(mesh: LoadedMesh, views: Sequence[str], size: int) -> dict[str, np.ndarray]:
    """Boolean foreground masks of ``views`` rendered at ``size`` pixels."""
    opts = RenderOptions(size=size, views=tuple(views), style="silhouette", supersample=1)
    return {v: render_view(mesh, v, opts)[..., 3] > 0 for v in views}


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
        size: silhouette resolution; 128 px is plenty for choosing a rotation.
        orientations: subset of ``(up, front)`` pairs to try (default: all 24).

    Returns:
        ``(up, front, yaw)`` **relative to the frame the candidate was loaded
        in** (apply with :func:`apply_orientation`), plus the ranking.
    """
    views = tuple(views)
    if not views:
        raise ValueError("auto_orient needs at least one view")
    base = candidate if isinstance(candidate, LoadedMesh) else load_mesh(candidate)
    ref_masks = silhouette_masks(reference, views, size)

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
