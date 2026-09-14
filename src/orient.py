"""Automatic orientation of a candidate mesh against a reference.

AI-generated or downloaded models rarely agree on which axis is "up" and
which side is the "front". Comparing views of two models that are rotated
relative to each other gives meaningless scores, so :func:`auto_orient`
tries every one of the 24 axis-aligned orientations of the candidate,
renders cheap low-resolution silhouettes of the requested views and keeps
the orientation whose silhouettes overlap the reference best (mean IoU over
the views).

Only 90-degree rotations are considered; a model that is tilted by an
arbitrary angle cannot be fixed here. Mirroring is not considered either.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .metrics import compute_silhouette_iou
from .render import LoadedMesh, RenderOptions, all_orientations, load_mesh, render_view, reorient

logger = logging.getLogger(__name__)


@dataclass
class OrientResult:
    up: str
    front: str
    mean_iou: float
    ranking: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"up": self.up, "front": self.front, "mean_iou": self.mean_iou, "ranking": self.ranking[:8]}


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
) -> OrientResult:
    """Pick the axis-aligned orientation of ``candidate`` that best matches ``reference``.

    Args:
        candidate: mesh path (loaded in the identity frame) or an already
            loaded mesh whose frame is treated as the starting point.
        reference: reference mesh in its final frame.
        views: view names used for the comparison (e.g. front/side/top).
        size: silhouette resolution; 128 px is plenty for choosing a rotation.
        orientations: subset of ``(up, front)`` pairs to try (default: all 24).

    Returns:
        The winning ``(up, front)`` **relative to the frame the candidate was
        loaded in**, plus the full ranking.
    """
    views = tuple(views)
    if not views:
        raise ValueError("auto_orient needs at least one view")
    base = candidate if isinstance(candidate, LoadedMesh) else load_mesh(candidate)
    ref_masks = silhouette_masks(reference, views, size)

    ranking: list[dict[str, Any]] = []
    for up, front in orientations or all_orientations():
        mesh = reorient(base, up, front)
        masks = silhouette_masks(mesh, views, size)
        ious = {v: compute_silhouette_iou(ref_masks[v], masks[v]) or 0.0 for v in views}
        mean = float(np.mean(list(ious.values())))
        ranking.append({"up": up, "front": front, "mean_iou": mean, "iou": ious})

    ranking.sort(key=lambda r: r["mean_iou"], reverse=True)
    best = ranking[0]
    logger.info(
        "auto-orient: best up=%s front=%s (mean IoU %.3f); runner-up up=%s front=%s (%.3f)",
        best["up"],
        best["front"],
        best["mean_iou"],
        ranking[1]["up"] if len(ranking) > 1 else "-",
        ranking[1]["front"] if len(ranking) > 1 else "-",
        ranking[1]["mean_iou"] if len(ranking) > 1 else 0.0,
    )
    return OrientResult(best["up"], best["front"], best["mean_iou"], ranking)
