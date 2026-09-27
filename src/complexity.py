"""Mesh complexity term for model comparisons: how the candidate's face count
compares with the reference's.

The image metrics only see silhouettes and shading, so a 200-face blocky copy
and a 200k-face replica of the same shape score alike. The face ratio is a
cheap proxy for the level of detail: far fewer faces than the reference means
the candidate is under-modelled, far more means it is bloated (typical for
AI generators that output dense scans). Both directions are penalised
symmetrically on a log2 scale.

Counts are taken after trimesh loading, i.e. triangulated and with all scene
parts merged, for BOTH models - so a quad-based reference counts twice its
quads, exactly like the candidate would.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["MeshStats", "mesh_complexity", "blend_with_shape_score", "apply_penalties"]


@dataclass(frozen=True)
class MeshStats:
    faces: int
    vertices: int

    def to_dict(self) -> dict[str, int]:
        return {"faces": int(self.faces), "vertices": int(self.vertices)}


def complexity_score(face_ratio: float, free_log2: float, zero_log2: float) -> float:
    """0-100 score from ``candidate_faces / reference_faces``.

    ``|log2(ratio)| <= free_log2`` gives 100 (within a factor 2 by default);
    the score then falls linearly and reaches 0 at ``zero_log2`` (factor 32 by
    default). Symmetric: half the faces costs the same as twice the faces.
    """
    if face_ratio <= 0 or not math.isfinite(face_ratio):
        return 0.0
    d = abs(math.log2(face_ratio))
    if d <= free_log2:
        return 100.0
    if d >= zero_log2:
        return 0.0
    return 100.0 * (1.0 - (d - free_log2) / (zero_log2 - free_log2))


def mesh_complexity(reference: MeshStats, candidate: MeshStats, cfg: Any) -> dict[str, Any]:
    """Face-count comparison of two loaded meshes as a JSON-friendly dict.

    ``cfg`` is a :class:`src.config.MeshComplexityConfig` (``weight``,
    ``free_log2``, ``zero_log2``).
    """
    ratio = candidate.faces / reference.faces if reference.faces else float("inf")
    return {
        "reference": reference.to_dict(),
        "candidate": candidate.to_dict(),
        "face_ratio": float(ratio) if math.isfinite(ratio) else None,
        "log2_face_ratio": float(math.log2(ratio)) if ratio > 0 and math.isfinite(ratio) else None,
        "score": complexity_score(ratio, cfg.free_log2, cfg.zero_log2),
        "weight": float(cfg.weight),
        "free_log2": float(cfg.free_log2),
        "zero_log2": float(cfg.zero_log2),
    }


def blend_with_shape_score(shape_score: float | None, mesh: dict[str, Any] | None) -> float | None:
    """``overall = shape * (1 - w * (1 - mesh_score / 100))``.

    A multiplicative penalty rather than a weighted average: a candidate with a
    matching face count keeps exactly its shape score (so the calibrated scale
    - identical 100, unrelated ~3 - is untouched), and a face count 32x off in
    either direction costs at most ``w`` of the score. Unchanged when w = 0 or
    there is no mesh information.
    """
    if shape_score is None or not mesh:
        return shape_score
    w = float(mesh.get("weight", 0.0))
    if w <= 0:
        return shape_score
    penalty = 1.0 - float(mesh["score"]) / 100.0
    return float(shape_score * (1.0 - w * penalty))


def apply_penalties(shape_score: float | None, terms: Sequence[dict[str, Any] | None]) -> float | None:
    """Apply several independent penalty terms (face count, rig, ...) in turn.

    Each term is a dict with ``score`` (0-100), ``weight`` and an optional
    ``applicable`` flag (False = skip, e.g. the rig term when the reference has
    no skeleton); ``None`` terms are skipped.
    """
    out = shape_score
    for term in terms:
        if term is None or not term.get("applicable", True) or term.get("score") is None:
            continue
        out = blend_with_shape_score(out, term)
    return out
