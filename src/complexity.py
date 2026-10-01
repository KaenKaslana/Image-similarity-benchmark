"""Mesh complexity term for model comparisons: how the candidate's face count
compares with the reference's.

The image metrics only see silhouettes and shading, so a 200-face blocky copy
and a 200k-face replica of the same shape score alike. The face ratio
``candidate_faces / reference_faces`` is scored on a log2 scale in one of two
modes:

* ``fewer_is_better`` (configs/shape.yaml): the shape score already says how
  well the model matches, so among equally good models the leaner one wins.
  More faces than the reference (beyond ``free_log2``) is penalised down to 0
  at ``zero_log2``; fewer faces earn an efficiency bonus of up to
  ``bonus_weight`` of the shape score, reached at ``bonus_log2`` below the
  reference. The overall score is capped at 100.
* ``symmetric``: far fewer faces means under-modelled, far more means bloated
  (typical for AI generators that output dense scans); both directions are
  penalised alike.

Counts are taken after trimesh loading, i.e. triangulated and with all scene
parts merged, for BOTH models - so a quad-based reference counts twice its
quads, exactly like the candidate would.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["MeshStats", "MODES", "mesh_complexity", "blend_with_shape_score", "apply_penalties", "efficiency_bonus"]


@dataclass(frozen=True)
class MeshStats:
    faces: int
    vertices: int

    def to_dict(self) -> dict[str, int]:
        return {"faces": int(self.faces), "vertices": int(self.vertices)}


MODES = ("fewer_is_better", "symmetric")


def complexity_score(face_ratio: float, free_log2: float, zero_log2: float, mode: str = "symmetric") -> float:
    """0-100 penalty score from ``candidate_faces / reference_faces``.

    ``log2(ratio) <= free_log2`` gives 100 (within a factor 2 by default);
    the score then falls linearly and reaches 0 at ``zero_log2`` (factor 32 by
    default). ``symmetric`` uses ``|log2(ratio)|`` so half the faces costs the
    same as twice the faces; ``fewer_is_better`` never penalises fewer faces.
    """
    if face_ratio <= 0 or not math.isfinite(face_ratio):
        return 0.0
    d = math.log2(face_ratio)
    if mode == "symmetric":
        d = abs(d)
    if d <= free_log2:
        return 100.0
    if d >= zero_log2:
        return 0.0
    return 100.0 * (1.0 - (d - free_log2) / (zero_log2 - free_log2))


def efficiency_bonus(face_ratio: float, bonus_log2: float) -> float:
    """0-1: how much of the ``fewer_is_better`` bonus a candidate earns.

    0 at the reference's face count or above, rising linearly to 1 when the
    candidate has ``2**bonus_log2`` times fewer faces (a quarter by default).
    """
    if face_ratio <= 0 or not math.isfinite(face_ratio) or bonus_log2 <= 0:
        return 0.0
    d = -math.log2(face_ratio)  # > 0 when the candidate has fewer faces
    return float(min(1.0, max(0.0, d / bonus_log2)))


def mesh_complexity(reference: MeshStats, candidate: MeshStats, cfg: Any) -> dict[str, Any]:
    """Face-count comparison of two loaded meshes as a JSON-friendly dict.

    ``cfg`` is a :class:`src.config.MeshComplexityConfig` (``weight``,
    ``free_log2``, ``zero_log2``).
    """
    ratio = candidate.faces / reference.faces if reference.faces else float("inf")
    mode = getattr(cfg, "mode", "symmetric")
    bonus_weight = float(getattr(cfg, "bonus_weight", 0.0)) if mode == "fewer_is_better" else 0.0
    bonus_log2 = float(getattr(cfg, "bonus_log2", 0.0))
    return {
        "reference": reference.to_dict(),
        "candidate": candidate.to_dict(),
        "face_ratio": float(ratio) if math.isfinite(ratio) else None,
        "log2_face_ratio": float(math.log2(ratio)) if ratio > 0 and math.isfinite(ratio) else None,
        "mode": mode,
        "score": complexity_score(ratio, cfg.free_log2, cfg.zero_log2, mode),
        "weight": float(cfg.weight),
        "free_log2": float(cfg.free_log2),
        "zero_log2": float(cfg.zero_log2),
        "bonus": efficiency_bonus(ratio, bonus_log2) if bonus_weight > 0 else 0.0,
        "bonus_weight": bonus_weight,
        "bonus_log2": bonus_log2,
    }


def blend_with_shape_score(shape_score: float | None, mesh: dict[str, Any] | None) -> float | None:
    """``overall = shape * (1 - w * (1 - mesh_score / 100)) * (1 + bonus_weight * bonus)``, capped at 100.

    A multiplicative penalty rather than a weighted average: a candidate with a
    matching face count keeps exactly its shape score (so the calibrated scale
    - identical 100, unrelated ~3 - is untouched), and a face count 32x off
    costs at most ``w`` of the score. In ``fewer_is_better`` mode a leaner
    candidate gains up to ``bonus_weight`` of its shape score on top (also
    multiplicative, so a poor shape stays poor). Unchanged when there is no
    mesh information or both weights are 0.
    """
    if shape_score is None or not mesh:
        return shape_score
    w = float(mesh.get("weight", 0.0))
    bw = float(mesh.get("bonus_weight", 0.0))
    out = float(shape_score)
    if w > 0:
        out *= 1.0 - w * (1.0 - float(mesh["score"]) / 100.0)
    if bw > 0:
        out *= 1.0 + bw * float(mesh.get("bonus", 0.0))
    return min(out, 100.0)


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
