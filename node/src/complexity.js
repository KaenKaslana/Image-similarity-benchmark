/**
 * Mesh complexity term for model comparisons (port of src/complexity.py):
 * how the candidate's face count compares with the reference's, scored on a
 * log2 scale and applied as a multiplicative penalty on the shape score.
 */

export const MODES = ['fewer_is_better', 'symmetric'];

/** 0-100 score from ``candidate_faces / reference_faces``. */
export function complexityScore(faceRatio, freeLog2, zeroLog2, mode = 'symmetric') {
  if (!(faceRatio > 0) || !Number.isFinite(faceRatio)) return 0.0;
  let d = Math.log2(faceRatio);
  if (mode === 'symmetric') d = Math.abs(d);
  if (d <= freeLog2) return 100.0;
  if (d >= zeroLog2) return 0.0;
  return 100.0 * (1.0 - (d - freeLog2) / (zeroLog2 - freeLog2));
}

/** 0-1 share of the ``fewer_is_better`` bonus: 0 at the reference's count, 1 at ``2**bonusLog2`` fewer faces. */
export function efficiencyBonus(faceRatio, bonusLog2) {
  if (!(faceRatio > 0) || !Number.isFinite(faceRatio) || bonusLog2 <= 0) return 0.0;
  const d = -Math.log2(faceRatio);
  return Math.min(1.0, Math.max(0.0, d / bonusLog2));
}

/** Face-count comparison of two meshes (``{faces, vertices}``) as a JSON-friendly object. */
export function meshComplexity(reference, candidate, cfg) {
  const ratio = reference.faces ? candidate.faces / reference.faces : Infinity;
  const mode = cfg.mode ?? 'symmetric';
  const bonusWeight = mode === 'fewer_is_better' ? cfg.bonus_weight ?? 0.0 : 0.0;
  const bonusLog2 = cfg.bonus_log2 ?? 0.0;
  return {
    reference: { faces: reference.faces, vertices: reference.vertices },
    candidate: { faces: candidate.faces, vertices: candidate.vertices },
    face_ratio: Number.isFinite(ratio) ? ratio : null,
    log2_face_ratio: ratio > 0 && Number.isFinite(ratio) ? Math.log2(ratio) : null,
    mode,
    score: complexityScore(ratio, cfg.free_log2, cfg.zero_log2, mode),
    weight: cfg.weight,
    free_log2: cfg.free_log2,
    zero_log2: cfg.zero_log2,
    bonus: bonusWeight > 0 ? efficiencyBonus(ratio, bonusLog2) : 0.0,
    bonus_weight: bonusWeight,
    bonus_log2: bonusLog2,
  };
}

/** ``shape * (1 - w * (1 - score / 100)) * (1 + bonus_weight * bonus)``, capped at 100. */
export function blendWithShapeScore(shapeScore, term) {
  if (shapeScore === null || !term) return shapeScore;
  const w = term.weight ?? 0.0;
  const bw = term.bonus_weight ?? 0.0;
  let out = shapeScore;
  if (w > 0) out *= 1.0 - w * (1.0 - term.score / 100.0);
  if (bw > 0) out *= 1.0 + bw * (term.bonus ?? 0.0);
  return Math.min(out, 100.0);
}

/** Apply independent penalty terms (face count, rig, ...) in turn; skips ``null`` / non-applicable terms. */
export function applyPenalties(shapeScore, terms) {
  let out = shapeScore;
  for (const term of terms) {
    if (!term || term.applicable === false || term.score === null || term.score === undefined) continue;
    out = blendWithShapeScore(out, term);
  }
  return out;
}
