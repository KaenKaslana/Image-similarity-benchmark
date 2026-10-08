/**
 * Automatic orientation of a candidate mesh against a reference (port of
 * src/orient.py).
 *
 * Stage 1 tries the 24 axis-aligned orientations and keeps the one whose
 * low-resolution silhouettes overlap the reference best (mean IoU over the
 * views). Stage 2 keeps that up axis and scans the yaw in coarse, then fine
 * steps. Every silhouette is zoomed so the projected bounding box fills the
 * canvas, and dense meshes are reduced by vertex clustering first.
 */

import { computeSilhouetteIou } from './metrics.js';
import { allOrientations, LoadedMesh, loadMesh, projectFaces, rasterize, reorient, rotate, viewBasis, yawMatrix } from './render.js';
import { getLogger, mean } from './util.js';

const logger = getLogger('orient');

const FIT_MARGIN = 0.02;

/** ``reorient`` then rotate about +Y by ``yaw`` degrees. */
export function applyOrientation(mesh, up, front, yaw = 0.0) {
  let out = reorient(mesh, up, front);
  if (Math.abs(yaw) > 1e-9) out = rotate(out, yawMatrix(yaw), { yaw });
  return out;
}

function meanIou(refMasks, masks) {
  const ious = {};
  for (const v of Object.keys(refMasks)) ious[v] = computeSilhouetteIou(refMasks[v], masks[v]) || 0.0;
  return { mean: mean(Object.values(ious)), ious };
}

/** 0/1 silhouette of one view, zoomed so the projected bounding box fills the canvas. */
export function fitSilhouette(mesh, view, size) {
  const { right, up } = viewBasis(view);
  const V = mesh.vertices;
  let uLo = Infinity, uHi = -Infinity, wLo = Infinity, wHi = -Infinity;
  for (let i = 0; i < V.length; i += 3) {
    const u = V[i] * right[0] + V[i + 1] * right[1] + V[i + 2] * right[2];
    const w = V[i] * up[0] + V[i + 1] * up[1] + V[i + 2] * up[2];
    if (u < uLo) uLo = u;
    if (u > uHi) uHi = u;
    if (w < wLo) wLo = w;
    if (w > wHi) wHi = w;
  }
  const scale = (size * (1.0 - 2.0 * FIT_MARGIN)) / Math.max(uHi - uLo, wHi - wLo, 1e-12);
  const { xy, depth } = projectFaces(mesh, view, size, scale, [(uLo + uHi) / 2, (wLo + wHi) / 2]);
  const { zbuf } = rasterize(xy, depth, new Float32Array(depth.length / 3), size);
  const mask = new Uint8Array(size * size);
  for (let i = 0; i < mask.length; i++) mask[i] = zbuf[i] !== Infinity ? 1 : 0;
  return mask;
}

export function silhouetteMasks(mesh, views, size) {
  return Object.fromEntries(views.map((v) => [v, fitSilhouette(mesh, v, size)]));
}

/**
 * Vertex clustering for the silhouette search: snap vertices to ``cells``
 * grid cells per unit, merge each cell, drop collapsed or duplicate faces.
 * Meshes with at most ``maxFaces`` faces are returned unchanged.
 */
export function decimateForSilhouettes(mesh, cells, maxFaces = 60000) {
  if (mesh.numFaces <= maxFaces) return mesh;
  const V = mesh.vertices;
  const nv = V.length / 3;
  const K = cells + 1;
  const cellOf = new Map();
  const remap = new Int32Array(nv);
  const sums = [];
  const counts = [];
  for (let i = 0; i < nv; i++) {
    const gx = Math.floor((V[i * 3] + 0.5) * cells);
    const gy = Math.floor((V[i * 3 + 1] + 0.5) * cells);
    const gz = Math.floor((V[i * 3 + 2] + 0.5) * cells);
    const key = (gx * K + gy) * K + gz;
    let c = cellOf.get(key);
    if (c === undefined) {
      c = counts.length;
      cellOf.set(key, c);
      sums.push(0, 0, 0);
      counts.push(0);
    }
    remap[i] = c;
    sums[c * 3] += V[i * 3];
    sums[c * 3 + 1] += V[i * 3 + 1];
    sums[c * 3 + 2] += V[i * 3 + 2];
    counts[c] += 1;
  }
  const vertices = new Float64Array(counts.length * 3);
  for (let c = 0; c < counts.length; c++) for (let k = 0; k < 3; k++) vertices[c * 3 + k] = sums[c * 3 + k] / counts[c];
  const seen = new Set();
  const faces = [];
  const F = mesh.faces;
  for (let f = 0; f < F.length; f += 3) {
    const a = remap[F[f]], b = remap[F[f + 1]], c = remap[F[f + 2]];
    if (a === b || b === c || a === c) continue;
    const s = [a, b, c].sort((x, y) => x - y);
    const key = `${s[0]},${s[1]},${s[2]}`;
    if (seen.has(key)) continue;
    seen.add(key);
    faces.push(s[0], s[1], s[2]);
  }
  logger.debug(`decimated ${mesh.numFaces} -> ${faces.length / 3} faces for orientation search`);
  return new LoadedMesh(vertices, Uint32Array.from(faces), new Float64Array(faces.length), mesh.source, mesh.originalExtents, { ...mesh.meta });
}

/**
 * Pick the orientation of ``candidate`` (mesh path or loaded mesh) that best
 * matches ``reference``. Returns ``{up, front, yaw, mean_iou,
 * axis_aligned_iou, ranking}`` relative to the frame the candidate was loaded
 * in (apply with :func:`applyOrientation`).
 */
export function autoOrient(candidate, reference, views, { size = 128, orientations = null, yawStep = 10.0, yawRefineStep = 2.0 } = {}) {
  views = [...views];
  if (!views.length) throw new Error('autoOrient needs at least one view');
  let base = candidate instanceof LoadedMesh ? candidate : loadMesh(candidate);
  base = decimateForSilhouettes(base, 2 * size);
  const refMasks = silhouetteMasks(decimateForSilhouettes(reference, 2 * size), views, size);

  const ranking = [];
  for (const [up, front] of orientations ?? allOrientations()) {
    const { mean: m, ious } = meanIou(refMasks, silhouetteMasks(reorient(base, up, front), views, size));
    ranking.push({ up, front, yaw: 0.0, mean_iou: m, iou: ious });
  }
  ranking.sort((a, b) => b.mean_iou - a.mean_iou);
  const best = ranking[0];
  const second = ranking[1];
  logger.info(
    `auto-orient: best axis-aligned up=${best.up} front=${best.front} (mean IoU ${best.mean_iou.toFixed(3)}); ` +
      `runner-up up=${second?.up ?? '-'} front=${second?.front ?? '-'} (${(second?.mean_iou ?? 0).toFixed(3)})`,
  );
  const result = { up: best.up, front: best.front, yaw: 0.0, mean_iou: best.mean_iou, axis_aligned_iou: best.mean_iou, ranking };
  if (yawStep <= 0) return result;

  const aligned = reorient(base, best.up, best.front);
  const score = (yaw) => meanIou(refMasks, silhouetteMasks(Math.abs(yaw) < 1e-9 ? aligned : rotate(aligned, yawMatrix(yaw)), views, size));

  const tried = new Map([[0.0, { mean: best.mean_iou, ious: best.iou }]]);
  for (let i = 1; yawStep * i < 360.0; i++) tried.set(yawStep * i, score(yawStep * i));
  const argmax = () => {
    let bestKey = null;
    for (const [k, v] of tried) if (bestKey === null || v.mean > tried.get(bestKey).mean) bestKey = k;
    return bestKey;
  };
  const coarseBest = argmax();
  if (yawRefineStep > 0) {
    const start = coarseBest - yawStep + yawRefineStep;
    for (let i = 0; start + i * yawRefineStep < coarseBest + yawStep; i++) {
      const y = (((start + i * yawRefineStep) % 360.0) + 360.0) % 360.0;
      if (!tried.has(y)) tried.set(y, score(y));
    }
  }
  let bestYaw = argmax();
  const { mean: m, ious } = tried.get(bestYaw);
  if (bestYaw > 180.0) bestYaw -= 360.0;
  if (m > result.mean_iou + 1e-6) {
    logger.info(`auto-orient: yaw ${bestYaw.toFixed(1)} deg improves mean IoU ${result.mean_iou.toFixed(3)} -> ${m.toFixed(3)}`);
    result.yaw = Math.round(bestYaw * 100) / 100;
    result.mean_iou = m;
    result.ranking.unshift({ up: best.up, front: best.front, yaw: result.yaw, mean_iou: m, iou: ious });
  }
  return result;
}

/** JSON summary as written to models.json (top 8 of the ranking). */
export function orientResultToDict(r) {
  return { up: r.up, front: r.front, yaw: r.yaw, mean_iou: r.mean_iou, axis_aligned_iou: r.axis_aligned_iou, ranking: r.ranking.slice(0, 8) };
}
