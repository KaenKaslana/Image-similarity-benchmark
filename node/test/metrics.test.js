import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Image8 } from '../src/image.js';
import {
  applyScoreFloor,
  canny,
  combineScores,
  computeEdgeSimilarity,
  computeSilhouetteIou,
  computeSsim,
  distanceToNearestEdge,
  ssimToScore,
} from '../src/metrics.js';
import { defaultConfig } from '../src/config.js';

function solid(size, value) {
  const img = new Image8(size, size, 3);
  img.data.fill(value);
  return img;
}

/** White canvas with a grey axis-aligned square at [x0, x1) x [y0, y1). */
function square(size, x0, y0, x1, y1, value = 90) {
  const img = solid(size, 255);
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) img.data.fill(value, (y * size + x) * 3, (y * size + x) * 3 + 3);
  return img;
}

function squareMask(size, x0, y0, x1, y1) {
  const m = new Uint8Array(size * size);
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) m[y * size + x] = 1;
  return m;
}

const cfg = defaultConfig();

test('SSIM of identical images is 1 and of a different image is lower', () => {
  const a = square(64, 16, 16, 48, 48);
  assert.ok(Math.abs(computeSsim(a, a, cfg.metrics.ssim) - 1) < 1e-12);
  const b = square(64, 20, 16, 52, 48);
  const s = computeSsim(a, b, cfg.metrics.ssim);
  assert.ok(s < 1 && s > 0);
  assert.equal(ssimToScore(1.2), 100);
  assert.equal(ssimToScore(-0.3), 0);
});

test('SSIM with the uniform (non-Gaussian) window also works', () => {
  const a = square(64, 16, 16, 48, 48);
  const b = square(64, 18, 16, 50, 48);
  const s = computeSsim(a, b, { gaussian_weights: false, sigma: 1.5, win_size: 7 });
  assert.ok(s < 1 && s > 0);
});

test('silhouette IoU', () => {
  const a = squareMask(32, 0, 0, 16, 16);
  const b = squareMask(32, 8, 0, 24, 16);
  assert.ok(Math.abs(computeSilhouetteIou(a, b) - 1 / 3) < 1e-12);
  assert.equal(computeSilhouetteIou(a, null), null);
  assert.equal(computeSilhouetteIou(new Uint8Array(4), new Uint8Array(4)), null);
});

test('Canny finds the four sides of a square and nothing on a flat image', () => {
  const img = square(64, 16, 16, 48, 48);
  const gray = new Uint8Array(64 * 64);
  for (let i = 0; i < gray.length; i++) gray[i] = img.data[i * 3];
  const edges = canny(gray, 64, 64, 100, 200);
  let n = 0;
  for (const e of edges) n += e;
  // a 32 px square has a perimeter of ~128 edge pixels (one pixel wide)
  assert.ok(n >= 120 && n <= 140, `edge pixels ${n}`);
  assert.equal(edges[16 * 64 + 32] + edges[15 * 64 + 32], 1); // exactly one of the two rows at the top side
  const flat = canny(new Uint8Array(64 * 64).fill(90), 64, 64, 100, 200);
  assert.equal(flat.reduce((s, v) => s + v, 0), 0);
});

test('distance transform: zero on edges, 1 next to them, ~1.4 diagonally', () => {
  const edges = new Uint8Array(9 * 9);
  edges[4 * 9 + 4] = 1;
  const d = distanceToNearestEdge(edges, 9, 9);
  assert.equal(d[4 * 9 + 4], 0);
  assert.ok(Math.abs(d[4 * 9 + 5] - 1) < 1e-4);
  assert.ok(Math.abs(d[3 * 9 + 3] - 1.4) < 1e-4);
  assert.ok(Math.abs(d[2 * 9 + 3] - 2.1969) < 1e-3);
});

test('edge similarity: identical 100, shifted lower, flat images not available', () => {
  const a = square(128, 32, 32, 96, 96);
  const same = computeEdgeSimilarity(a, a, cfg.metrics.edge);
  assert.equal(same.score, 100);
  const shifted = computeEdgeSimilarity(a, square(128, 36, 32, 100, 96), cfg.metrics.edge);
  assert.ok(shifted.score < 100 && shifted.score > 50, `score ${shifted.score}`);
  const flat = computeEdgeSimilarity(solid(64, 200), solid(64, 200), cfg.metrics.edge);
  assert.equal(flat.score, null);
  const oneSided = computeEdgeSimilarity(a, solid(128, 255), cfg.metrics.edge);
  assert.equal(oneSided.score, 0);
});

test('score floors and gamma', () => {
  assert.equal(applyScoreFloor(70, 0, 1), 70);
  assert.equal(applyScoreFloor(45, 45, 1), 0);
  assert.equal(applyScoreFloor(100, 45, 0.4), 100);
  assert.ok(Math.abs(applyScoreFloor(72.5, 45, 1) - 50) < 1e-9);
  assert.ok(applyScoreFloor(72.5, 45, 0.4) > 50);
  assert.equal(applyScoreFloor(null, 45, 0.4), null);
});

test('combineScores renormalises the weights of missing metrics', () => {
  const weights = { ssim: 0.5, silhouette: 0.3, edge: 0.2 };
  const full = combineScores({ ssim: 80, silhouette: 60, edge: 40 }, weights);
  assert.ok(Math.abs(full.pairScore - (40 + 18 + 8)) < 1e-9);
  assert.deepEqual(full.effectiveWeights, weights);
  const partial = combineScores({ ssim: 80, silhouette: null, edge: 40 }, weights);
  assert.ok(Math.abs(partial.effectiveWeights.ssim - 0.5 / 0.7) < 1e-12);
  assert.ok(Math.abs(partial.pairScore - (80 * 0.5 + 40 * 0.2) / 0.7) < 1e-9);
  assert.deepEqual(combineScores({ ssim: null, silhouette: null, edge: null }, weights), { pairScore: null, effectiveWeights: {} });
});
