/**
 * Similarity metrics (port of src/metrics.py, without LPIPS).
 *
 * All functions take preprocessed RGB :class:`Image8` canvases of identical
 * size. ``null`` means the metric is not available for the inputs.
 *
 * * ``ssim_score       = 100 * clip(ssim, 0, 1)``    (skimage structural_similarity)
 * * ``silhouette_score = 100 * IoU``
 * * ``edge_score       = 100 * (1 - symmetric truncated Chamfer distance)``
 *   (OpenCV Canny + 5x5 chamfer distance transform, integer-exact ports)
 */

import { METRIC_NAMES } from './config.js';
import { getLogger } from './util.js';

const logger = getLogger('metrics');

function checkPair(a, b) {
  if (a.width !== b.width || a.height !== b.height || a.channels !== b.channels) {
    throw new Error(`Image shapes differ: ${a.width}x${a.height}x${a.channels} vs ${b.width}x${b.height}x${b.channels}`);
  }
  if (a.channels !== 3) throw new Error(`Expected RGB images, got ${a.channels} channel(s)`);
}

// ---------------------------------------------------------------------------
// SSIM (skimage.metrics.structural_similarity, data_range 255, channel average)
// ---------------------------------------------------------------------------
/** scipy.ndimage ``reflect`` boundary: ``d c b a | a b c d | d c b a``. */
function reflectIndex(i, n) {
  if (n === 1) return 0;
  const period = 2 * n;
  i %= period;
  if (i < 0) i += period;
  return i < n ? i : period - i - 1;
}

/** Separable correlation of a float64 ``h x w`` image with a 1-D kernel along both axes. */
function separableFilter(src, w, h, kernel) {
  const r = (kernel.length - 1) >> 1;
  const tmp = new Float64Array(w * h);
  const out = new Float64Array(w * h);
  // axis 0 (rows / y) first, as scipy iterates the axes in order
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) s += kernel[k + r] * src[reflectIndex(y + k, h) * w + x];
      tmp[y * w + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) s += kernel[k + r] * tmp[row + reflectIndex(x + k, w)];
      out[row + x] = s;
    }
  }
  return out;
}

function gaussianKernel(sigma, radius) {
  const k = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    k[i + radius] = Math.exp((-0.5 / (sigma * sigma)) * i * i);
    sum += k[i + radius];
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}

function ssimChannel(a, b, w, h, kernel, covNorm) {
  const n = w * h;
  const aa = new Float64Array(n);
  const bb = new Float64Array(n);
  const ab = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    aa[i] = a[i] * a[i];
    bb[i] = b[i] * b[i];
    ab[i] = a[i] * b[i];
  }
  const ux = separableFilter(a, w, h, kernel);
  const uy = separableFilter(b, w, h, kernel);
  const uxx = separableFilter(aa, w, h, kernel);
  const uyy = separableFilter(bb, w, h, kernel);
  const uxy = separableFilter(ab, w, h, kernel);
  const C1 = (0.01 * 255) ** 2;
  const C2 = (0.03 * 255) ** 2;
  const pad = (kernel.length - 1) >> 1;
  let sum = 0;
  let count = 0;
  for (let y = pad; y < h - pad; y++) {
    for (let x = pad; x < w - pad; x++) {
      const i = y * w + x;
      const vx = covNorm * (uxx[i] - ux[i] * ux[i]);
      const vy = covNorm * (uyy[i] - uy[i] * uy[i]);
      const vxy = covNorm * (uxy[i] - ux[i] * uy[i]);
      const A1 = 2 * ux[i] * uy[i] + C1;
      const A2 = 2 * vxy + C2;
      const B1 = ux[i] * ux[i] + uy[i] * uy[i] + C1;
      const B2 = vx + vy + C2;
      sum += (A1 * A2) / (B1 * B2);
      count++;
    }
  }
  return sum / count;
}

/** Mean SSIM over the three channels (Gaussian 11-tap window by default, as in Wang et al. 2004). */
export function computeSsim(ref, cand, cfg = { gaussian_weights: true, sigma: 1.5, win_size: 7 }) {
  checkPair(ref, cand);
  const { width: w, height: h } = ref;
  let kernel;
  let covNorm;
  if (cfg.gaussian_weights) {
    const radius = Math.trunc(3.5 * cfg.sigma + 0.5);
    kernel = gaussianKernel(cfg.sigma, radius);
    covNorm = 1.0;
  } else {
    const size = cfg.win_size;
    kernel = new Float64Array(size).fill(1 / size);
    const NP = size * size;
    covNorm = NP / (NP - 1);
  }
  const n = w * h;
  let total = 0;
  for (let c = 0; c < 3; c++) {
    const a = new Float64Array(n);
    const b = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      a[i] = ref.data[i * 3 + c];
      b[i] = cand.data[i * 3 + c];
    }
    total += ssimChannel(a, b, w, h, kernel, covNorm);
  }
  return total / 3;
}

export function ssimToScore(ssim) {
  return 100.0 * Math.min(1, Math.max(0, ssim));
}

// ---------------------------------------------------------------------------
// Silhouette IoU
// ---------------------------------------------------------------------------
/** IoU of two 0/1 masks; ``null`` when a mask is missing or both are empty. */
export function computeSilhouetteIou(maskRef, maskCand) {
  if (maskRef === null || maskCand === null) return null;
  if (maskRef.length !== maskCand.length) throw new Error(`Mask sizes differ: ${maskRef.length} vs ${maskCand.length}`);
  let inter = 0;
  let union = 0;
  for (let i = 0; i < maskRef.length; i++) {
    const a = maskRef[i] !== 0;
    const b = maskCand[i] !== 0;
    if (a || b) union++;
    if (a && b) inter++;
  }
  return union === 0 ? null : inter / union;
}

export function iouToScore(iou) {
  return iou === null ? null : 100.0 * iou;
}

// ---------------------------------------------------------------------------
// Edge similarity
// ---------------------------------------------------------------------------
/**
 * ``cv2.cvtColor(rgb, COLOR_RGB2GRAY)`` for 8-bit data: OpenCV 4.x's 15-bit
 * fixed-point coefficients (``RY15 / GY15 / BY15``), verified against
 * OpenCV 4.10 over all 2^24 colours.
 */
export function rgbToGray(img) {
  const n = img.width * img.height;
  const out = new Uint8Array(n);
  const d = img.data;
  for (let i = 0; i < n; i++) out[i] = (d[i * 3] * 9798 + d[i * 3 + 1] * 19235 + d[i * 3 + 2] * 3735 + 16384) >> 15;
  return out;
}

function reflect101(i, n) {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

/** Integer taps of OpenCV's ``getGaussianKernel(k, 0)`` for k = 3, 5, 7 (the fixed small-kernel table). */
const SMALL_GAUSSIAN = { 3: [1, 2, 1], 5: [1, 4, 6, 4, 1], 7: [2, 7, 14, 18, 14, 7, 2] };

/** ``cv2.GaussianBlur(gray, (k, k), 0)`` on 8-bit data with BORDER_REFLECT_101. */
export function gaussianBlur8(src, w, h, k) {
  let taps = SMALL_GAUSSIAN[k];
  if (!taps) {
    const sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8;
    const g = gaussianKernel(sigma, (k - 1) >> 1);
    taps = Array.from(g, (v) => Math.round(v * 256));
  }
  const norm = taps.reduce((a, b) => a + b, 0);
  const r = (k - 1) >> 1;
  const tmp = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let t = -r; t <= r; t++) s += taps[t + r] * src[y * w + reflect101(x + t, w)];
      tmp[y * w + x] = s;
    }
  }
  const out = new Uint8Array(w * h);
  const div = norm * norm;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let t = -r; t <= r; t++) s += taps[t + r] * tmp[reflect101(y + t, h) * w + x];
      const v = Math.floor((s + div / 2) / div);
      out[y * w + x] = v > 255 ? 255 : v;
    }
  }
  return out;
}

const CANNY_SHIFT = 15;
const TG22 = Math.trunc(0.4142135623730950488016887242097 * (1 << CANNY_SHIFT) + 0.5);

/** ``cv2.Canny(gray, low, high)``: 3x3 Sobel (BORDER_REPLICATE), L1 magnitude, NMS, hysteresis. */
export function canny(gray, w, h, low, high) {
  if (low > high) [low, high] = [high, low];
  low = Math.floor(low);
  high = Math.floor(high);
  const at = (x, y) => gray[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  const dx = new Int32Array(w * h);
  const dy = new Int32Array(w * h);
  // magnitude with a zero border of one pixel on every side
  const W2 = w + 2;
  const mag = new Int32Array(W2 * (h + 2));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p00 = at(x - 1, y - 1), p01 = at(x, y - 1), p02 = at(x + 1, y - 1);
      const p10 = at(x - 1, y), p12 = at(x + 1, y);
      const p20 = at(x - 1, y + 1), p21 = at(x, y + 1), p22 = at(x + 1, y + 1);
      const gx = p02 + 2 * p12 + p22 - p00 - 2 * p10 - p20;
      const gy = p20 + 2 * p21 + p22 - p00 - 2 * p01 - p02;
      dx[y * w + x] = gx;
      dy[y * w + x] = gy;
      mag[(y + 1) * W2 + x + 1] = Math.abs(gx) + Math.abs(gy);
    }
  }
  // map: 0 = weak candidate, 1 = not an edge, 2 = edge
  const map = new Uint8Array(w * h).fill(1);
  const stack = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const mi = (y + 1) * W2 + x + 1;
      const m = mag[mi];
      if (m <= low) continue;
      const xs = dx[y * w + x];
      const ys = dy[y * w + x];
      const ax = Math.abs(xs);
      const ay = Math.abs(ys) * (1 << CANNY_SHIFT);
      const tg22x = ax * TG22;
      let isMax;
      if (ay < tg22x) {
        isMax = m > mag[mi - 1] && m >= mag[mi + 1];
      } else {
        const tg67x = tg22x + ax * (1 << (CANNY_SHIFT + 1));
        if (ay > tg67x) {
          isMax = m > mag[mi - W2] && m >= mag[mi + W2];
        } else {
          const s = (xs ^ ys) < 0 ? -1 : 1;
          isMax = m > mag[mi - W2 - s] && m > mag[mi + W2 + s];
        }
      }
      if (!isMax) continue;
      if (m > high) {
        map[y * w + x] = 2;
        stack.push(y * w + x);
      } else {
        map[y * w + x] = 0;
      }
    }
  }
  while (stack.length) {
    const i = stack.pop();
    const x = i % w;
    const y = (i - x) / w;
    for (let oy = -1; oy <= 1; oy++) {
      const ny = y + oy;
      if (ny < 0 || ny >= h) continue;
      for (let ox = -1; ox <= 1; ox++) {
        const nx = x + ox;
        if (nx < 0 || nx >= w) continue;
        const j = ny * w + nx;
        if (map[j] === 0) {
          map[j] = 2;
          stack.push(j);
        }
      }
    }
  }
  const edges = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) edges[i] = map[i] === 2 ? 1 : 0;
  return edges;
}

/** Canny edge map (0/1) of an RGB image. */
export function extractEdges(rgb, cfg) {
  const { width: w, height: h } = rgb;
  let gray = rgbToGray(rgb);
  if (cfg.blur_kernel > 0) gray = gaussianBlur8(gray, w, h, cfg.blur_kernel);
  return canny(gray, w, h, cfg.canny_low, cfg.canny_high);
}

const DIST_SHIFT = 16;
const INIT_DIST0 = 2147483647;

/**
 * ``cv2.distanceTransform(non_edge, DIST_L2, 5)``: distance of every pixel to
 * the nearest edge pixel with OpenCV's 5x5 chamfer mask (1, 1.4, 2.1969) in
 * 16-bit fixed point.
 */
export function distanceToNearestEdge(edges, w, h) {
  const HV = Math.round(1 * (1 << DIST_SHIFT));
  const DIAG = Math.round(1.4 * (1 << DIST_SHIFT));
  const LONG = Math.round(2.1969 * (1 << DIST_SHIFT));
  const B = 2;
  const W = w + 2 * B;
  const tmp = new Float64Array(W * (h + 2 * B)).fill(INIT_DIST0);
  const t = (x, y) => tmp[(y + B) * W + x + B];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let d;
      if (edges[y * w + x]) {
        d = 0;
      } else {
        d = Math.min(
          t(x - 1, y - 2) + LONG, t(x + 1, y - 2) + LONG,
          t(x - 2, y - 1) + LONG, t(x - 1, y - 1) + DIAG, t(x, y - 1) + HV, t(x + 1, y - 1) + DIAG, t(x + 2, y - 1) + LONG,
          t(x - 1, y) + HV,
        );
      }
      tmp[(y + B) * W + x + B] = d;
    }
  }
  const out = new Float32Array(w * h);
  const scale = Math.fround(1 / (1 << DIST_SHIFT));
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      let d = t(x, y);
      if (d > HV) {
        d = Math.min(
          d,
          t(x + 1, y + 2) + LONG, t(x - 1, y + 2) + LONG,
          t(x + 2, y + 1) + LONG, t(x + 1, y + 1) + DIAG, t(x, y + 1) + HV, t(x - 1, y + 1) + DIAG, t(x - 2, y + 1) + LONG,
          t(x + 1, y) + HV,
        );
        tmp[(y + B) * W + x + B] = d;
      }
      out[y * w + x] = d * scale;
    }
  }
  return out;
}

/**
 * Symmetric truncated Chamfer edge similarity:
 * ``100 * (1 - 0.5 * (mean_norm_dist_ref->cand + mean_norm_dist_cand->ref))``.
 */
export function computeEdgeSimilarity(ref, cand, cfg) {
  checkPair(ref, cand);
  const { width: w, height: h } = ref;
  const eRef = extractEdges(ref, cfg);
  const eCand = extractEdges(cand, cfg);
  let nRef = 0;
  let nCand = 0;
  for (let i = 0; i < w * h; i++) {
    nRef += eRef[i];
    nCand += eCand[i];
  }
  const fracRef = nRef / (w * h);
  const fracCand = nCand / (w * h);
  const result = (score, a, b) => ({ score, chamferRefToCand: a, chamferCandToRef: b, edgeFractionRef: fracRef, edgeFractionCand: fracCand });

  if (fracRef < cfg.min_edge_fraction && fracCand < cfg.min_edge_fraction) {
    logger.debug(`Edge metric not available: too few edges (${fracRef.toFixed(5)}, ${fracCand.toFixed(5)})`);
    return result(null, null, null);
  }
  if (nRef === 0 || nCand === 0) return result(0.0, 1.0, 1.0);

  const maxD = cfg.max_distance * (h / 512.0);
  const dtRef = distanceToNearestEdge(eRef, w, h);
  const dtCand = distanceToNearestEdge(eCand, w, h);
  let sRef = 0;
  let sCand = 0;
  for (let i = 0; i < w * h; i++) {
    if (eRef[i]) sRef += Math.min(dtCand[i] / maxD, 1.0);
    if (eCand[i]) sCand += Math.min(dtRef[i] / maxD, 1.0);
  }
  const dRefToCand = sRef / nRef;
  const dCandToRef = sCand / nCand;
  const score = 100.0 * (1.0 - 0.5 * (dRefToCand + dCandToRef));
  return result(Math.min(100, Math.max(0, score)), dRefToCand, dCandToRef);
}

// ---------------------------------------------------------------------------
// Weighted combination
// ---------------------------------------------------------------------------
/** ``100 * clip((score - floor) / (100 - floor), 0, 1) ** gamma``. */
export function applyScoreFloor(score, floor, gamma = 1.0) {
  if (score === null) return null;
  if (floor <= 0 && gamma === 1.0) return score;
  const x = Math.min(1.0, Math.max(0.0, (score - floor) / (100.0 - floor)));
  return 100.0 * x ** gamma;
}

/** Weighted average of the available scores, renormalising the weights of missing metrics away. */
export function combineScores(scores, weights) {
  const available = {};
  for (const [k, v] of Object.entries(scores)) if (v !== null && k in weights && weights[k] > 0) available[k] = v;
  const total = Object.keys(available).reduce((s, k) => s + weights[k], 0);
  if (!Object.keys(available).length || total <= 0) return { pairScore: null, effectiveWeights: {} };
  const effective = {};
  for (const k of METRIC_NAMES) {
    if (k in available) effective[k] = Math.abs(total - 1.0) < 1e-9 ? weights[k] : weights[k] / total;
  }
  let pairScore = 0;
  for (const k of Object.keys(effective)) pairScore += effective[k] * available[k];
  return { pairScore, effectiveWeights: effective };
}
