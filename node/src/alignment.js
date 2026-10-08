/**
 * Translation alignment of a candidate image to its reference (port of
 * src/alignment.py). The candidate is only shifted, never scaled or rotated.
 *
 * ``phase_correlation`` re-implements ``cv2.phaseCorrelate`` (Hanning
 * window, optimal DFT size padding, normalised cross-power spectrum, 5x5
 * weighted centroid around the peak).
 */

import { optimalDftSize, fft2d } from './fft.js';
import { Image8 } from './image.js';
import { PreprocessedImage } from './preprocessing.js';
import { getLogger, roundHalfEven } from './util.js';

const logger = getLogger('alignment');
const FLT_EPSILON = 1.1920928955078125e-7;
const DBL_EPSILON = 2.220446049250313e-16;

/** Float ``(H*W)`` map in [0, 1]: the mask, or the per-pixel distance from the background colour. */
export function foregroundSignal(img, background, useMask) {
  const n = img.rgb.width * img.rgb.height;
  const out = new Float32Array(n);
  if (useMask && img.mask !== null) {
    for (let i = 0; i < n; i++) out[i] = img.mask[i];
    return out;
  }
  const d = img.rgb.data;
  for (let i = 0; i < n; i++) {
    out[i] = Math.max(Math.abs(d[i * 3] - background[0]), Math.abs(d[i * 3 + 1] - background[1]), Math.abs(d[i * 3 + 2] - background[2])) / 255.0;
  }
  return out;
}

function centroid(signal, w, h) {
  let total = 0;
  let sx = 0;
  let sy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = signal[y * w + x];
      total += v;
      sx += x * v;
      sy += y * v;
    }
  }
  return total <= 0 ? null : [sx / total, sy / total];
}

/** Shift ``[dx, dy]`` moving the candidate centroid onto the reference centroid. */
export function estimateShiftCentroid(sigRef, sigCand, w, h) {
  const a = centroid(sigRef, w, h);
  const b = centroid(sigCand, w, h);
  if (a === null || b === null) return null;
  return [a[0] - b[0], a[1] - b[1]];
}

/** ``cv2.createHanningWindow``: ``sqrt(wr[i] * wc[j])``. */
function hanningWindow(w, h) {
  const wc = new Float64Array(w);
  const c0 = (2 * Math.PI) / (w - 1);
  const c1 = (2 * Math.PI) / (h - 1);
  for (let j = 0; j < w; j++) wc[j] = 0.5 * (1 - Math.cos(c0 * j));
  const out = new Float32Array(w * h);
  for (let i = 0; i < h; i++) {
    const wr = 0.5 * (1 - Math.cos(c1 * i));
    for (let j = 0; j < w; j++) out[i * w + j] = Math.sqrt(wr * wc[j]);
  }
  return out;
}

/**
 * ``cv2.phaseCorrelate(sigRef, sigCand, hanning)``: returns the translation
 * of the candidate relative to the reference and the peak response.
 *
 * Side effect reproduced from OpenCV: when the images already have an
 * optimal DFT size (512 does) OpenCV skips the padding copy and multiplies
 * the caller's arrays by the window in place. The Python benchmark therefore
 * runs its overlap check on the windowed signals; doing the same here keeps
 * both implementations choosing the same shifts.
 */
export function phaseCorrelate(sigRef, sigCand, w, h) {
  const win = hanningWindow(w, h);
  const M = optimalDftSize(h);
  const N = optimalDftSize(w);
  if (M === h && N === w) {
    for (let i = 0; i < w * h; i++) {
      sigRef[i] *= win[i];
      sigCand[i] *= win[i];
    }
  }
  const ar = new Float64Array(M * N);
  const ai = new Float64Array(M * N);
  const br = new Float64Array(M * N);
  const bi = new Float64Array(M * N);
  const padded = M !== h || N !== w;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      ar[y * N + x] = padded ? Math.fround(sigRef[i] * win[i]) : sigRef[i];
      br[y * N + x] = padded ? Math.fround(sigCand[i] * win[i]) : sigCand[i];
    }
  }
  fft2d(ar, ai, M, N);
  fft2d(br, bi, M, N);
  // C = (A conj(B)) / |A conj(B)|, with OpenCV's divSpectrums epsilon.
  for (let k = 0; k < M * N; k++) {
    const pr = ar[k] * br[k] + ai[k] * bi[k];
    const pi = ai[k] * br[k] - ar[k] * bi[k];
    const mag = Math.sqrt(pr * pr + pi * pi);
    const denom = mag * mag + FLT_EPSILON;
    ar[k] = (pr * mag) / denom;
    ai[k] = (pi * mag) / denom;
  }
  fft2d(ar, ai, M, N, true);
  // fftShift + peak search
  const cx = Math.floor(N / 2);
  const cy = Math.floor(M / 2);
  const C = new Float64Array(M * N);
  for (let y = 0; y < M; y++) {
    for (let x = 0; x < N; x++) C[((y + cy) % M) * N + ((x + cx) % N)] = ar[y * N + x];
  }
  let peak = 0;
  for (let k = 1; k < M * N; k++) if (C[k] > C[peak]) peak = k;
  const px = peak % N;
  const py = Math.floor(peak / N);
  // 5x5 weighted centroid
  const minr = Math.max(py - 2, 0);
  const maxr = Math.min(py + 2, M - 1);
  const minc = Math.max(px - 2, 0);
  const maxc = Math.min(px + 2, N - 1);
  let sx = 0;
  let sy = 0;
  let sum = 0;
  for (let y = minr; y <= maxr; y++) {
    for (let x = minc; x <= maxc; x++) {
      const v = C[y * N + x];
      sx += v * x;
      sy += v * y;
      sum += v;
    }
  }
  const response = sum / (M * N);
  sum += DBL_EPSILON;
  return { shift: [N / 2 - sx / sum, M / 2 - sy / sum], response };
}

/** Shift ``[dx, dy]`` to apply to the candidate (the negated phase correlation result). */
export function estimateShiftPhaseCorrelation(sigRef, sigCand, w, h) {
  const { shift, response } = phaseCorrelate(sigRef, sigCand, w, h);
  return { shift: [-shift[0], -shift[1]], response };
}

/** Integer translation of an interleaved buffer; uncovered pixels get ``fill``. */
function shiftBuffer(src, w, h, channels, dx, dy, fill, Ctor = Uint8Array) {
  const out = new Ctor(src.length);
  for (let i = 0; i < w * h; i++) for (let c = 0; c < channels; c++) out[i * channels + c] = fill[c];
  for (let y = Math.max(0, dy); y < Math.min(h, h + dy); y++) {
    const x0 = Math.max(0, dx);
    const x1 = Math.min(w, w + dx);
    if (x1 <= x0) break;
    const srcOff = ((y - dy) * w + (x0 - dx)) * channels;
    out.set(src.subarray(srcOff, srcOff + (x1 - x0) * channels), (y * w + x0) * channels);
  }
  return out;
}

/** Copy of ``img`` shifted by integer ``(dx, dy)``; uncovered areas get the background colour / mask 0. */
export function translateImage(img, dx, dy, background) {
  const { width: w, height: h } = img.rgb;
  const rgb = new Image8(w, h, 3, shiftBuffer(img.rgb.data, w, h, 3, dx, dy, background));
  const mask = img.mask === null ? null : shiftBuffer(img.mask, w, h, 1, dx, dy, [0]);
  return new PreprocessedImage(rgb, mask, img.maskSource, img.maskReliable, { ...img.meta, alignment_shift_px: [dx, dy] });
}

function overlap(a, b) {
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  const denom = Math.sqrt(aa * bb);
  return denom <= 0 ? 0 : ab / denom;
}

function alignmentInfo(method, shiftPx, shiftFraction, response, signal, applied, note = '') {
  return { method, shift_px: shiftPx, shift_fraction: shiftFraction, response, signal, applied, note };
}

/**
 * Shift ``cand`` so its object overlaps the one in ``ref``. Returns
 * ``{image, info}``; the shift is rejected when it exceeds
 * ``alignment_max_shift`` or would reduce the overlap of the foreground signals.
 */
export function alignCandidate(ref, cand, cfg) {
  const method = cfg.alignment;
  const size = cfg.canvas_size;
  if (method === 'none') return { image: cand, info: alignmentInfo('none', [0, 0], [0, 0], null, 'n/a', false) };

  const { width: w, height: h } = ref.rgb;
  const useMask = ref.hasMask && cand.hasMask;
  const sigRef = foregroundSignal(ref, cfg.background_color, useMask);
  const sigCand = foregroundSignal(cand, cfg.background_color, useMask);
  const signal = useMask ? 'mask' : 'background_distance';
  let response = null;
  let fdx;
  let fdy;
  if (method === 'centroid') {
    const est = estimateShiftCentroid(sigRef, sigCand, w, h);
    if (est === null) return { image: cand, info: alignmentInfo(method, [0, 0], [0, 0], null, signal, false, 'empty foreground signal') };
    [fdx, fdy] = est;
  } else if (method === 'phase_correlation') {
    const est = estimateShiftPhaseCorrelation(sigRef, sigCand, w, h);
    [fdx, fdy] = est.shift;
    response = est.response;
  } else {
    throw new Error(`unknown alignment method ${method}`);
  }

  const dx = roundHalfEven(fdx) || 0;
  const dy = roundHalfEven(fdy) || 0;
  const frac = [dx / size, dy / size];
  const name = ref.meta.name ?? '<pair>';
  const maxShift = cfg.alignment_max_shift;
  if (Math.abs(frac[0]) > maxShift || Math.abs(frac[1]) > maxShift) {
    const note = `estimated shift (${frac[0]}, ${frac[1]}) exceeds alignment_max_shift=${maxShift}; not applied`;
    logger.warning(`${name}: ${note}`);
    return { image: cand, info: alignmentInfo(method, [dx, dy], frac, response, signal, false, note) };
  }
  if (dx === 0 && dy === 0) {
    return { image: cand, info: alignmentInfo(method, [0, 0], [0, 0], response, signal, true, 'already aligned') };
  }

  // Only apply the shift if it increases the overlap of the foreground signals.
  const before = overlap(sigRef, sigCand);
  const after = overlap(sigRef, shiftBuffer(sigCand, w, h, 1, dx, dy, [0], Float32Array));
  if (after < before) {
    const note = `shift (${dx}, ${dy}) would reduce overlap (${before.toFixed(3)} -> ${after.toFixed(3)}); not applied`;
    logger.info(`${name}: ${note}`);
    return { image: cand, info: alignmentInfo(method, [dx, dy], frac, response, signal, false, note) };
  }
  logger.debug(`${name}: aligning candidate by (${dx}, ${dy}) px via ${method}`);
  return { image: translateImage(cand, dx, dy, cfg.background_color), info: alignmentInfo(method, [dx, dy], frac, response, signal, true) };
}
