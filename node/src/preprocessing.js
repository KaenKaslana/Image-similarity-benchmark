/**
 * Image preprocessing (port of src/preprocessing.py).
 *
 * Applied identically to reference and candidate images: load, split off
 * alpha, build a foreground mask, composite onto the background colour, crop
 * (none / center_crop / foreground_bbox), then resize to fit a square canvas
 * keeping the aspect ratio. The mask follows the same geometry.
 */

import fs from 'node:fs';
import path from 'node:path';
import { Image8, ImageLoadError, loadImage, resizePil, savePng } from './image.js';
import { getLogger, roundHalfEven } from './util.js';

export { ImageLoadError };

const logger = getLogger('preprocessing');
const f32 = Math.fround;

/**
 * Result of :func:`preprocessImage`: ``rgb`` is an RGB :class:`Image8` on the
 * canvas, ``mask`` a 0/1 ``Uint8Array`` (or ``null``).
 */
export class PreprocessedImage {
  constructor(rgb, mask, maskSource, maskReliable, meta = {}) {
    this.rgb = rgb;
    this.mask = mask;
    this.maskSource = maskSource;
    this.maskReliable = maskReliable;
    this.meta = meta;
  }

  get hasMask() {
    return this.mask !== null && this.maskReliable;
  }
}

/** ``(mask, source, reliable)`` from alpha or by thresholding the background colour. */
export function buildForegroundMask(rgb, alpha, n, cfg) {
  if (alpha !== null && (cfg.mask_mode === 'alpha' || cfg.mask_mode === 'auto')) {
    const mask = new Uint8Array(n);
    const t = Math.trunc(cfg.alpha_threshold);
    for (let i = 0; i < n; i++) mask[i] = alpha[i] >= t ? 1 : 0;
    return { mask, source: 'alpha', reliable: true };
  }
  if (cfg.mask_mode === 'alpha') return { mask: null, source: 'none', reliable: false };

  const [br, bg, bb] = cfg.background_color;
  const t = Math.trunc(cfg.mask_background_threshold);
  const mask = new Uint8Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.max(Math.abs(rgb[i * 3] - br), Math.abs(rgb[i * 3 + 1] - bg), Math.abs(rgb[i * 3 + 2] - bb));
    if (d > t) {
      mask[i] = 1;
      count++;
    }
  }
  const frac = n ? count / n : 0;
  const reliable = cfg.mask_min_foreground_fraction <= frac && frac <= cfg.mask_max_foreground_fraction;
  if (!reliable) {
    logger.debug(
      `Background-threshold mask rejected: foreground fraction ${frac.toFixed(4)} outside ` +
        `[${cfg.mask_min_foreground_fraction}, ${cfg.mask_max_foreground_fraction}]`,
    );
  }
  return { mask, source: 'background_threshold', reliable };
}

/** Alpha-composite over a solid colour in float32, rounded half-to-even like numpy. */
export function compositeOnBackground(rgb, alpha, n, background) {
  if (alpha === null) return rgb;
  const out = new Uint8Array(n * 3);
  const bgf = background.map(f32);
  for (let i = 0; i < n; i++) {
    const a = f32(alpha[i] / 255.0);
    const inv = f32(1.0 - a);
    for (let c = 0; c < 3; c++) {
      const v = f32(f32(rgb[i * 3 + c] * a) + f32(bgf[c] * inv));
      const r = roundHalfEven(v);
      out[i * 3 + c] = r < 0 ? 0 : r > 255 ? 255 : r;
    }
  }
  return out;
}

/** Crop box ``[left, top, right, bottom]`` for the configured mode, or ``null``. */
export function computeCropBox(width, height, mask, maskReliable, cfg) {
  if (cfg.crop_mode === 'none') return null;
  if (cfg.crop_mode === 'center_crop') {
    const side = Math.min(height, width);
    const left = Math.floor((width - side) / 2);
    const top = Math.floor((height - side) / 2);
    return [left, top, left + side, top + side];
  }
  if (cfg.crop_mode === 'foreground_bbox') {
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -1;
    let y1 = -1;
    if (mask !== null && maskReliable) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (mask[y * width + x]) {
            if (x < x0) x0 = x;
            if (x > x1) x1 = x;
            if (y < y0) y0 = y;
            if (y > y1) y1 = y;
          }
        }
      }
    }
    if (x1 < 0) {
      logger.warning('foreground_bbox requested but no reliable foreground mask; falling back to no crop');
      return null;
    }
    x1 += 1;
    y1 += 1;
    const side = Math.max(x1 - x0, y1 - y0);
    const pad = roundHalfEven(side * cfg.foreground_padding);
    const sidePadded = side + 2 * pad;
    const cx = (x0 + x1) / 2.0;
    const cy = (y0 + y1) / 2.0;
    const left = roundHalfEven(cx - sidePadded / 2.0);
    const top = roundHalfEven(cy - sidePadded / 2.0);
    return [left, top, left + sidePadded, top + sidePadded];
  }
  throw new Error(`Unknown crop_mode ${cfg.crop_mode}`);
}

/** Crop an interleaved buffer to ``box``; areas outside are filled with ``fill`` (per channel). */
function cropWithPadding(data, width, height, channels, box, fill) {
  const [left, top, right, bottom] = box;
  const ow = right - left;
  const oh = bottom - top;
  const out = new Uint8Array(ow * oh * channels);
  for (let i = 0; i < ow * oh; i++) for (let c = 0; c < channels; c++) out[i * channels + c] = fill[c];
  const sx0 = Math.max(left, 0);
  const sy0 = Math.max(top, 0);
  const sx1 = Math.min(right, width);
  const sy1 = Math.min(bottom, height);
  for (let y = sy0; y < sy1; y++) {
    const srcOff = (y * width + sx0) * channels;
    const dstOff = ((y - top) * ow + (sx0 - left)) * channels;
    out.set(data.subarray(srcOff, srcOff + (sx1 - sx0) * channels), dstOff);
  }
  return { data: out, width: ow, height: oh };
}

/** Resize keeping the aspect ratio and centre on the square canvas; the mask follows (bilinear, >= 128). */
export function fitToCanvas(rgb, mask, cfg) {
  const size = cfg.canvas_size;
  const { width: w, height: h } = rgb;
  const scale = size / Math.max(h, w);
  const newW = Math.max(1, roundHalfEven(w * scale));
  const newH = Math.max(1, roundHalfEven(h * scale));
  const offX = Math.floor((size - newW) / 2);
  const offY = Math.floor((size - newH) / 2);

  const resized = resizePil(rgb, newW, newH, cfg.resample);
  const canvas = new Image8(size, size, 3);
  const [br, bg, bb] = cfg.background_color;
  for (let i = 0; i < size * size; i++) {
    canvas.data[i * 3] = br;
    canvas.data[i * 3 + 1] = bg;
    canvas.data[i * 3 + 2] = bb;
  }
  for (let y = 0; y < newH; y++) {
    canvas.data.set(resized.data.subarray(y * newW * 3, (y + 1) * newW * 3), ((y + offY) * size + offX) * 3);
  }

  let outMask = null;
  if (mask !== null) {
    const m = new Image8(w, h, 1);
    for (let i = 0; i < w * h; i++) m.data[i] = mask[i] ? 255 : 0;
    const rm = resizePil(m, newW, newH, 'bilinear');
    outMask = new Uint8Array(size * size);
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) outMask[(y + offY) * size + x + offX] = rm.data[y * newW + x] >= 128 ? 1 : 0;
    }
  }
  return { rgb: canvas, mask: outMask, info: { scale, resized_size: [newW, newH], offset: [offX, offY], canvas_size: size } };
}

/** Full preprocessing pipeline on decoded pixels. */
export function preprocessDecoded(decoded, cfg, name = '<image>') {
  const { width, height } = decoded;
  const n = width * height;
  const meta = { name, original_mode: decoded.mode, original_size: [width, height], has_alpha: decoded.alpha !== null };

  const { mask: mask0, source, reliable } = buildForegroundMask(decoded.rgb, decoded.alpha, n, cfg);
  let rgbData = compositeOnBackground(decoded.rgb, decoded.alpha, n, cfg.background_color);
  let mask = mask0;
  let w = width;
  let h = height;

  const box = computeCropBox(w, h, mask, reliable, cfg);
  meta.crop_mode = cfg.crop_mode;
  meta.crop_box = box;
  if (box !== null) {
    const c = cropWithPadding(rgbData, w, h, 3, box, cfg.background_color);
    if (mask !== null) mask = cropWithPadding(mask, w, h, 1, box, [0]).data;
    rgbData = c.data;
    w = c.width;
    h = c.height;
  }

  const fit = fitToCanvas(new Image8(w, h, 3, rgbData), mask, cfg);
  Object.assign(meta, fit.info);
  if (fit.mask !== null) {
    let count = 0;
    for (const v of fit.mask) count += v;
    meta.foreground_fraction = count / fit.mask.length;
  }
  return new PreprocessedImage(fit.rgb, fit.mask, source, reliable, meta);
}

/** Load ``file`` and preprocess it. */
export function preprocessImage(file, cfg) {
  const decoded = loadImage(file);
  const result = preprocessDecoded(decoded, cfg, path.basename(file));
  result.meta.path = String(file);
  logger.debug(
    `Preprocessed ${path.basename(file)}: ${decoded.mode} [${decoded.width}, ${decoded.height}] -> canvas ` +
      `${cfg.canvas_size}, mask=${result.maskSource} (reliable=${result.maskReliable})`,
  );
  return result;
}

/** Save the preprocessed RGB image (and ``<stem>_mask.png``); returns the RGB path. */
export function savePreprocessed(image, directory, name) {
  fs.mkdirSync(directory, { recursive: true });
  const stem = path.parse(name).name;
  const rgbPath = path.join(directory, `${stem}.png`);
  savePng(image.rgb, rgbPath);
  if (image.mask !== null) {
    const m = new Image8(image.rgb.width, image.rgb.height, 1);
    for (let i = 0; i < m.data.length; i++) m.data[i] = image.mask[i] ? 255 : 0;
    savePng(m, path.join(directory, `${stem}_mask.png`));
  }
  return rgbPath;
}
