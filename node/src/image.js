/**
 * Image I/O and resampling.
 *
 * Decoding mirrors what Pillow hands to the Python benchmark: an RGB array
 * plus an alpha channel only for images Pillow treats as transparent (RGBA,
 * LA, palette + tRNS), EXIF orientation applied to JPEGs, 16-bit greyscale
 * min-max scaled to 8 bit. :func:`resizePil` is a port of Pillow's separable
 * convolution resampler (same filters, supports and 8-bit fixed-point
 * rounding), so resized pixels match Pillow's.
 */

import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import zlib from 'node:zlib';
import pngjs from 'pngjs';
import mozjpegFactory from '@jsquash/jpeg/codec/dec/mozjpeg_dec.js';

const { PNG } = pngjs;

export class ImageLoadError extends Error {}

/**
 * An 8-bit image. ``channels`` is 1 (grey / mask), 3 (RGB) or 4 (RGBA);
 * ``data`` is row-major interleaved.
 */
export class Image8 {
  constructor(width, height, channels, data = null) {
    this.width = width;
    this.height = height;
    this.channels = channels;
    this.data = data ?? new Uint8Array(width * height * channels);
  }
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------
function pngInfo(buf) {
  const info = { bitDepth: buf[24], colorType: buf[25], hasTrns: false };
  let off = 8;
  while (off + 8 <= buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString('latin1', off + 4, off + 8);
    if (type === 'tRNS') info.hasTrns = true;
    if (type === 'IDAT' || type === 'IEND') break;
    off += 12 + len;
  }
  return info;
}

const PIL_MODES = { 0: 'L', 2: 'RGB', 3: 'P', 4: 'LA', 6: 'RGBA' };

function decodePng(buf) {
  const info = pngInfo(buf);
  const sixteen = info.bitDepth === 16;
  const png = PNG.sync.read(buf, sixteen ? { skipRescale: true } : {});
  const { width, height } = png;
  const n = width * height;
  const src = png.data;
  const rgb = new Uint8Array(n * 3);
  let alpha = null;
  let mode = PIL_MODES[info.colorType] ?? 'RGB';

  if (sixteen && info.colorType === 0) {
    // Pillow opens 16-bit greyscale as mode I;16 and the benchmark min-max scales it.
    mode = 'I;16';
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = src[i * 4];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    for (let i = 0; i < n; i++) {
      const g = hi > lo ? Math.trunc(((src[i * 4] - lo) / (hi - lo)) * 255) : 0;
      rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = g;
    }
    return { width, height, rgb, alpha, mode };
  }

  const shift = sixteen ? 8 : 0; // Pillow keeps the high byte of 16-bit samples
  const transparent = info.colorType === 4 || info.colorType === 6 || (info.colorType === 3 && info.hasTrns);
  if (transparent) alpha = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    rgb[i * 3] = src[i * 4] >> shift;
    rgb[i * 3 + 1] = src[i * 4 + 1] >> shift;
    rgb[i * 3 + 2] = src[i * 4 + 2] >> shift;
    if (alpha) alpha[i] = src[i * 4 + 3] >> shift;
  }
  return { width, height, rgb, alpha, mode };
}

/** EXIF orientation (1-8) from a JPEG's APP1 segment, 1 when absent. */
function jpegOrientation(buf) {
  let off = 2;
  while (off + 4 <= buf.length && buf[off] === 0xff) {
    const marker = buf[off + 1];
    const len = buf.readUInt16BE(off + 2);
    if (marker === 0xe1 && buf.toString('latin1', off + 4, off + 10) === 'Exif\0\0') {
      const tiff = off + 10;
      const le = buf.toString('latin1', tiff, tiff + 2) === 'II';
      const u16 = (p) => (le ? buf.readUInt16LE(p) : buf.readUInt16BE(p));
      const u32 = (p) => (le ? buf.readUInt32LE(p) : buf.readUInt32BE(p));
      const ifd = tiff + u32(tiff + 4);
      const count = u16(ifd);
      for (let i = 0; i < count; i++) {
        const entry = ifd + 2 + i * 12;
        if (u16(entry) === 0x0112) return u16(entry + 8);
      }
      return 1;
    }
    if (marker === 0xda) break;
    off += 2 + len;
  }
  return 1;
}

/** Apply an EXIF orientation to an interleaved buffer (ImageOps.exif_transpose). */
function orient(width, height, channels, data, orientation) {
  if (orientation <= 1 || orientation > 8) return { width, height, data };
  const swap = orientation >= 5;
  const ow = swap ? height : width;
  const oh = swap ? width : height;
  const out = new Uint8Array(data.length);
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      let sx;
      let sy;
      switch (orientation) {
        case 2: sx = width - 1 - x; sy = y; break;
        case 3: sx = width - 1 - x; sy = height - 1 - y; break;
        case 4: sx = x; sy = height - 1 - y; break;
        case 5: sx = y; sy = x; break;
        case 6: sx = y; sy = height - 1 - x; break;
        case 7: sx = width - 1 - y; sy = height - 1 - x; break;
        case 8: sx = width - 1 - y; sy = x; break;
        default: sx = x; sy = y;
      }
      const s = (sy * width + sx) * channels;
      const d = (y * ow + x) * channels;
      for (let c = 0; c < channels; c++) out[d + c] = data[s + c];
    }
  }
  return { width: ow, height: oh, data: out };
}

/**
 * libjpeg-turbo (mozjpeg's decoder) compiled to WebAssembly, the same decoder
 * Pillow's wheels use, so decoded pixels match the Python benchmark exactly.
 * Instantiated synchronously on first use.
 */
let mozjpeg = null;

function jpegDecoder() {
  if (mozjpeg === null) {
    const require = createRequire(import.meta.url);
    const wasmPath = path.join(path.dirname(require.resolve('@jsquash/jpeg/codec/dec/mozjpeg_dec.js')), 'mozjpeg_dec.wasm');
    const wasmModule = new WebAssembly.Module(fs.readFileSync(wasmPath));
    const module = {
      noInitialRun: true,
      instantiateWasm: (imports, callback) => {
        const instance = new WebAssembly.Instance(wasmModule, imports);
        callback(instance);
        return instance.exports;
      },
    };
    mozjpegFactory(module); // with a precompiled module this completes synchronously
    if (typeof module.decode !== 'function') throw new Error('mozjpeg decoder did not initialise synchronously');
    mozjpeg = module;
  }
  return mozjpeg;
}

function decodeJpeg(buf) {
  const img = jpegDecoder().decode(buf, true); // preserveOrientation: EXIF is applied below, like Pillow
  if (!img) throw new Error('corrupt JPEG data');
  const n = img.width * img.height;
  const rgb = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    rgb[i * 3] = img.data[i * 4];
    rgb[i * 3 + 1] = img.data[i * 4 + 1];
    rgb[i * 3 + 2] = img.data[i * 4 + 2];
  }
  const o = orient(img.width, img.height, 3, rgb, jpegOrientation(buf));
  return { width: o.width, height: o.height, rgb: o.data, alpha: null, mode: 'RGB' };
}

/**
 * Load an image file as ``{width, height, rgb, alpha, mode}``.
 *
 * @throws {ImageLoadError} when the file is missing, unsupported or corrupt.
 */
export function loadImage(path) {
  if (!fs.existsSync(path) || !fs.statSync(path).isFile()) throw new ImageLoadError(`Image file not found: ${path}`);
  const buf = fs.readFileSync(path);
  try {
    if (buf.length >= 8 && buf.readUInt32BE(0) === 0x89504e47) return decodePng(buf);
    if (buf.length >= 3 && buf[0] === 0xff && buf[1] === 0xd8) return decodeJpeg(buf);
  } catch (exc) {
    throw new ImageLoadError(`Cannot read image ${path}: ${exc.message}`);
  }
  throw new ImageLoadError(`Cannot read image ${path}: unsupported format (PNG and JPEG are supported)`);
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------
/** Write an 8-bit image (1, 3 or 4 channels) as PNG. */
export function savePng(img, path) {
  const png = new PNG({ width: img.width, height: img.height });
  const n = img.width * img.height;
  const out = png.data;
  for (let i = 0; i < n; i++) {
    if (img.channels === 1) {
      out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = img.data[i];
      out[i * 4 + 3] = 255;
    } else {
      out[i * 4] = img.data[i * img.channels];
      out[i * 4 + 1] = img.data[i * img.channels + 1];
      out[i * 4 + 2] = img.data[i * img.channels + 2];
      out[i * 4 + 3] = img.channels === 4 ? img.data[i * 4 + 3] : 255;
    }
  }
  const colorType = img.channels === 1 ? 0 : img.channels === 3 ? 2 : 6;
  fs.writeFileSync(path, PNG.sync.write(png, { colorType, inputColorType: 6, deflateLevel: zlib.constants.Z_DEFAULT_COMPRESSION }));
  return path;
}

// ---------------------------------------------------------------------------
// Pillow-compatible resampling
// ---------------------------------------------------------------------------
function sinc(x) {
  if (x === 0.0) return 1.0;
  x *= Math.PI;
  return Math.sin(x) / x;
}

const FILTERS = {
  bilinear: { support: 1.0, fn: (x) => { x = Math.abs(x); return x < 1.0 ? 1.0 - x : 0.0; } },
  bicubic: {
    support: 2.0,
    fn: (x) => {
      const a = -0.5;
      x = Math.abs(x);
      if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
      if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
      return 0.0;
    },
  },
  lanczos: { support: 3.0, fn: (x) => (x >= -3.0 && x < 3.0 ? sinc(x) * sinc(x / 3) : 0.0) },
};

const PRECISION_BITS = 32 - 8 - 2;

/** Pillow ``precompute_coeffs`` + ``normalize_coeffs_8bpc``. */
function coefficients(inSize, outSize, filter) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const ss = 1.0 / filterscale;
  const k = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0.0;
    for (let x = 0; x < xmax; x++) {
      const w = filter.fn((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0.0 ? k[x] / ww : k[x];
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * (1 << PRECISION_BITS)) : Math.trunc(0.5 + v * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

function clip8(v) {
  // v is the fixed-point sum; Pillow shifts right by PRECISION_BITS (arithmetic shift).
  const r = Math.floor(v / (1 << PRECISION_BITS));
  return r < 0 ? 0 : r > 255 ? 255 : r;
}

/**
 * Resize an interleaved 8-bit image like ``PIL.Image.resize`` (horizontal
 * pass, then vertical pass, both rounded to 8 bit).
 */
export function resizePil(img, outW, outH, mode = 'lanczos') {
  const filter = FILTERS[mode];
  if (!filter) throw new Error(`unknown resample mode ${mode}`);
  const C = img.channels;
  let cur = img;
  const half = 1 << (PRECISION_BITS - 1);
  if (outW !== cur.width) {
    const { ksize, bounds, kk } = coefficients(cur.width, outW, filter);
    const out = new Image8(outW, cur.height, C);
    const src = cur.data;
    const dst = out.data;
    for (let y = 0; y < cur.height; y++) {
      const row = y * cur.width * C;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2];
        const xmax = bounds[xx * 2 + 1];
        const kb = xx * ksize;
        for (let c = 0; c < C; c++) {
          let s = half;
          for (let x = 0; x < xmax; x++) s += src[row + (x + xmin) * C + c] * kk[kb + x];
          dst[(y * outW + xx) * C + c] = clip8(s);
        }
      }
    }
    cur = out;
  }
  if (outH !== cur.height) {
    const { ksize, bounds, kk } = coefficients(cur.height, outH, filter);
    const W = cur.width;
    const out = new Image8(W, outH, C);
    const src = cur.data;
    const dst = out.data;
    for (let yy = 0; yy < outH; yy++) {
      const ymin = bounds[yy * 2];
      const ymax = bounds[yy * 2 + 1];
      const kb = yy * ksize;
      for (let xx = 0; xx < W; xx++) {
        for (let c = 0; c < C; c++) {
          let s = half;
          for (let y = 0; y < ymax; y++) s += src[((y + ymin) * W + xx) * C + c] * kk[kb + y];
          dst[(yy * W + xx) * C + c] = clip8(s);
        }
      }
    }
    cur = out;
  }
  if (cur === img) cur = new Image8(img.width, img.height, C, img.data.slice());
  return cur;
}
