/**
 * Mixed-radix complex FFT for the sizes OpenCV's ``getOptimalDFTSize``
 * produces (products of 2, 3 and 5; other prime factors fall back to a
 * direct DFT of that factor).
 */

/** Smallest ``n' >= n`` of the form ``2^a 3^b 5^c`` (cv2.getOptimalDFTSize). */
export function optimalDftSize(n) {
  for (let m = Math.max(n, 1); ; m++) {
    let r = m;
    for (const p of [2, 3, 5]) while (r % p === 0) r /= p;
    if (r === 1) return m;
  }
}

function factorize(n) {
  const out = [];
  for (const p of [4, 2, 3, 5]) {
    while (n % p === 0) {
      out.push(p);
      n /= p;
    }
  }
  for (let p = 7; n > 1; p += 2) {
    while (n % p === 0) {
      out.push(p);
      n /= p;
    }
  }
  return out;
}

const planCache = new Map();

function plan(n) {
  let p = planCache.get(n);
  if (!p) {
    const cos = new Float64Array(n);
    const sin = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      cos[k] = Math.cos((2 * Math.PI * k) / n);
      sin[k] = Math.sin((2 * Math.PI * k) / n);
    }
    p = { factors: factorize(n), cos, sin };
    planCache.set(n, p);
  }
  return p;
}

/**
 * Recursive decimation in time. ``sign`` -1 = forward, +1 = inverse
 * (unscaled). Reads ``n`` samples from ``(re, im)`` at ``off + k * stride``
 * and writes them contiguously to ``(ore, oim)`` at ``oOff``.
 */
function fftRec(re, im, off, stride, n, ore, oim, oOff, factors, fi, tw, twStep, sign) {
  if (n === 1) {
    ore[oOff] = re[off];
    oim[oOff] = im[off];
    return;
  }
  const p = factors[fi];
  const m = n / p;
  for (let q = 0; q < p; q++) {
    fftRec(re, im, off + q * stride, stride * p, m, ore, oim, oOff + q * m, factors, fi + 1, tw, twStep * p, sign);
  }
  const N = tw.cos.length;
  const tr = new Float64Array(p);
  const ti = new Float64Array(p);
  for (let k = 0; k < m; k++) {
    for (let q = 0; q < p; q++) {
      const idx = (q * k * twStep) % N;
      const c = tw.cos[idx];
      const s = sign * tw.sin[idx];
      const xr = ore[oOff + q * m + k];
      const xi = oim[oOff + q * m + k];
      tr[q] = xr * c - xi * s;
      ti[q] = xr * s + xi * c;
    }
    for (let j = 0; j < p; j++) {
      let sr = 0;
      let si = 0;
      for (let q = 0; q < p; q++) {
        const idx = (((q * j * m) % n) * twStep) % N;
        const c = tw.cos[idx];
        const s = sign * tw.sin[idx];
        sr += tr[q] * c - ti[q] * s;
        si += tr[q] * s + ti[q] * c;
      }
      ore[oOff + j * m + k] = sr;
      oim[oOff + j * m + k] = si;
    }
  }
}

/** In-place 1-D FFT of ``n`` complex samples at ``off + k * stride``. */
function fft1d(re, im, off, stride, n, sign, scratch) {
  const p = plan(n);
  const { re: sr, im: si, ore, oim } = scratch;
  for (let k = 0; k < n; k++) {
    sr[k] = re[off + k * stride];
    si[k] = im[off + k * stride];
  }
  fftRec(sr, si, 0, 1, n, ore, oim, 0, p.factors, 0, p, 1, sign);
  for (let k = 0; k < n; k++) {
    re[off + k * stride] = ore[k];
    im[off + k * stride] = oim[k];
  }
}

/** In-place 2-D FFT of a ``rows x cols`` complex array (row-major). ``inverse`` is unscaled. */
export function fft2d(re, im, rows, cols, inverse = false) {
  const sign = inverse ? 1 : -1;
  const n = Math.max(rows, cols);
  const scratch = { re: new Float64Array(n), im: new Float64Array(n), ore: new Float64Array(n), oim: new Float64Array(n) };
  for (let r = 0; r < rows; r++) fft1d(re, im, r * cols, 1, cols, sign, scratch);
  for (let c = 0; c < cols; c++) fft1d(re, im, c, cols, rows, sign, scratch);
}
