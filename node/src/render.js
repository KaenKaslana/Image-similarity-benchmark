/**
 * Orthographic multi-view rendering of 3D meshes (port of src/render.py).
 *
 * A software z-buffer rasteriser: the mesh is rotated into the canonical
 * frame (+Y up, +Z front), centred and scaled to unit max extent, then each
 * view is projected orthographically, flat headlight shaded and written as an
 * RGBA PNG with a transparent background (so the benchmark gets an exact
 * silhouette mask from alpha). Both models of a comparison must use the same
 * render options.
 */

import fs from 'node:fs';
import path from 'node:path';
import { Image8, savePng } from './image.js';
import { loadMeshFile, MeshLoadError } from './mesh.js';
import { getLogger, roundHalfEven } from './util.js';

const logger = getLogger('render');

export class RenderError extends Error {}

export const AXES = {
  '+x': [1, 0, 0],
  '-x': [-1, 0, 0],
  '+y': [0, 1, 0],
  '-y': [0, -1, 0],
  '+z': [0, 0, 1],
  '-z': [0, 0, -1],
};
export const AXIS_NAMES = Object.keys(AXES);

const isoNorm = Math.hypot(1.0, 0.8, 1.0);
const ISO_FORWARD = [-1.0 / isoNorm, -0.8 / isoNorm, -1.0 / isoNorm];
// (forward direction the camera looks along, image-up direction), third-angle projection
export const VIEWS = {
  front: ['-z', '+y'],
  back: ['+z', '+y'],
  side: ['-x', '+y'],
  left: ['+x', '+y'],
  top: ['-y', '-z'],
  bottom: ['+y', '+z'],
  iso: [ISO_FORWARD, '+y'],
};
export const ORTHO_VIEWS = ['front', 'back', 'side', 'left', 'top', 'bottom'];
export const DEFAULT_VIEWS = ORTHO_VIEWS;
export const STYLES = ['shaded', 'silhouette'];

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function defaultRenderOptions() {
  return { size: 512, views: [...DEFAULT_VIEWS], up: '+y', front: '+z', fill: 0.85, style: 'shaded', ambient: 0.35, supersample: 2, gray_min: 60, gray_max: 210 };
}

export function validateRenderOptions(o) {
  if (o.size < 16) throw new RenderError('size must be >= 16');
  if (!o.views.length) throw new RenderError('at least one view is required');
  for (const v of o.views) if (!(v in VIEWS)) throw new RenderError(`unknown view '${v}'; choose from ${Object.keys(VIEWS).join(', ')}`);
  if (!(o.up in AXES) || !(o.front in AXES)) throw new RenderError(`up/front must be one of ${AXIS_NAMES.join(', ')}`);
  if (Math.abs(dot(AXES[o.up], AXES[o.front])) > 1e-9) throw new RenderError(`up (${o.up}) and front (${o.front}) must be perpendicular`);
  if (!(o.fill >= 0.1 && o.fill <= 1.0)) throw new RenderError('fill must be in [0.1, 1.0]');
  if (!STYLES.includes(o.style)) throw new RenderError(`style must be one of ${STYLES.join(', ')}`);
  if (!(o.ambient >= 0 && o.ambient <= 1)) throw new RenderError('ambient must be in [0, 1]');
  if (o.supersample < 1) throw new RenderError('supersample must be >= 1');
}

/**
 * A mesh in the canonical frame, centred and scaled to unit max extent:
 * ``vertices`` Float64Array(V*3), ``faces`` Uint32Array(F*3),
 * ``faceNormals`` Float64Array(F*3).
 */
export class LoadedMesh {
  constructor(vertices, faces, faceNormals, source, originalExtents, meta = {}) {
    this.vertices = vertices;
    this.faces = faces;
    this.faceNormals = faceNormals;
    this.source = source;
    this.originalExtents = originalExtents;
    this.meta = meta;
  }

  get numVertices() {
    return this.vertices.length / 3;
  }

  get numFaces() {
    return this.faces.length / 3;
  }
}

// ---------------------------------------------------------------------------
// Loading and normalisation
// ---------------------------------------------------------------------------
/** The 24 right-handed ``[up, front]`` axis pairs. */
export function allOrientations() {
  const out = [];
  for (const u of AXIS_NAMES) for (const f of AXIS_NAMES) if (Math.abs(dot(AXES[u], AXES[f])) < 1e-9) out.push([u, f]);
  return out;
}

/** Rotation ``R`` (3x3 rows) with ``R @ up = +Y`` and ``R @ front = +Z``. */
export function canonicalRotation(up, front) {
  const y = AXES[up];
  const z = AXES[front];
  return [cross(y, z), [...y], [...z]];
}

/** Rotation about +Y; positive turns the front towards +X. */
export function yawMatrix(degrees) {
  const t = (degrees * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
}

export function matMul3(a, b) {
  return a.map((row) => [0, 1, 2].map((j) => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j]));
}

/** ``points @ R.T`` for an interleaved xyz array. */
export function applyRotation(points, R) {
  const out = new Float64Array(points.length);
  for (let i = 0; i < points.length; i += 3) {
    const x = points[i];
    const y = points[i + 1];
    const z = points[i + 2];
    out[i] = x * R[0][0] + y * R[0][1] + z * R[0][2];
    out[i + 1] = x * R[1][0] + y * R[1][1] + z * R[1][2];
    out[i + 2] = x * R[2][0] + y * R[2][1] + z * R[2][2];
  }
  return out;
}

export function bounds(vertices) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < vertices.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = vertices[i + k];
      if (v < lo[k]) lo[k] = v;
      if (v > hi[k]) hi[k] = v;
    }
  }
  return { lo, hi };
}

function normalise(vertices) {
  const { lo, hi } = bounds(vertices);
  const centre = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
  const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const out = new Float64Array(vertices.length);
  for (let i = 0; i < vertices.length; i += 3) for (let k = 0; k < 3; k++) out[i + k] = (vertices[i + k] - centre[k]) / extent;
  return { vertices: out, extent };
}

/** Rotate a canonical mesh and re-normalise centre / scale. */
export function rotate(mesh, R, note = null) {
  const { vertices } = normalise(applyRotation(mesh.vertices, R));
  const normals = applyRotation(mesh.faceNormals, R);
  return new LoadedMesh(vertices, mesh.faces, normals, mesh.source, mesh.originalExtents, { ...mesh.meta, ...(note ?? {}) });
}

/** Re-interpret an already canonical mesh with a new up/front. */
export function reorient(mesh, up, front) {
  return rotate(mesh, canonicalRotation(up, front), { reoriented: { up, front } });
}

function faceNormals(positions, faces) {
  const out = new Float64Array(faces.length);
  for (let f = 0; f < faces.length; f += 3) {
    const a = faces[f] * 3;
    const b = faces[f + 1] * 3;
    const c = faces[f + 2] * 3;
    const u = [positions[b] - positions[a], positions[b + 1] - positions[a + 1], positions[b + 2] - positions[a + 2]];
    const v = [positions[c] - positions[a], positions[c + 1] - positions[a + 1], positions[c + 2] - positions[a + 2]];
    const n = cross(u, v);
    const len = Math.hypot(n[0], n[1], n[2]);
    if (len > 1e-13) {
      out[f] = n[0] / len;
      out[f + 1] = n[1] / len;
      out[f + 2] = n[2] / len;
    }
  }
  return out;
}

/** Load a mesh file, rotate it into the canonical frame and normalise it. */
export function loadMesh(file, up = '+y', front = '+z') {
  let raw;
  try {
    raw = loadMeshFile(file);
  } catch (exc) {
    if (exc instanceof MeshLoadError) throw new RenderError(exc.message);
    throw exc;
  }
  const { lo, hi } = bounds(raw.positions);
  const originalExtents = [0, 1, 2].map((k) => hi[k] - lo[k]);
  const rot = canonicalRotation(up, front);
  const normals = applyRotation(faceNormals(raw.positions, raw.faces), rot);
  const { vertices, extent } = normalise(applyRotation(raw.positions, rot));
  if (!Number.isFinite(extent) || extent <= 0) throw new RenderError(`${path.basename(file)} has a degenerate (zero-size) bounding box`);
  const meta = { vertices: raw.vertexCount, faces: raw.faces.length / 3, original_extents: originalExtents, scale: extent };
  logger.info(`Loaded ${path.basename(file)}: ${meta.vertices} vertices, ${meta.faces} faces`);
  return new LoadedMesh(vertices, raw.faces, normals, file, originalExtents, meta);
}

// ---------------------------------------------------------------------------
// Rasterisation
// ---------------------------------------------------------------------------
/**
 * Z-buffer rasterisation of flat-valued triangles. ``xy`` holds 6 pixel
 * coordinates per triangle (pixel ``i`` covers ``[i, i+1)``), ``depth`` 3
 * depths per triangle (smaller is closer), ``value`` one value per triangle.
 * Returns ``{zbuf, vbuf}``; ``zbuf`` is ``+Infinity`` where nothing was drawn.
 */
export function rasterize(xy, depth, value, size, zbuf = null, vbuf = null) {
  zbuf = zbuf ?? new Float64Array(size * size).fill(Infinity);
  vbuf = vbuf ?? new Float32Array(size * size);
  const n = depth.length / 3;
  for (let t = 0; t < n; t++) {
    const ax = xy[t * 6], ay = xy[t * 6 + 1];
    const bx = xy[t * 6 + 2], by = xy[t * 6 + 3];
    const cx = xy[t * 6 + 4], cy = xy[t * 6 + 5];
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (!(Math.abs(area) > 1e-12)) continue;
    const minX = Math.min(ax, bx, cx), maxX = Math.max(ax, bx, cx);
    const minY = Math.min(ay, by, cy), maxY = Math.max(ay, by, cy);
    if (maxX < 0 || minX > size || maxY < 0 || minY > size) continue;
    const xmin = Math.max(0, Math.min(size - 1, Math.ceil(minX - 0.5)));
    const xmax = Math.max(0, Math.min(size - 1, Math.floor(maxX - 0.5)));
    const ymin = Math.max(0, Math.min(size - 1, Math.ceil(minY - 0.5)));
    const ymax = Math.max(0, Math.min(size - 1, Math.floor(maxY - 0.5)));
    if (xmax < xmin || ymax < ymin) continue;
    const s = Math.sign(area);
    const invArea = 1.0 / Math.abs(area);
    const d0 = depth[t * 3], d1 = depth[t * 3 + 1], d2 = depth[t * 3 + 2];
    const val = value[t];
    for (let py = ymin; py <= ymax; py++) {
      const fy = py + 0.5;
      for (let px = xmin; px <= xmax; px++) {
        const fx = px + 0.5;
        const w0 = ((cx - bx) * (fy - by) - (cy - by) * (fx - bx)) * s;
        if (w0 < 0) continue;
        const w1 = ((ax - cx) * (fy - cy) - (ay - cy) * (fx - cx)) * s;
        if (w1 < 0) continue;
        const w2 = ((bx - ax) * (fy - ay) - (by - ay) * (fx - ax)) * s;
        if (w2 < 0) continue;
        const z = (w0 * d0 + w1 * d1 + w2 * d2) * invArea;
        const lin = py * size + px;
        if (z < zbuf[lin]) {
          zbuf[lin] = z;
          vbuf[lin] = val;
        }
      }
    }
  }
  return { zbuf, vbuf };
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------
const axis = (spec) => (typeof spec === 'string' ? AXES[spec] : spec);

/** ``{right, up, forward}`` orthonormal vectors of a named view. */
export function viewBasis(view) {
  const [fSpec, uSpec] = VIEWS[view];
  const forward = axis(fSpec);
  let up = axis(uSpec);
  const d = dot(up, forward);
  up = [up[0] - forward[0] * d, up[1] - forward[1] * d, up[2] - forward[2] * d];
  const len = Math.hypot(up[0], up[1], up[2]);
  up = up.map((v) => v / len);
  return { right: cross(forward, up), up, forward };
}

/** Projected triangle corners (6 per face) and depths (3 per face) of a view. */
export function projectFaces(mesh, view, size, scale, centre = null) {
  const { right, up, forward } = viewBasis(view);
  const V = mesh.vertices;
  const nv = V.length / 3;
  const xs = new Float64Array(nv);
  const ys = new Float64Array(nv);
  const zs = new Float64Array(nv);
  for (let i = 0; i < nv; i++) {
    const x = V[i * 3], y = V[i * 3 + 1], z = V[i * 3 + 2];
    xs[i] = x * right[0] + y * right[1] + z * right[2];
    ys[i] = x * up[0] + y * up[1] + z * up[2];
    zs[i] = x * forward[0] + y * forward[1] + z * forward[2];
  }
  const [cu, cw] = centre ?? [0, 0];
  const F = mesh.faces;
  const nf = F.length / 3;
  const xy = new Float64Array(nf * 6);
  const depth = new Float64Array(nf * 3);
  for (let f = 0; f < nf; f++) {
    for (let k = 0; k < 3; k++) {
      const v = F[f * 3 + k];
      xy[f * 6 + k * 2] = size / 2.0 + (xs[v] - cu) * scale;
      xy[f * 6 + k * 2 + 1] = size / 2.0 - (ys[v] - cw) * scale;
      depth[f * 3 + k] = zs[v];
    }
  }
  return { xy, depth, forward };
}

/** Render one orthographic view to an RGBA :class:`Image8`. */
export function renderView(mesh, view, opts) {
  const ss = Math.trunc(opts.supersample);
  const size = Math.trunc(opts.size) * ss;
  const scale = opts.fill * size;
  const { xy, depth, forward } = projectFaces(mesh, view, size, scale);
  const nf = mesh.numFaces;
  const gray = new Float32Array(nf);
  if (opts.style !== 'silhouette') {
    const N = mesh.faceNormals;
    for (let f = 0; f < nf; f++) {
      const facing = Math.abs(N[f * 3] * forward[0] + N[f * 3 + 1] * forward[1] + N[f * 3 + 2] * forward[2]);
      const intensity = opts.ambient + (1.0 - opts.ambient) * facing;
      gray[f] = opts.gray_min + (opts.gray_max - opts.gray_min) * intensity;
    }
  }
  const { zbuf, vbuf } = rasterize(xy, depth, gray, size);
  const out = Math.trunc(opts.size);
  const rgba = new Image8(out, out, 4);
  const f32 = Math.fround;
  const inv = f32(1 / (ss * ss));
  for (let y = 0; y < out; y++) {
    for (let x = 0; x < out; x++) {
      // INTER_AREA downscale of coverage and premultiplied grey
      let cov = 0;
      let premul = 0;
      for (let dy = 0; dy < ss; dy++) {
        for (let dx = 0; dx < ss; dx++) {
          const i = (y * ss + dy) * size + x * ss + dx;
          if (zbuf[i] !== Infinity) {
            cov += 1;
            premul += vbuf[i];
          }
        }
      }
      const c = f32(cov * inv);
      const p = f32(premul * inv);
      const g = ss > 1 ? f32(p / Math.max(c, f32(1e-6))) : vbuf[y * size + x];
      const gv = Math.min(255, Math.max(0, roundHalfEven(g)));
      const o = (y * out + x) * 4;
      rgba.data[o] = rgba.data[o + 1] = rgba.data[o + 2] = gv;
      rgba.data[o + 3] = Math.min(255, Math.max(0, roundHalfEven(f32(c * 255))));
    }
  }
  return rgba;
}

/** Render every view in ``opts.views`` to ``<prefix><view>.png`` plus ``views.json``. */
export function renderViews(model, outputDir, opts = null, prefix = '') {
  opts = opts ?? defaultRenderOptions();
  validateRenderOptions(opts);
  const mesh = model instanceof LoadedMesh ? model : loadMesh(model, opts.up, opts.front);
  fs.mkdirSync(outputDir, { recursive: true });
  const written = {};
  for (const view of opts.views) {
    const file = path.join(outputDir, `${prefix}${view}.png`);
    savePng(renderView(mesh, view, opts), file);
    written[view] = file;
    logger.info(`Rendered ${view} -> ${file}`);
  }
  const meta = {
    model: String(mesh.source),
    mesh: mesh.meta,
    render: { ...opts, views: [...opts.views] },
    images: Object.fromEntries(Object.entries(written).map(([k, p]) => [k, path.basename(p)])),
  };
  fs.writeFileSync(path.join(outputDir, 'views.json'), JSON.stringify(meta, null, 2), 'utf-8');
  return written;
}

/** ``"front,side,top"`` or ``"all"`` -> list of view names. */
export function parseViews(text) {
  let parts;
  if (typeof text === 'string') {
    if (text.trim().toLowerCase() === 'all') return [...ORTHO_VIEWS];
    parts = text.split(',').map((p) => p.trim().toLowerCase()).filter(Boolean);
  } else {
    parts = text.map((p) => String(p).trim().toLowerCase());
  }
  const unknown = parts.filter((p) => !(p in VIEWS));
  if (unknown.length) throw new RenderError(`unknown view(s) ${JSON.stringify(unknown)}; choose from ${Object.keys(VIEWS).join(', ')} or 'all'`);
  if (!parts.length) throw new RenderError('no views given');
  return [...new Set(parts)];
}
