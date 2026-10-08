/**
 * Minimal glTF 2.0 / GLB reader shared by the mesh loader and the rig
 * analysis (port of ``GLTFFile`` in src/rig.py, plus sparse accessors).
 *
 * Matrices are row-major ``Float64Array(16)``.
 */

import fs from 'node:fs';
import path from 'node:path';

export class GLTFError extends Error {}

const COMPONENTS = {
  5120: { size: 1, max: 127 },
  5121: { size: 1, max: 255 },
  5122: { size: 2, max: 32767 },
  5123: { size: 2, max: 65535 },
  5125: { size: 4, max: 4294967295 },
  5126: { size: 4, max: null },
};
const TYPE_SIZES = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };

function readComponent(view, offset, componentType) {
  switch (componentType) {
    case 5120: return view.getInt8(offset);
    case 5121: return view.getUint8(offset);
    case 5122: return view.getInt16(offset, true);
    case 5123: return view.getUint16(offset, true);
    case 5125: return view.getUint32(offset, true);
    case 5126: return view.getFloat32(offset, true);
    default: throw new GLTFError(`unknown componentType ${componentType}`);
  }
}

export class GLTFFile {
  constructor(file) {
    this.path = file;
    this.name = path.basename(file);
    this.json = {};
    this.buffers = [];
    this.load();
  }

  load() {
    const data = fs.readFileSync(this.path);
    let bin = null;
    if (data.length >= 4 && data.toString('latin1', 0, 4) === 'glTF') {
      if (data.length < 20) throw new GLTFError(`${this.name}: truncated GLB header`);
      const length = data.readUInt32LE(8);
      let offset = 12;
      while (offset + 8 <= Math.min(length, data.length)) {
        const chunkLen = data.readUInt32LE(offset);
        const chunkType = data.readUInt32LE(offset + 4);
        const chunk = data.subarray(offset + 8, offset + 8 + chunkLen);
        if (chunkType === 0x4e4f534a) this.json = JSON.parse(chunk.toString('utf-8'));
        else if (chunkType === 0x004e4942 && bin === null) bin = chunk;
        offset += 8 + chunkLen;
      }
      if (!Object.keys(this.json).length) throw new GLTFError(`${this.name}: GLB without a JSON chunk`);
    } else {
      try {
        this.json = JSON.parse(data.toString('utf-8'));
      } catch (exc) {
        throw new GLTFError(`${this.name}: not a glTF/GLB file (${exc.message})`);
      }
    }
    (this.json.buffers ?? []).forEach((buf, i) => {
      const uri = buf.uri;
      if (uri === undefined) {
        if (bin === null) throw new GLTFError(`${this.name}: buffer ${i} has no uri and the GLB has no BIN chunk`);
        this.buffers.push(bin);
      } else if (uri.startsWith('data:')) {
        this.buffers.push(Buffer.from(uri.split(',', 2)[1], 'base64'));
      } else {
        const ext = path.join(path.dirname(this.path), decodeURIComponent(uri));
        if (!fs.existsSync(ext)) throw new GLTFError(`${this.name}: external buffer ${uri} not found`);
        this.buffers.push(fs.readFileSync(ext));
      }
    });
  }

  get nodes() {
    return this.json.nodes ?? [];
  }

  /**
   * Decode accessor ``index`` to ``{data: Float64Array(count * n), count, n}``.
   * Normalised integers are mapped to [-1, 1] / [0, 1] like the Python reader.
   */
  accessor(index) {
    const acc = this.json.accessors[index];
    if (!acc) throw new GLTFError(`${this.name}: accessor ${index} does not exist`);
    const n = TYPE_SIZES[acc.type];
    const comp = COMPONENTS[acc.componentType];
    if (!n || !comp) throw new GLTFError(`${this.name}: unsupported accessor type ${acc.type}/${acc.componentType}`);
    const count = acc.count;
    const out = new Float64Array(count * n);
    if (acc.bufferView !== undefined) this.readView(acc.bufferView, acc.byteOffset ?? 0, acc.componentType, n, count, out);
    if (acc.sparse) {
      const sp = acc.sparse;
      const idx = new Float64Array(sp.count);
      this.readView(sp.indices.bufferView, sp.indices.byteOffset ?? 0, sp.indices.componentType, 1, sp.count, idx);
      const vals = new Float64Array(sp.count * n);
      this.readView(sp.values.bufferView, sp.values.byteOffset ?? 0, acc.componentType, n, sp.count, vals);
      for (let i = 0; i < sp.count; i++) for (let c = 0; c < n; c++) out[idx[i] * n + c] = vals[i * n + c];
    }
    if (acc.normalized && comp.max !== null) {
      for (let i = 0; i < out.length; i++) out[i] = Math.max(out[i] / comp.max, -1.0);
    }
    return { data: out, count, n };
  }

  readView(viewIndex, accOffset, componentType, n, count, out) {
    const view = this.json.bufferViews[viewIndex];
    const buf = this.buffers[view.buffer];
    if (!buf) throw new GLTFError(`${this.name}: buffer ${view.buffer} missing`);
    const comp = COMPONENTS[componentType];
    const start = (view.byteOffset ?? 0) + accOffset;
    const stride = view.byteStride || n * comp.size;
    if (count > 0 && start + stride * (count - 1) + n * comp.size > buf.length) {
      throw new GLTFError(`${this.name}: accessor reads past the end of buffer ${view.buffer}`);
    }
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    for (let i = 0; i < count; i++) {
      const base = start + i * stride;
      for (let c = 0; c < n; c++) out[i * n + c] = readComponent(dv, base + c * comp.size, componentType);
    }
  }

  parents() {
    const parent = new Array(this.nodes.length).fill(null);
    this.nodes.forEach((node, i) => {
      for (const child of node.children ?? []) parent[child] = i;
    });
    return parent;
  }

  /** Local transform from ``matrix`` or TRS; ``trs`` overrides individual channels. */
  static localMatrix(node, trs = null) {
    if (node.matrix && !trs) return colMajorToRowMajor(node.matrix);
    let t = node.translation ?? [0, 0, 0];
    let r = node.rotation ?? [0, 0, 0, 1];
    let s = node.scale ?? [1, 1, 1];
    if (trs) {
      t = trs.translation ?? t;
      r = trs.rotation ?? r;
      s = trs.scale ?? s;
    }
    return trsMatrix(t, r, s);
  }

  /** World matrices of all nodes, with optional per-node TRS overrides (``Map`` of node -> {translation, rotation, scale}). */
  globalMatrices(overrides = null) {
    const parent = this.parents();
    const n = this.nodes.length;
    const local = this.nodes.map((node, i) => GLTFFile.localMatrix(node, overrides?.get?.(i) ?? null));
    const world = new Array(n).fill(null);
    const compute = (i) => {
      if (world[i] !== null) return world[i];
      const p = parent[i];
      world[i] = p === null ? local[i] : matMul(compute(p), local[i]);
      return world[i];
    };
    for (let i = 0; i < n; i++) compute(i);
    return world;
  }
}

// ---------------------------------------------------------------------------
// 4x4 matrix helpers (row-major)
// ---------------------------------------------------------------------------
export function identity4() {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function colMajorToRowMajor(a) {
  const m = new Float64Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) m[r * 4 + c] = a[c * 4 + r];
  return m;
}

export function matMul(a, b) {
  const m = new Float64Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
      m[r * 4 + c] = s;
    }
  }
  return m;
}

export function trsMatrix(t, q, s) {
  const norm = Math.hypot(q[0], q[1], q[2], q[3]) || 1.0;
  const [x, y, z, w] = [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
  const rot = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
  const m = identity4();
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) m[r * 4 + c] = rot[r][c] * s[c];
    m[r * 4 + 3] = t[r];
  }
  return m;
}

/** ``m @ [x, y, z, 1]`` -> ``[x', y', z']``. */
export function transformPoint(m, x, y, z) {
  return [
    m[0] * x + m[1] * y + m[2] * z + m[3],
    m[4] * x + m[5] * y + m[6] * z + m[7],
    m[8] * x + m[9] * y + m[10] * z + m[11],
  ];
}
