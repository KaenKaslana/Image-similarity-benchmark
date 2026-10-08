/**
 * Mesh file loading, standing in for ``trimesh.load(path, force="mesh")``.
 *
 * Supported: glTF / GLB (node transforms of the default scene applied,
 * triangle / strip / fan primitives), OBJ (polygons fan-triangulated) and STL
 * (binary and ASCII). All parts are merged into one triangle soup; only
 * vertices referenced by a face are kept, as trimesh's vertex merge does.
 */

import fs from 'node:fs';
import path from 'node:path';
import { GLTFError, GLTFFile, identity4, matMul, transformPoint } from './gltf.js';

export class MeshLoadError extends Error {}

export const MESH_EXTENSIONS = ['.glb', '.gltf', '.obj', '.stl'];

/** ``{positions: Float64Array(V*3), faces: Uint32Array(F*3), vertexCount}``, built incrementally. */
class Soup {
  constructor() {
    this.pos = [];
    this.faces = [];
  }

  /** Append ``count`` positions; returns the index of the first one. */
  addPositions(arr, count) {
    const base = this.pos.length / 3;
    for (let i = 0; i < count * 3; i++) this.pos.push(arr[i]);
    return base;
  }

  finish(vertexCount) {
    // keep referenced vertices only, renumbered in first-use order
    const used = new Int32Array(this.pos.length / 3).fill(-1);
    const faces = new Uint32Array(this.faces.length);
    const kept = [];
    for (let i = 0; i < this.faces.length; i++) {
      const v = this.faces[i];
      if (used[v] < 0) {
        used[v] = kept.length;
        kept.push(v);
      }
      faces[i] = used[v];
    }
    const positions = new Float64Array(kept.length * 3);
    kept.forEach((v, i) => {
      positions[i * 3] = this.pos[v * 3];
      positions[i * 3 + 1] = this.pos[v * 3 + 1];
      positions[i * 3 + 2] = this.pos[v * 3 + 2];
    });
    return { positions, faces, vertexCount: vertexCount ?? kept.length };
  }
}

// ---------------------------------------------------------------------------
// glTF / GLB
// ---------------------------------------------------------------------------
function primitiveTriangles(mode, indices) {
  const out = [];
  const n = indices.length;
  if (mode === 4) {
    for (let i = 0; i + 2 < n; i += 3) out.push(indices[i], indices[i + 1], indices[i + 2]);
  } else if (mode === 5) {
    for (let i = 0; i + 2 < n; i++) {
      if (i % 2 === 0) out.push(indices[i], indices[i + 1], indices[i + 2]);
      else out.push(indices[i + 1], indices[i], indices[i + 2]);
    }
  } else if (mode === 6) {
    for (let i = 1; i + 1 < n; i++) out.push(indices[0], indices[i], indices[i + 1]);
  }
  return out;
}

function loadGltf(file) {
  let gl;
  try {
    gl = new GLTFFile(file);
  } catch (exc) {
    throw new MeshLoadError(`could not load ${path.basename(file)}: ${exc.message}`);
  }
  const required = gl.json.extensionsRequired ?? [];
  const unsupported = required.filter((e) => ['KHR_draco_mesh_compression', 'EXT_meshopt_compression'].includes(e));
  if (unsupported.length) throw new MeshLoadError(`${path.basename(file)} uses ${unsupported.join(', ')}, which is not supported`);

  const soup = new Soup();
  const nodes = gl.nodes;
  let roots;
  const scenes = gl.json.scenes ?? [];
  if (scenes.length) {
    roots = scenes[gl.json.scene ?? 0]?.nodes ?? [];
  } else {
    const parent = gl.parents();
    roots = nodes.map((_, i) => i).filter((i) => parent[i] === null);
  }
  const visit = (ni, parentWorld) => {
    const node = nodes[ni];
    const world = matMul(parentWorld, GLTFFile.localMatrix(node));
    if (node.mesh !== undefined) {
      const mesh = gl.json.meshes[node.mesh];
      for (const prim of mesh.primitives ?? []) {
        const mode = prim.mode ?? 4;
        if (![4, 5, 6].includes(mode) || prim.attributes?.POSITION === undefined) continue;
        const pos = gl.accessor(prim.attributes.POSITION);
        const local = new Float64Array(pos.count * 3);
        for (let i = 0; i < pos.count; i++) {
          const p = transformPoint(world, pos.data[i * 3], pos.data[i * 3 + 1], pos.data[i * 3 + 2]);
          local[i * 3] = p[0];
          local[i * 3 + 1] = p[1];
          local[i * 3 + 2] = p[2];
        }
        const indices = prim.indices !== undefined ? gl.accessor(prim.indices).data : Float64Array.from({ length: pos.count }, (_, i) => i);
        const base = soup.addPositions(local, pos.count);
        for (const v of primitiveTriangles(mode, indices)) soup.faces.push(base + v);
      }
    }
    for (const child of node.children ?? []) visit(child, world);
  };
  try {
    for (const r of roots) visit(r, identity4());
  } catch (exc) {
    if (exc instanceof GLTFError || exc instanceof TypeError) throw new MeshLoadError(`could not load ${path.basename(file)}: ${exc.message}`);
    throw exc;
  }
  return soup.finish();
}

// ---------------------------------------------------------------------------
// OBJ
// ---------------------------------------------------------------------------
function loadObj(file) {
  const text = fs.readFileSync(file, 'utf-8');
  const v = [];
  let nvt = 0;
  const soup = new Soup();
  const corners = new Set(); // unique (v, vt) pairs, the vertices trimesh keeps
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.startsWith('v ')) {
      const p = line.split(/\s+/);
      v.push(Number(p[1]), Number(p[2]), Number(p[3]));
    } else if (line.startsWith('vt ')) {
      nvt++;
    } else if (line.startsWith('f ')) {
      const items = line.split(/\s+/).slice(1);
      const idx = items.map((it) => {
        const [a, b] = it.split('/');
        let i = parseInt(a, 10);
        i = i < 0 ? v.length / 3 + i : i - 1;
        let t = b ? parseInt(b, 10) : NaN;
        if (!Number.isNaN(t)) t = t < 0 ? nvt + t : t - 1;
        corners.add(Number.isNaN(t) ? `${i}` : `${i}/${t}`);
        return i;
      });
      for (let k = 1; k + 1 < idx.length; k++) soup.faces.push(idx[0], idx[k], idx[k + 1]);
    }
  }
  soup.pos = v;
  return soup.finish(corners.size);
}

// ---------------------------------------------------------------------------
// STL
// ---------------------------------------------------------------------------
function loadStl(file) {
  const buf = fs.readFileSync(file);
  const soup = new Soup();
  const tris = [];
  const isBinary = buf.length >= 84 && 84 + buf.readUInt32LE(80) * 50 === buf.length;
  if (isBinary) {
    const n = buf.readUInt32LE(80);
    for (let i = 0; i < n; i++) {
      const o = 84 + i * 50 + 12;
      for (let k = 0; k < 9; k++) tris.push(buf.readFloatLE(o + k * 4));
    }
  } else {
    const re = /vertex\s+(\S+)\s+(\S+)\s+(\S+)/g;
    let m;
    const text = buf.toString('utf-8');
    while ((m = re.exec(text)) !== null) tris.push(Number(m[1]), Number(m[2]), Number(m[3]));
    tris.length -= tris.length % 9;
  }
  // trimesh merges STL corners that share a position
  const index = new Map();
  for (let i = 0; i < tris.length; i += 3) {
    const key = `${tris[i]},${tris[i + 1]},${tris[i + 2]}`;
    let vi = index.get(key);
    if (vi === undefined) {
      vi = soup.pos.length / 3;
      soup.pos.push(tris[i], tris[i + 1], tris[i + 2]);
      index.set(key, vi);
    }
    soup.faces.push(vi);
  }
  return soup.finish();
}

/** Load a mesh file as ``{positions, faces, vertexCount}`` (raw coordinates, all parts merged). */
export function loadMeshFile(file) {
  if (!fs.existsSync(file) || !fs.statSync(file).isFile()) throw new MeshLoadError(`model file not found: ${file}`);
  const ext = path.extname(file).toLowerCase();
  let out;
  try {
    if (ext === '.glb' || ext === '.gltf') out = loadGltf(file);
    else if (ext === '.obj') out = loadObj(file);
    else if (ext === '.stl') out = loadStl(file);
    else throw new MeshLoadError(`${path.basename(file)}: unsupported format ${ext} (supported: ${MESH_EXTENSIONS.join(', ')})`);
  } catch (exc) {
    if (exc instanceof MeshLoadError) throw exc;
    throw new MeshLoadError(`could not load ${path.basename(file)}: ${exc.message}`);
  }
  if (out.faces.length === 0) throw new MeshLoadError(`${path.basename(file)} contains no triangle geometry (point cloud or empty scene?)`);
  return out;
}
