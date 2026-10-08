/**
 * Rig and animation comparison (port of src/rig.py).
 *
 * Reads skeleton, skinning and animation facts from glTF / GLB files, poses
 * the mesh with linear blend skinning at a few times of the longest clip, and
 * scores the candidate's rig against the reference's. The score is applied as
 * a penalty on the shape score only when the reference is rigged.
 */

import path from 'node:path';
import { complexityScore } from './complexity.js';
import { GLTFFile, matMul } from './gltf.js';
import { applyRotation, bounds } from './render.js';
import { getLogger, mean } from './util.js';

const logger = getLogger('rig');

export const COMPONENT_WEIGHTS = { bones: 0.15, skeleton: 0.25, skinning: 0.15, animation: 0.25, motion: 0.2 };

function newRigInfo(file, format) {
  return {
    path: String(file),
    format,
    readable: true,
    error: null,
    has_skin: false,
    skins: 0,
    joints: 0,
    joint_depth: 0,
    root_joints: 0,
    joint_names: [],
    joint_positions: new Float64Array(0), // world, rest pose
    vertices_total: 0,
    vertices_skinned: 0,
    skinned_fraction: 0.0,
    weight_coverage: 0.0,
    weights_normalised: 0.0,
    max_influences: 0,
    clips: [],
    animated_joint_fraction: 0.0,
    motion_amplitude: null,
    motion_max: null,
    motion_clip: null,
    deformation_ok: null,
    bind_vertices: new Float64Array(0),
    extent: 0.0,
  };
}

export const hasAnimation = (info) => info.clips.some((c) => c.animated_joints > 0);

/** JSON summary of a rig (without the large arrays). */
export function rigInfoToDict(info) {
  return {
    path: info.path,
    format: info.format,
    readable: info.readable,
    error: info.error,
    has_skin: info.has_skin,
    skins: info.skins,
    joints: info.joints,
    joint_depth: info.joint_depth,
    root_joints: info.root_joints,
    vertices_total: info.vertices_total,
    vertices_skinned: info.vertices_skinned,
    skinned_fraction: info.skinned_fraction,
    weight_coverage: info.weight_coverage,
    weights_normalised: info.weights_normalised,
    max_influences: info.max_influences,
    clips: info.clips,
    has_animation: hasAnimation(info),
    animated_joint_fraction: info.animated_joint_fraction,
    motion_amplitude: info.motion_amplitude,
    motion_max: info.motion_max,
    motion_clip: info.motion_clip,
    deformation_ok: info.deformation_ok,
  };
}

// ---------------------------------------------------------------------------
// Animation sampling
// ---------------------------------------------------------------------------
/** Value of one sampler at time ``t`` (clamped to the key range). */
function sampleChannel(times, values, n, interpolation, t, isRotation) {
  let vals = values;
  if (interpolation === 'CUBICSPLINE') {
    // [in-tangent, value, out-tangent] per key -> keep the values
    const keys = values.length / n / 3;
    vals = new Float64Array(keys * n);
    for (let k = 0; k < keys; k++) for (let c = 0; c < n; c++) vals[k * n + c] = values[(3 * k + 1) * n + c];
  }
  const row = (k) => Array.from(vals.subarray(k * n, k * n + n));
  if (times.length === 1 || t <= times[0]) return row(0);
  if (t >= times[times.length - 1]) return row(times.length - 1);
  let k = 0;
  while (k + 1 < times.length && times[k + 1] <= t) k++;
  if (interpolation === 'STEP') return row(k);
  const t0 = times[k];
  const t1 = times[k + 1];
  const a = t1 <= t0 ? 0.0 : (t - t0) / (t1 - t0);
  const v0 = row(k);
  let v1 = row(k + 1);
  if (isRotation) {
    if (v0.reduce((s, v, i) => s + v * v1[i], 0) < 0) v1 = v1.map((v) => -v);
    const out = v0.map((v, i) => (1 - a) * v + a * v1[i]);
    const norm = Math.hypot(...out) || 1.0;
    return out.map((v) => v / norm);
  }
  return v0.map((v, i) => (1 - a) * v + a * v1[i]);
}

function animationChannels(gl, anim) {
  const out = [];
  for (const ch of anim.channels ?? []) {
    const node = ch.target?.node;
    const p = ch.target?.path;
    if (node === undefined || !['translation', 'rotation', 'scale'].includes(p)) continue;
    const sampler = anim.samplers[ch.sampler];
    const times = gl.accessor(sampler.input).data;
    const values = gl.accessor(sampler.output);
    out.push({ node, path: p, times, values: values.data, n: values.n, interp: sampler.interpolation ?? 'LINEAR' });
  }
  return out;
}

function poseOverrides(channels, t) {
  const overrides = new Map();
  for (const ch of channels) {
    if (!overrides.has(ch.node)) overrides.set(ch.node, {});
    overrides.get(ch.node)[ch.path] = sampleChannel(ch.times, ch.values, ch.n, ch.interp, t, ch.path === 'rotation');
  }
  return overrides;
}

// ---------------------------------------------------------------------------
// Skinning
// ---------------------------------------------------------------------------
/** ``[node index, skin index or null, primitive]`` for every primitive with positions. */
function skinnedPrimitives(gl) {
  const out = [];
  gl.nodes.forEach((node, ni) => {
    if (node.mesh === undefined) return;
    const mesh = gl.json.meshes[node.mesh];
    const skin = node.skin ?? null;
    for (const prim of mesh.primitives ?? []) if (prim.attributes?.POSITION !== undefined) out.push([ni, skin, prim]);
  });
  return out;
}

/** World-space vertices of all primitives under node matrices ``world`` (skinned ones via LBS). */
function posedVertices(gl, prims, world, ibms) {
  const chunks = [];
  let total = 0;
  for (const [ni, skinIdx, prim] of prims) {
    const attrs = prim.attributes;
    const pos = gl.accessor(attrs.POSITION);
    const out = new Float64Array(pos.count * 3);
    const nodeM = world[ni];
    const skinned = skinIdx !== null && attrs.JOINTS_0 !== undefined && attrs.WEIGHTS_0 !== undefined;
    let jm = null;
    let ji = null;
    let w = null;
    if (skinned) {
      const joints = gl.json.skins[skinIdx].joints;
      jm = joints.map((j, k) => matMul(world[j], ibms[skinIdx][k]));
      ji = gl.accessor(attrs.JOINTS_0);
      w = gl.accessor(attrs.WEIGHTS_0);
    }
    for (let v = 0; v < pos.count; v++) {
      const x = pos.data[v * 3], y = pos.data[v * 3 + 1], z = pos.data[v * 3 + 2];
      let m = nodeM;
      if (skinned) {
        const nk = w.n;
        let wsum = 0;
        const blended = new Float64Array(16);
        for (let k = 0; k < nk; k++) {
          const wk = w.data[v * nk + k];
          wsum += wk;
          const j = Math.min(Math.max(ji.data[v * ji.n + k], 0), jm.length - 1);
          const M = jm[j];
          for (let e = 0; e < 16; e++) blended[e] += wk * M[e];
        }
        // vertices without any weight would collapse to the origin; keep them in place
        if (wsum > 1e-8) m = blended;
      }
      out[v * 3] = m[0] * x + m[1] * y + m[2] * z + m[3];
      out[v * 3 + 1] = m[4] * x + m[5] * y + m[6] * z + m[7];
      out[v * 3 + 2] = m[8] * x + m[9] * y + m[10] * z + m[11];
    }
    chunks.push(out);
    total += out.length;
  }
  const all = new Float64Array(total);
  let off = 0;
  for (const c of chunks) {
    all.set(c, off);
    off += c.length;
  }
  return all;
}

function clipInfo(gl, anim, jointSet) {
  const nodes = new Set();
  let keyframes = 0;
  let duration = 0.0;
  for (const ch of anim.channels ?? []) {
    const node = ch.target?.node;
    if (node !== undefined) nodes.add(node);
    const sampler = anim.samplers[ch.sampler];
    const acc = gl.json.accessors[sampler.input];
    keyframes += acc.count ?? 0;
    if (acc.max && acc.max.length) duration = Math.max(duration, acc.max[0]);
  }
  if (duration === 0.0 && (anim.channels ?? []).length) {
    try {
      duration = Math.max(...anim.channels.map((ch) => Math.max(...gl.accessor(anim.samplers[ch.sampler].input).data)));
    } catch {
      duration = 0.0;
    }
  }
  return {
    name: String(anim.name ?? ''),
    duration,
    animated_joints: [...nodes].filter((n) => jointSet.has(n)).length,
    animated_nodes: nodes.size,
    channels: (anim.channels ?? []).length,
    keyframes,
  };
}

function fillRigInfo(gl, info, motionSamples) {
  const skins = gl.json.skins ?? [];
  const prims = skinnedPrimitives(gl);
  const world = gl.globalMatrices();
  const ibms = skins.map((skin) => {
    if (skin.inverseBindMatrices !== undefined) {
      const acc = gl.accessor(skin.inverseBindMatrices);
      return Array.from({ length: acc.count }, (_, k) => {
        const m = new Float64Array(16);
        // column-major in glTF -> row-major
        for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) m[r * 4 + c] = acc.data[k * 16 + c * 4 + r];
        return m;
      });
    }
    return skin.joints.map(() => Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
  });

  // geometry (bind pose)
  const bind = posedVertices(gl, prims, world, ibms);
  info.bind_vertices = bind;
  info.vertices_total = bind.length / 3;
  if (bind.length) {
    const { lo, hi } = bounds(bind);
    info.extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  }

  // skeleton
  const jointNodes = [];
  for (const skin of skins) for (const j of skin.joints ?? []) if (!jointNodes.includes(j)) jointNodes.push(j);
  info.skins = skins.length;
  info.joints = jointNodes.length;
  info.has_skin = jointNodes.length > 0;
  if (!info.has_skin) {
    info.clips = (gl.json.animations ?? []).map((a) => clipInfo(gl, a, new Set()));
    return;
  }
  const parent = gl.parents();
  const jointSet = new Set(jointNodes);
  info.root_joints = jointNodes.filter((j) => parent[j] === null || !jointSet.has(parent[j])).length;
  const depth = (j) => {
    let d = 1;
    let p = parent[j];
    while (p !== null && jointSet.has(p)) {
      d += 1;
      p = parent[p];
    }
    return d;
  };
  info.joint_depth = Math.max(...jointNodes.map(depth));
  info.joint_names = jointNodes.map((j) => String(gl.nodes[j].name ?? `node${j}`));
  info.joint_positions = Float64Array.from(jointNodes.flatMap((j) => [world[j][3], world[j][7], world[j][11]]));

  // skinning
  let skinned = 0;
  let covered = 0;
  let normalised = 0;
  let maxInf = 0;
  for (const [, skinIdx, prim] of prims) {
    if (skinIdx === null || prim.attributes.WEIGHTS_0 === undefined) continue;
    const w = gl.accessor(prim.attributes.WEIGHTS_0);
    for (let v = 0; v < w.count; v++) {
      let total = 0;
      let inf = 0;
      for (let k = 0; k < w.n; k++) {
        const x = w.data[v * w.n + k];
        total += x;
        if (x > 1e-6) inf++;
      }
      if (total > 1e-3) covered++;
      if (Math.abs(total - 1.0) < 1e-2) normalised++;
      if (inf > maxInf) maxInf = inf;
    }
    skinned += w.count;
  }
  info.vertices_skinned = skinned;
  info.skinned_fraction = info.vertices_total ? skinned / info.vertices_total : 0.0;
  info.weight_coverage = skinned ? covered / skinned : 0.0;
  info.weights_normalised = skinned ? normalised / skinned : 0.0;
  info.max_influences = maxInf;

  // animation
  const anims = gl.json.animations ?? [];
  info.clips = anims.map((a) => clipInfo(gl, a, jointSet));
  if (info.clips.length) info.animated_joint_fraction = Math.max(...info.clips.map((c) => c.animated_joints)) / info.joints;
  let best = null;
  for (const c of info.clips) {
    if (c.animated_joints <= 0) continue;
    if (best === null || c.animated_joints > best.animated_joints || (c.animated_joints === best.animated_joints && c.duration > best.duration)) best = c;
  }
  if (best === null || info.extent <= 0 || motionSamples <= 0) return;
  const anim = anims[info.clips.findIndex((c) => c.name === best.name)];
  const channels = animationChannels(gl, anim);
  const tEnd = channels.length ? Math.max(...channels.map((ch) => ch.times[ch.times.length - 1])) : 0.0;
  const times = tEnd > 0 ? Array.from({ length: motionSamples }, (_, i) => (tEnd * (i + 1)) / motionSamples) : [0.0];
  const amplitudes = [];
  const maxima = [];
  let ok = true;
  for (const t of times) {
    const posed = posedVertices(gl, prims, gl.globalMatrices(poseOverrides(channels, t)), ibms);
    let sum = 0;
    let mx = -Infinity;
    const nv = posed.length / 3;
    for (let v = 0; v < nv; v++) {
      const d = Math.hypot(posed[v * 3] - bind[v * 3], posed[v * 3 + 1] - bind[v * 3 + 1], posed[v * 3 + 2] - bind[v * 3 + 2]) / info.extent;
      if (!Number.isFinite(d)) {
        ok = false;
        break;
      }
      sum += d;
      if (d > mx) mx = d;
    }
    if (!ok) break;
    amplitudes.push(sum / nv);
    maxima.push(mx);
  }
  info.motion_clip = best.name;
  if (ok) {
    info.motion_amplitude = mean(amplitudes);
    info.motion_max = Math.max(...maxima);
    // a limb swinging moves at most ~2 extents; more than 5 means the skin matrices are broken
    info.deformation_ok = info.motion_max <= 5.0;
  } else {
    info.deformation_ok = false;
  }
}

/**
 * Read skeleton, skinning and animation facts from a glTF / GLB model. Never
 * throws for a model without a rig; unreadable files come back with
 * ``readable: false`` and ``error``.
 */
export function analyseRig(file, motionSamples = 8) {
  const ext = path.extname(String(file)).toLowerCase().replace(/^\./, '');
  const info = newRigInfo(file, ext);
  if (ext !== 'glb' && ext !== 'gltf') return info;
  try {
    fillRigInfo(new GLTFFile(String(file)), info, motionSamples);
  } catch (exc) {
    info.readable = false;
    info.error = exc.message;
    logger.warning(`rig: could not analyse ${path.basename(String(file))}: ${exc.message}`);
  }
  return info;
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------
/** Joint positions in the render frame: rotated like the mesh, centred and scaled by its bounding box. */
export function canonicalJointPositions(info, rotation) {
  if (!info.has_skin || info.bind_vertices.length === 0) return new Float64Array(0);
  const verts = applyRotation(info.bind_vertices, rotation);
  const { lo, hi } = bounds(verts);
  const centre = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
  const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1.0;
  const joints = applyRotation(info.joint_positions, rotation);
  for (let i = 0; i < joints.length; i += 3) for (let k = 0; k < 3; k++) joints[i + k] = (joints[i + k] - centre[k]) / extent;
  return joints;
}

/** Symmetric mean nearest-neighbour distance between two point sets (xyz interleaved). */
export function chamferDistance(a, b) {
  const na = a.length / 3;
  const nb = b.length / 3;
  if (na === 0 || nb === 0) return Infinity;
  const minA = new Float64Array(na).fill(Infinity);
  const minB = new Float64Array(nb).fill(Infinity);
  for (let i = 0; i < na; i++) {
    for (let j = 0; j < nb; j++) {
      const d = Math.hypot(a[i * 3] - b[j * 3], a[i * 3 + 1] - b[j * 3 + 1], a[i * 3 + 2] - b[j * 3 + 2]);
      if (d < minA[i]) minA[i] = d;
      if (d < minB[j]) minB[j] = d;
    }
  }
  return 0.5 * (mean(minA) + mean(minB));
}

function ratioScore(cand, ref, freeLog2, zeroLog2) {
  if (ref <= 0 && cand <= 0) return 100.0;
  if (ref <= 0 || cand <= 0) return 0.0;
  return complexityScore(cand / ref, freeLog2, zeroLog2);
}

const IDENTITY3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/**
 * Score the candidate's rig against the reference's. ``*Rotation`` are the
 * 3x3 matrices that took each model into the render frame.
 */
export function rigComparison(ref, cand, cfg, refRotation = IDENTITY3, candRotation = IDENTITY3) {
  const out = {
    applicable: ref.has_skin,
    weight: cfg.weight,
    score: null,
    components: {},
    component_weights: {},
    skeleton_chamfer: null,
    reference: rigInfoToDict(ref),
    candidate: rigInfoToDict(cand),
  };
  if (!ref.has_skin) return out;

  const components = {};
  if (cand.has_skin) {
    components.bones = ratioScore(cand.joints, ref.joints, cfg.bone_free_log2, cfg.bone_zero_log2);
    const d = chamferDistance(canonicalJointPositions(ref, refRotation), canonicalJointPositions(cand, candRotation));
    out.skeleton_chamfer = Number.isFinite(d) ? d : null;
    components.skeleton = Number.isFinite(d) ? 100.0 * Math.max(0.0, 1.0 - d / cfg.skeleton_max_distance) : 0.0;
    components.skinning = 100.0 * cand.weight_coverage * cand.weights_normalised;
  } else {
    Object.assign(components, { bones: 0.0, skeleton: 0.0, skinning: 0.0 });
  }
  if (hasAnimation(ref)) {
    if (cand.has_skin && hasAnimation(cand) && cand.deformation_ok !== false) {
      components.animation = 100.0 * Math.min(1.0, cand.animated_joint_fraction / Math.max(ref.animated_joint_fraction, 1e-9));
      components.motion = ratioScore(cand.motion_amplitude ?? 0.0, ref.motion_amplitude ?? 0.0, cfg.motion_free_log2, cfg.motion_zero_log2);
    } else {
      components.animation = 0.0;
      components.motion = 0.0;
    }
  }
  const total = Object.keys(components).reduce((s, k) => s + COMPONENT_WEIGHTS[k], 0);
  const weights = Object.fromEntries(Object.keys(components).map((k) => [k, COMPONENT_WEIGHTS[k] / total]));
  out.components = components;
  out.component_weights = weights;
  out.score = Object.keys(components).reduce((s, k) => s + components[k] * weights[k], 0);
  return out;
}

/** One-line description, e.g. ``66 bones, 1 clip(s), best 'Take 001' 3.2s animating 66 bones``. */
export function summarizeRig(info) {
  if (!info.readable) return 'unreadable';
  if (!info.has_skin) return 'no rig' + (info.clips.length ? `, ${info.clips.length} clip(s) on plain nodes` : '');
  const parts = [`${info.joints} bones`];
  if (info.clips.length) {
    const best = info.clips.reduce((a, c) => (c.animated_joints > a.animated_joints ? c : a));
    parts.push(`${info.clips.length} clip(s), best '${best.name}' ${best.duration.toFixed(1)}s animating ${best.animated_joints} bones`);
  } else {
    parts.push('no animation');
  }
  if (info.motion_amplitude !== null) parts.push(`motion ${info.motion_amplitude.toFixed(3)}`);
  if (info.deformation_ok === false) parts.push('DEFORMATION BROKEN');
  return parts.join(', ');
}
