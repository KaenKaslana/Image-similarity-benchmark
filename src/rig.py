"""Rig and animation comparison for ``compare-models`` / ``reproduce``.

Characters, creatures and machines are usually delivered as *rigged* models:
a skeleton (glTF ``skins``), per-vertex bone weights and one or more animation
clips. The image metrics only see the rest pose, so this module reads the
glTF/GLB structure directly and answers three questions about the candidate
relative to the reference:

1. **Skeleton** - is there one, how many bones, and do the bones sit where the
   reference's bones sit (Chamfer distance between joint positions, both
   models in the same canonical frame as the renders)?
2. **Skinning** - are the vertices bound to the skeleton with sane weights?
3. **Animation** - are there clips, do they move a similar share of the
   skeleton, does the mesh move about as much as the reference's does, and
   does it survive posing (no exploding vertices)? Motion is measured by
   actually posing the mesh with linear blend skinning at a few sampled
   times of the longest clip.

The result is a 0-100 ``rig score`` that ``compare-models`` applies as a
penalty on the shape score, exactly like the face-count term (see
:mod:`src.complexity`), but only when the *reference* is rigged - a static
reference (a mug) never asks the candidate for bones.

Only glTF / GLB carry rigs in this project; other formats report "no rig".
The parser is deliberately small (no sparse accessors, no morph targets), the
formats produced by Blender, Mixamo, Sketchfab and the AI generators are
covered.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .complexity import complexity_score

logger = logging.getLogger(__name__)

__all__ = ["RigError", "RigInfo", "analyse_rig", "canonical_joint_positions", "rig_comparison", "GLTFFile"]


class RigError(RuntimeError):
    """The file could not be read as glTF (missing buffers, corrupt chunks)."""


# ---------------------------------------------------------------------------
# Minimal glTF 2.0 reader
# ---------------------------------------------------------------------------
_COMPONENT_DTYPES = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
_TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class GLTFFile:
    """JSON + binary buffers of a .glb or .gltf file with accessor decoding."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.json: dict[str, Any] = {}
        self._buffers: list[bytes] = []
        self._load()

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        data = self.path.read_bytes()
        bin_chunk: bytes | None = None
        if data[:4] == b"glTF":
            if len(data) < 20:
                raise RigError(f"{self.path.name}: truncated GLB header")
            _, version, length = struct.unpack_from("<4sII", data, 0)
            offset = 12
            while offset + 8 <= min(length, len(data)):
                chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
                chunk = data[offset + 8: offset + 8 + chunk_len]
                if chunk_type == 0x4E4F534A:  # JSON
                    self.json = json.loads(chunk.decode("utf-8"))
                elif chunk_type == 0x004E4942 and bin_chunk is None:  # BIN
                    bin_chunk = chunk
                offset += 8 + chunk_len
            if not self.json:
                raise RigError(f"{self.path.name}: GLB without a JSON chunk")
        else:
            try:
                self.json = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RigError(f"{self.path.name}: not a glTF/GLB file ({exc})") from exc
        for i, buf in enumerate(self.json.get("buffers", [])):
            uri = buf.get("uri")
            if uri is None:
                if bin_chunk is None:
                    raise RigError(f"{self.path.name}: buffer {i} has no uri and the GLB has no BIN chunk")
                self._buffers.append(bin_chunk)
            elif uri.startswith("data:"):
                self._buffers.append(base64.b64decode(uri.split(",", 1)[1]))
            else:
                ext = self.path.parent / uri
                if not ext.is_file():
                    raise RigError(f"{self.path.name}: external buffer {uri} not found")
                self._buffers.append(ext.read_bytes())

    # -- accessors ---------------------------------------------------------
    def accessor(self, index: int) -> np.ndarray:
        """Decode accessor ``index`` to a float64/int64 array of shape (count, n)."""
        acc = self.json["accessors"][index]
        if "sparse" in acc:
            raise RigError(f"{self.path.name}: sparse accessors are not supported")
        n = _TYPE_SIZES[acc["type"]]
        dtype = np.dtype(_COMPONENT_DTYPES[acc["componentType"]])
        count = int(acc["count"])
        if "bufferView" not in acc:
            return np.zeros((count, n), dtype=np.float64)
        view = self.json["bufferViews"][acc["bufferView"]]
        buf = self._buffers[view["buffer"]]
        start = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
        stride = int(view.get("byteStride", 0)) or n * dtype.itemsize
        if stride == n * dtype.itemsize:
            raw = np.frombuffer(buf, dtype=dtype, count=count * n, offset=start).reshape(count, n)
        else:
            rows = np.frombuffer(buf, dtype=np.uint8, count=stride * (count - 1) + n * dtype.itemsize, offset=start)
            idx = (np.arange(count)[:, None] * stride + np.arange(n * dtype.itemsize)[None, :])
            raw = rows[idx].copy().view(dtype).reshape(count, n)
        if dtype.kind == "f":
            return raw.astype(np.float64)
        if acc.get("normalized"):
            info = np.iinfo(dtype)
            return np.maximum(raw.astype(np.float64) / info.max, -1.0)
        return raw.astype(np.int64)

    # -- nodes -------------------------------------------------------------
    @property
    def nodes(self) -> list[dict[str, Any]]:
        return self.json.get("nodes", [])

    def parents(self) -> list[int | None]:
        parent: list[int | None] = [None] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            for child in node.get("children", []):
                parent[child] = i
        return parent

    @staticmethod
    def local_matrix(node: dict[str, Any], trs: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Local transform from ``matrix`` or TRS; ``trs`` overrides individual channels."""
        if "matrix" in node and trs is None:
            return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T  # column-major in glTF
        t = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
        r = np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
        s = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if trs:
            t = trs.get("translation", t)
            r = trs.get("rotation", r)
            s = trs.get("scale", s)
        return trs_matrix(t, r, s)

    def global_matrices(self, overrides: dict[int, dict[str, np.ndarray]] | None = None) -> np.ndarray:
        """(N, 4, 4) world matrices of all nodes, with optional per-node TRS overrides."""
        overrides = overrides or {}
        parent = self.parents()
        n = len(self.nodes)
        local = np.stack([self.local_matrix(node, overrides.get(i)) for i, node in enumerate(self.nodes)]) if n else np.zeros((0, 4, 4))
        world = np.zeros_like(local)
        done = [False] * n

        def compute(i: int) -> np.ndarray:
            if done[i]:
                return world[i]
            p = parent[i]
            world[i] = local[i] if p is None else compute(p) @ local[i]
            done[i] = True
            return world[i]

        for i in range(n):
            compute(i)
        return world


def trs_matrix(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    x, y, z, w = q / (np.linalg.norm(q) or 1.0)
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot * s[None, :]
    m[:3, 3] = t
    return m


# ---------------------------------------------------------------------------
# Rig analysis
# ---------------------------------------------------------------------------
@dataclass
class ClipInfo:
    name: str
    duration: float
    animated_joints: int
    animated_nodes: int
    channels: int
    keyframes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "duration": self.duration, "animated_joints": self.animated_joints,
            "animated_nodes": self.animated_nodes, "channels": self.channels, "keyframes": self.keyframes,
        }


@dataclass
class RigInfo:
    """Everything the score needs about one model's rig."""

    path: str
    format: str  # "gltf", "glb" or the extension for unsupported formats
    readable: bool = True
    error: str | None = None
    has_skin: bool = False
    skins: int = 0
    joints: int = 0
    joint_depth: int = 0
    root_joints: int = 0
    joint_names: list[str] = field(default_factory=list)
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # world, rest pose
    vertices_total: int = 0
    vertices_skinned: int = 0
    skinned_fraction: float = 0.0  # share of all vertices that belong to a skinned primitive
    weight_coverage: float = 0.0  # share of skinned vertices with a non-zero total weight
    weights_normalised: float = 0.0  # share of skinned vertices whose weights sum to ~1
    max_influences: int = 0
    clips: list[ClipInfo] = field(default_factory=list)
    animated_joint_fraction: float = 0.0  # best clip: animated joints / joints
    motion_amplitude: float | None = None  # mean vertex displacement / extent over the sampled poses
    motion_max: float | None = None  # largest single-vertex displacement / extent
    motion_clip: str | None = None
    deformation_ok: bool | None = None  # False when posing explodes the mesh or produces NaNs
    bind_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), repr=False)  # world, all primitives
    extent: float = 0.0

    @property
    def has_animation(self) -> bool:
        return any(c.animated_joints > 0 for c in self.clips)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "readable": self.readable,
            "error": self.error,
            "has_skin": self.has_skin,
            "skins": self.skins,
            "joints": self.joints,
            "joint_depth": self.joint_depth,
            "root_joints": self.root_joints,
            "vertices_total": self.vertices_total,
            "vertices_skinned": self.vertices_skinned,
            "skinned_fraction": self.skinned_fraction,
            "weight_coverage": self.weight_coverage,
            "weights_normalised": self.weights_normalised,
            "max_influences": self.max_influences,
            "clips": [c.to_dict() for c in self.clips],
            "has_animation": self.has_animation,
            "animated_joint_fraction": self.animated_joint_fraction,
            "motion_amplitude": self.motion_amplitude,
            "motion_max": self.motion_max,
            "motion_clip": self.motion_clip,
            "deformation_ok": self.deformation_ok,
        }


def _sample_channel(times: np.ndarray, values: np.ndarray, interpolation: str, t: float, is_rotation: bool) -> np.ndarray:
    """Value of one animation sampler at time ``t`` (clamped to the key range)."""
    if interpolation == "CUBICSPLINE":
        values = values[1::3]  # [in-tangent, value, out-tangent] per key -> keep the values
    if len(times) == 1 or t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    k = int(np.searchsorted(times, t, side="right") - 1)
    if interpolation == "STEP":
        return values[k]
    t0, t1 = times[k], times[k + 1]
    a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    v0, v1 = values[k], values[k + 1]
    if is_rotation:
        if float(v0 @ v1) < 0:
            v1 = -v1
        out = (1 - a) * v0 + a * v1
        return out / (np.linalg.norm(out) or 1.0)
    return (1 - a) * v0 + a * v1


def _animation_channels(gl: GLTFFile, anim: dict[str, Any]) -> list[tuple[int, str, np.ndarray, np.ndarray, str]]:
    out = []
    for ch in anim.get("channels", []):
        target = ch.get("target", {})
        node = target.get("node")
        path = target.get("path")
        if node is None or path not in ("translation", "rotation", "scale"):
            continue
        sampler = anim["samplers"][ch["sampler"]]
        times = gl.accessor(sampler["input"])[:, 0]
        values = gl.accessor(sampler["output"])
        out.append((int(node), path, times, values, sampler.get("interpolation", "LINEAR")))
    return out


def _pose_overrides(channels, t: float) -> dict[int, dict[str, np.ndarray]]:
    overrides: dict[int, dict[str, np.ndarray]] = {}
    for node, path, times, values, interp in channels:
        overrides.setdefault(node, {})[path] = _sample_channel(times, values, interp, t, path == "rotation")
    return overrides


def _skinned_primitives(gl: GLTFFile) -> list[tuple[int, int | None, dict[str, Any]]]:
    """(node index, skin index or None, primitive) for every primitive in the scene."""
    out = []
    for ni, node in enumerate(gl.nodes):
        if "mesh" not in node:
            continue
        mesh = gl.json["meshes"][node["mesh"]]
        skin = node.get("skin")
        for prim in mesh.get("primitives", []):
            if "POSITION" in prim.get("attributes", {}):
                out.append((ni, skin, prim))
    return out


def _posed_vertices(gl: GLTFFile, prims, world: np.ndarray, ibms: dict[int, np.ndarray]) -> np.ndarray:
    """World-space vertices of all primitives under node matrices ``world`` (skinned ones via LBS)."""
    chunks = []
    for ni, skin_idx, prim in prims:
        attrs = prim["attributes"]
        pos = gl.accessor(attrs["POSITION"])
        hom = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
        if skin_idx is not None and "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
            skin = gl.json["skins"][skin_idx]
            joints = np.asarray(skin["joints"], dtype=np.int64)
            jm = world[joints] @ ibms[skin_idx]  # (J, 4, 4)
            ji = gl.accessor(attrs["JOINTS_0"]).astype(np.int64)
            w = gl.accessor(attrs["WEIGHTS_0"])
            ji = np.clip(ji, 0, len(joints) - 1)
            m = np.einsum("vk,vkij->vij", w, jm[ji])  # (V, 4, 4) blended skin matrix
            posed = np.einsum("vij,vj->vi", m, hom)[:, :3]
            # vertices without any weight would collapse to the origin; keep them in place
            zero = w.sum(1) <= 1e-8
            posed[zero] = (hom[zero] @ world[ni].T)[:, :3]
        else:
            posed = (hom @ world[ni].T)[:, :3]
        chunks.append(posed)
    return np.concatenate(chunks) if chunks else np.zeros((0, 3))


def analyse_rig(path: str | Path, motion_samples: int = 8) -> RigInfo:
    """Read skeleton, skinning and animation facts from a glTF/GLB model.

    Never raises for a model without a rig: unsupported formats and unreadable
    files come back with ``has_skin=False`` (and ``readable=False`` / ``error``
    for the latter), so a static candidate simply scores 0 on the rig.
    """
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    info = RigInfo(path=str(path), format=ext)
    if ext not in ("glb", "gltf"):
        return info
    try:
        gl = GLTFFile(path)
        _fill_rig_info(gl, info, motion_samples)
    except (RigError, KeyError, IndexError, ValueError) as exc:
        info.readable = False
        info.error = str(exc)
        logger.warning("rig: could not analyse %s: %s", path.name, exc)
    return info


def _fill_rig_info(gl: GLTFFile, info: RigInfo, motion_samples: int) -> None:
    skins = gl.json.get("skins", [])
    prims = _skinned_primitives(gl)
    world = gl.global_matrices()
    ibms = {}
    for si, skin in enumerate(skins):
        if "inverseBindMatrices" in skin:
            ibms[si] = gl.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        else:
            ibms[si] = np.tile(np.eye(4), (len(skin["joints"]), 1, 1))

    # geometry (bind pose) -------------------------------------------------
    bind = _posed_vertices(gl, prims, world, ibms)
    info.bind_vertices = bind
    info.vertices_total = int(len(bind))
    if len(bind):
        lo, hi = bind.min(0), bind.max(0)
        info.extent = float((hi - lo).max())

    # skeleton -------------------------------------------------------------
    joint_nodes: list[int] = []
    for skin in skins:
        for j in skin.get("joints", []):
            if j not in joint_nodes:
                joint_nodes.append(int(j))
    info.skins = len(skins)
    info.joints = len(joint_nodes)
    info.has_skin = bool(joint_nodes)
    if not info.has_skin:
        info.clips = [_clip_info(gl, a, set()) for a in gl.json.get("animations", [])]
        return
    parent = gl.parents()
    joint_set = set(joint_nodes)
    info.root_joints = sum(1 for j in joint_nodes if parent[j] is None or parent[j] not in joint_set)

    def depth(j: int) -> int:
        d, p = 1, parent[j]
        while p is not None and p in joint_set:
            d, p = d + 1, parent[p]
        return d

    info.joint_depth = max(depth(j) for j in joint_nodes)
    info.joint_names = [str(gl.nodes[j].get("name", f"node{j}")) for j in joint_nodes]
    info.joint_positions = world[joint_nodes][:, :3, 3].copy()

    # skinning -------------------------------------------------------------
    skinned = 0
    covered = 0
    normalised = 0
    max_inf = 0
    for _, skin_idx, prim in prims:
        attrs = prim["attributes"]
        if skin_idx is None or "WEIGHTS_0" not in attrs:
            continue
        w = gl.accessor(attrs["WEIGHTS_0"])
        total = w.sum(1)
        skinned += len(w)
        covered += int((total > 1e-3).sum())
        normalised += int((np.abs(total - 1.0) < 1e-2).sum())
        max_inf = max(max_inf, int((w > 1e-6).sum(1).max()) if len(w) else 0)
    info.vertices_skinned = skinned
    info.skinned_fraction = skinned / info.vertices_total if info.vertices_total else 0.0
    info.weight_coverage = covered / skinned if skinned else 0.0
    info.weights_normalised = normalised / skinned if skinned else 0.0
    info.max_influences = max_inf

    # animation ------------------------------------------------------------
    anims = gl.json.get("animations", [])
    info.clips = [_clip_info(gl, a, joint_set) for a in anims]
    if info.clips:
        info.animated_joint_fraction = max(c.animated_joints for c in info.clips) / info.joints
    best = max((c for c in info.clips if c.animated_joints > 0), key=lambda c: (c.animated_joints, c.duration), default=None)
    if best is None or info.extent <= 0 or motion_samples <= 0:
        return
    anim = anims[[c.name for c in info.clips].index(best.name)]
    channels = _animation_channels(gl, anim)
    t_end = max((float(ch[2][-1]) for ch in channels), default=0.0)
    times = np.linspace(0.0, t_end, motion_samples + 1)[1:] if t_end > 0 else np.zeros(1)
    amplitudes, maxima, ok = [], [], True
    for t in times:
        posed = _posed_vertices(gl, prims, gl.global_matrices(_pose_overrides(channels, float(t))), ibms)
        disp = np.linalg.norm(posed - bind, axis=1) / info.extent
        if not np.all(np.isfinite(disp)):
            ok = False
            break
        amplitudes.append(float(disp.mean()))
        maxima.append(float(disp.max()))
    info.motion_clip = best.name
    if ok:
        info.motion_amplitude = float(np.mean(amplitudes))
        info.motion_max = float(np.max(maxima))
        # a limb swinging moves at most ~2 extents; more than 5 means the skin matrices are broken
        info.deformation_ok = info.motion_max <= 5.0
    else:
        info.deformation_ok = False


def _clip_info(gl: GLTFFile, anim: dict[str, Any], joint_set: set[int]) -> ClipInfo:
    nodes = set()
    keyframes = 0
    duration = 0.0
    for ch in anim.get("channels", []):
        node = ch.get("target", {}).get("node")
        if node is not None:
            nodes.add(int(node))
        sampler = anim["samplers"][ch["sampler"]]
        acc = gl.json["accessors"][sampler["input"]]
        keyframes += int(acc.get("count", 0))
        if acc.get("max"):
            duration = max(duration, float(acc["max"][0]))
    if duration == 0.0 and anim.get("channels"):
        try:
            duration = max(float(gl.accessor(anim["samplers"][ch["sampler"]]["input"]).max()) for ch in anim["channels"])
        except (RigError, ValueError):
            duration = 0.0
    return ClipInfo(
        name=str(anim.get("name", "")), duration=duration, animated_joints=len(nodes & joint_set),
        animated_nodes=len(nodes), channels=len(anim.get("channels", [])), keyframes=keyframes,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def canonical_joint_positions(info: RigInfo, rotation: np.ndarray) -> np.ndarray:
    """Joint positions in the render frame: rotate like the mesh, then centre
    and scale by the rotated mesh's bounding box (max extent 1)."""
    if not info.has_skin or len(info.bind_vertices) == 0:
        return np.zeros((0, 3))
    verts = info.bind_vertices @ rotation.T
    lo, hi = verts.min(0), verts.max(0)
    centre, extent = (lo + hi) / 2.0, float((hi - lo).max()) or 1.0
    return ((info.joint_positions @ rotation.T) - centre) / extent


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(0.5 * (d.min(1).mean() + d.min(0).mean()))


def _ratio_score(cand: float, ref: float, free_log2: float, zero_log2: float) -> float:
    if ref <= 0 and cand <= 0:
        return 100.0
    if ref <= 0 or cand <= 0:
        return 0.0
    return complexity_score(cand / ref, free_log2, zero_log2)


COMPONENT_WEIGHTS = {"bones": 0.15, "skeleton": 0.25, "skinning": 0.15, "animation": 0.25, "motion": 0.20}


def rig_comparison(
    ref: RigInfo,
    cand: RigInfo,
    cfg: Any,
    ref_rotation: np.ndarray | None = None,
    cand_rotation: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score the candidate's rig against the reference's.

    ``cfg`` is a :class:`src.config.RigConfig`. ``*_rotation`` are the 3x3
    matrices that took each model into the render frame (up/front/yaw), so
    that the skeletons are compared in the same orientation as the images.
    Returns a JSON-friendly dict with ``applicable`` (reference is rigged),
    ``score``, ``weight``, per-component scores and both rig summaries.
    """
    ref_rotation = np.eye(3) if ref_rotation is None else ref_rotation
    cand_rotation = np.eye(3) if cand_rotation is None else cand_rotation
    out: dict[str, Any] = {
        "applicable": bool(ref.has_skin),
        "weight": float(cfg.weight),
        "score": None,
        "components": {},
        "component_weights": {},
        "skeleton_chamfer": None,
        "reference": ref.to_dict(),
        "candidate": cand.to_dict(),
    }
    if not ref.has_skin:
        return out

    components: dict[str, float] = {}
    if cand.has_skin:
        components["bones"] = _ratio_score(cand.joints, ref.joints, cfg.bone_free_log2, cfg.bone_zero_log2)
        d = chamfer_distance(canonical_joint_positions(ref, ref_rotation), canonical_joint_positions(cand, cand_rotation))
        out["skeleton_chamfer"] = d if math.isfinite(d) else None
        components["skeleton"] = 100.0 * max(0.0, 1.0 - d / cfg.skeleton_max_distance) if math.isfinite(d) else 0.0
        components["skinning"] = 100.0 * cand.weight_coverage * cand.weights_normalised
    else:
        components.update({"bones": 0.0, "skeleton": 0.0, "skinning": 0.0})

    if ref.has_animation:
        if cand.has_skin and cand.has_animation and cand.deformation_ok is not False:
            components["animation"] = 100.0 * min(1.0, cand.animated_joint_fraction / max(ref.animated_joint_fraction, 1e-9))
            ref_amp = ref.motion_amplitude or 0.0
            cand_amp = cand.motion_amplitude or 0.0
            components["motion"] = _ratio_score(cand_amp, ref_amp, cfg.motion_free_log2, cfg.motion_zero_log2)
        else:
            components["animation"] = 0.0
            components["motion"] = 0.0

    weights = {k: COMPONENT_WEIGHTS[k] for k in components}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    out["components"] = components
    out["component_weights"] = weights
    out["score"] = float(sum(components[k] * weights[k] for k in components))
    return out


def summarize_rig(info: RigInfo) -> str:
    """One-line human description, e.g. ``66 bones, 1 clip (Take 001, 3.2s)``."""
    if not info.readable:
        return "unreadable"
    if not info.has_skin:
        return "no rig" + (f", {len(info.clips)} clip(s) on plain nodes" if info.clips else "")
    parts = [f"{info.joints} bones"]
    if info.clips:
        best = max(info.clips, key=lambda c: c.animated_joints)
        parts.append(f"{len(info.clips)} clip(s), best '{best.name}' {best.duration:.1f}s animating {best.animated_joints} bones")
    else:
        parts.append("no animation")
    if info.motion_amplitude is not None:
        parts.append(f"motion {info.motion_amplitude:.3f}")
    if info.deformation_ok is False:
        parts.append("DEFORMATION BROKEN")
    return ", ".join(parts)
