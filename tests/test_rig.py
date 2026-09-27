"""Rig / animation term: glTF parsing, linear blend skinning and scoring."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.config import ConfigError, RigConfig, config_from_dict
from src.rig import (
    GLTFFile,
    RigInfo,
    analyse_rig,
    canonical_joint_positions,
    chamfer_distance,
    rig_comparison,
    summarize_rig,
)


def _write_glb(path: Path, gltf: dict, blob: bytes) -> Path:
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    return path


def rigged_glb(path: Path, animate: bool = True, bones: int = 2, unit_scale: float = 1.0) -> Path:
    """A vertical bar of ``bones`` segments, each vertex bound to the bone it sits on.

    Bone 0 is the root at the origin, every next bone is a child 1 unit up. The
    optional clip rotates the root 90 degrees about X between t=0 and t=1 so the
    whole bar swings from +Y towards +Z (a big, easy to check motion).
    """
    n_bones = bones
    verts, joints, weights = [], [], []
    for b in range(n_bones):
        for y in (b, b + 1):
            for x, z in ((-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)):
                verts.append((x * unit_scale, y * unit_scale, z * unit_scale))
                joints.append((b, 0, 0, 0))
                weights.append((1.0, 0.0, 0.0, 0.0))
    pos = np.asarray(verts, np.float32)
    idx = []
    for b in range(n_bones):
        base = b * 8
        for k in range(4):  # side quads as triangles
            a, c = base + k, base + (k + 1) % 4
            idx += [a, c, c + 4, a, c + 4, a + 4]
    idx = np.asarray(idx, np.uint16)
    ibm = []
    for b in range(n_bones):
        m = np.eye(4, dtype=np.float32)
        m[1, 3] = -b * unit_scale  # inverse of the bone's world translation
        ibm.append(m.T.ravel())  # column-major
    ibm = np.concatenate(ibm).astype(np.float32)
    times = np.asarray([0.0, 1.0], np.float32)
    s = np.sin(np.pi / 4)
    rots = np.asarray([[0, 0, 0, 1], [s, 0, 0, np.cos(np.pi / 4)]], np.float32)  # identity -> 90deg about X

    blob = b""
    views, accessors = [], []

    def add(arr: np.ndarray, kind: str, comp: int, extra: dict | None = None) -> int:
        nonlocal blob
        raw = arr.tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw)})
        blob += raw + b"\0" * (-len(raw) % 4)
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": len(arr), "type": kind}
        if extra:
            acc.update(extra)
        accessors.append(acc)
        return len(accessors) - 1

    a_pos = add(pos, "VEC3", 5126, {"min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
    a_idx = add(idx, "SCALAR", 5123)
    a_joint = add(np.asarray(joints, np.uint8), "VEC4", 5121)
    a_w = add(np.asarray(weights, np.float32), "VEC4", 5126)
    a_ibm = add(ibm.reshape(n_bones, 16), "MAT4", 5126)
    nodes = [{"name": "figure", "mesh": 0, "skin": 0}]
    for b in range(n_bones):
        node = {"name": f"bone{b}", "translation": [0.0, unit_scale if b else 0.0, 0.0]}
        if b + 1 < n_bones:
            node["children"] = [b + 2]
        nodes.append(node)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": a_pos, "JOINTS_0": a_joint, "WEIGHTS_0": a_w}, "indices": a_idx}]}],
        "skins": [{"joints": list(range(1, n_bones + 1)), "inverseBindMatrices": a_ibm}],
        "buffers": [{"byteLength": 0}],
        "bufferViews": views,
        "accessors": accessors,
    }
    if animate:
        a_t = add(times, "SCALAR", 5126, {"min": [0.0], "max": [1.0]})
        a_r = add(rots, "VEC4", 5126)
        gltf["animations"] = [{
            "name": "swing",
            "samplers": [{"input": a_t, "output": a_r, "interpolation": "LINEAR"}],
            "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}],
        }]
    gltf["buffers"][0]["byteLength"] = len(blob)
    return _write_glb(path, gltf, blob)


def test_parser_reads_skeleton_skinning_and_animation(tmp_path: Path) -> None:
    info = analyse_rig(rigged_glb(tmp_path / "bar.glb"))
    assert info.readable and info.has_skin and info.joints == 2 and info.joint_depth == 2 and info.root_joints == 1
    assert info.joint_names == ["bone0", "bone1"]
    np.testing.assert_allclose(info.joint_positions, [[0, 0, 0], [0, 1, 0]], atol=1e-6)
    assert info.vertices_total == 16 and info.vertices_skinned == 16 and info.skinned_fraction == 1.0
    assert info.weight_coverage == 1.0 and info.weights_normalised == 1.0 and info.max_influences == 1
    assert len(info.clips) == 1 and info.clips[0].name == "swing" and info.clips[0].duration == 1.0
    assert info.clips[0].animated_joints == 1 and info.animated_joint_fraction == 0.5
    # bind pose: LBS with rest-pose matrices reproduces the raw vertices
    np.testing.assert_allclose(info.bind_vertices.max(0), [0.1, 2.0, 0.1], atol=1e-6)
    # the root swings the whole bar by 90 degrees: the far end (y=2) travels ~2*sqrt(2) = 2.8 extents
    assert info.motion_clip == "swing" and info.deformation_ok is True
    assert 1.0 < info.motion_max < 1.5  # extent is 2 (bar length), displacement 2.83 / 2
    assert 0.2 < info.motion_amplitude < 1.0
    assert "2 bones" in summarize_rig(info) and "swing" in summarize_rig(info)


def test_parser_handles_static_and_unsupported_files(tmp_path: Path) -> None:
    static = analyse_rig(rigged_glb(tmp_path / "static.glb", animate=False))
    assert static.has_skin and not static.has_animation and static.motion_amplitude is None
    other = analyse_rig(tmp_path / "thing.obj")
    assert not other.has_skin and other.readable and other.format == "obj"
    (tmp_path / "junk.glb").write_bytes(b"glTF" + b"\0" * 4)
    broken = analyse_rig(tmp_path / "junk.glb")
    assert not broken.readable and broken.error and not broken.has_skin
    assert summarize_rig(broken) == "unreadable" and summarize_rig(other) == "no rig"


def test_accessor_strided_and_normalised(tmp_path: Path) -> None:
    gl = GLTFFile(rigged_glb(tmp_path / "bar.glb"))
    assert gl.accessor(0).shape == (16, 3)
    # interleave two VEC3 float accessors in one strided view
    data = np.arange(12, dtype=np.float32).reshape(2, 6)
    blob = data.tobytes()
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteLength": len(blob), "byteStride": 24}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3"},
            {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 2, "type": "VEC3"},
            {"bufferView": 0, "componentType": 5121, "count": 2, "type": "VEC4", "normalized": True},
        ],
    }
    g2 = GLTFFile(_write_glb(tmp_path / "strided.glb", gltf, blob))
    np.testing.assert_allclose(g2.accessor(0), [[0, 1, 2], [6, 7, 8]])
    np.testing.assert_allclose(g2.accessor(1), [[3, 4, 5], [9, 10, 11]])
    assert g2.accessor(2).dtype == np.float64 and g2.accessor(2).max() <= 1.0


def test_canonical_joints_and_chamfer(tmp_path: Path) -> None:
    info = analyse_rig(rigged_glb(tmp_path / "bar.glb"))
    pts = canonical_joint_positions(info, np.eye(3))
    # bar spans y in [0, 2] -> centred at 0, extent 1: joints at y=-0.5 and 0
    np.testing.assert_allclose(pts[:, 1], [-0.5, 0.0], atol=1e-6)
    big = analyse_rig(rigged_glb(tmp_path / "big.glb", unit_scale=10.0))  # same shape, other units
    np.testing.assert_allclose(canonical_joint_positions(big, np.eye(3)), pts, atol=1e-6)
    assert chamfer_distance(pts, pts) == 0.0
    assert chamfer_distance(pts, pts + [0.1, 0, 0]) == pytest.approx(0.1)
    assert chamfer_distance(pts, np.zeros((0, 3))) == float("inf")


def _cfg(**kw) -> RigConfig:
    c = RigConfig(**kw)
    c.validate()
    return c


def test_rig_comparison_scores(tmp_path: Path) -> None:
    ref = analyse_rig(rigged_glb(tmp_path / "ref.glb"))
    same = analyse_rig(rigged_glb(tmp_path / "cand.glb", unit_scale=3.0))
    cfg = _cfg(weight=0.2)
    out = rig_comparison(ref, same, cfg)
    assert out["applicable"] and out["weight"] == 0.2
    assert out["components"] == pytest.approx({"bones": 100, "skeleton": 100, "skinning": 100, "animation": 100, "motion": 100})
    assert out["score"] == pytest.approx(100.0) and sum(out["component_weights"].values()) == pytest.approx(1.0)

    # a static candidate: skeleton fine, animation parts 0
    static = analyse_rig(rigged_glb(tmp_path / "static.glb", animate=False))
    out = rig_comparison(ref, static, cfg)
    assert out["components"]["animation"] == 0 and out["components"]["motion"] == 0
    assert out["score"] == pytest.approx(100 * (0.15 + 0.25 + 0.15))

    # no skeleton at all: everything 0
    none = RigInfo(path="x.obj", format="obj")
    assert rig_comparison(ref, none, cfg)["score"] == 0.0

    # many more bones: bone score drops, placement still fine (extra joints lie on the bar)
    many = analyse_rig(rigged_glb(tmp_path / "many.glb", bones=16))
    out = rig_comparison(ref, many, cfg)
    assert 0 < out["components"]["bones"] < 100 and out["components"]["skeleton"] > 50

    # reference without animation: only the skeleton components count
    out = rig_comparison(static, same, cfg)
    assert set(out["components"]) == {"bones", "skeleton", "skinning"} and out["score"] == pytest.approx(100.0)

    # unrigged reference: not applicable, no score
    out = rig_comparison(none, same, cfg)
    assert not out["applicable"] and out["score"] is None


def test_rig_config_validation() -> None:
    cfg = config_from_dict({"rig": {"weight": 0.3, "motion_samples": 4}})
    assert cfg.rig.weight == 0.3 and cfg.rig.motion_samples == 4
    assert config_from_dict({}).rig.weight == 0.0
    for bad in ({"weight": 2}, {"bone_free_log2": 3, "bone_zero_log2": 2}, {"motion_samples": 0}, {"skeleton_max_distance": 0}):
        with pytest.raises(ConfigError):
            config_from_dict({"rig": bad})
