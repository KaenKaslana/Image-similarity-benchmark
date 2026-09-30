"""Tests for automatic candidate orientation (src/orient.py) and the iso view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.orient import apply_orientation, auto_orient, silhouette_masks  # noqa: E402
from src.render import RenderOptions, all_orientations, load_mesh, render_view, reorient  # noqa: E402


def _asymmetric_mesh() -> trimesh.Trimesh:
    """Tall box + knob on +X/+Y/+Z corner + a bar along -X: no symmetry left."""
    box = trimesh.creation.box(extents=[1.0, 2.0, 1.0])
    knob = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
    knob.apply_translation([0.7, 1.2, 0.7])
    bar = trimesh.creation.box(extents=[0.8, 0.2, 0.2])
    bar.apply_translation([-0.9, -0.6, 0.0])
    return trimesh.util.concatenate([box, knob, bar])


@pytest.fixture
def reference_file(tmp_path: Path) -> Path:
    path = tmp_path / "ref.glb"
    _asymmetric_mesh().export(path)
    return path


def test_all_orientations_are_distinct_rotations() -> None:
    combos = all_orientations()
    assert len(combos) == 24
    assert len(set(combos)) == 24


def test_reorient_identity_keeps_mesh(reference_file: Path) -> None:
    m = load_mesh(reference_file)
    same = reorient(m, "+y", "+z")
    assert np.allclose(same.vertices, m.vertices)


@pytest.mark.parametrize("axis,angle", [([1, 0, 0], 90), ([0, 0, 1], 90), ([0, 1, 0], 180), ([1, 0, 0], -90)])
def test_auto_orient_recovers_rotated_copy(reference_file: Path, tmp_path: Path, axis, angle) -> None:
    rotated = _asymmetric_mesh()
    rotated.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(angle), axis))
    cand_path = tmp_path / "cand.glb"
    rotated.export(cand_path)

    ref = load_mesh(reference_file)
    views = ("front", "side", "top")
    naive = silhouette_masks(load_mesh(cand_path), views, 96)
    ref_masks = silhouette_masks(ref, views, 96)
    naive_iou = np.mean([(a & b).sum() / (a | b).sum() for a, b in zip(ref_masks.values(), naive.values())])

    best = auto_orient(cand_path, ref, views, size=96)
    assert best.mean_iou > 0.97
    assert best.mean_iou > naive_iou
    fixed = load_mesh(cand_path, best.up, best.front)
    fixed_masks = silhouette_masks(fixed, views, 96)
    for v in views:
        a, b = ref_masks[v], fixed_masks[v]
        assert (a & b).sum() / (a | b).sum() > 0.97
    assert best.ranking[0]["mean_iou"] >= best.ranking[1]["mean_iou"]


def test_auto_orient_identity_wins_for_identical_model(reference_file: Path) -> None:
    ref = load_mesh(reference_file)
    best = auto_orient(reference_file, ref, ("front", "side", "top"), size=64)
    assert (best.up, best.front) == ("+y", "+z")
    assert best.mean_iou == pytest.approx(1.0)


def test_iso_view_renders_and_differs_from_front(reference_file: Path) -> None:
    m = load_mesh(reference_file)
    opts = RenderOptions(size=96, supersample=1)
    iso = render_view(m, "iso", opts)
    front = render_view(m, "front", opts)
    assert (iso[..., 3] > 0).any()
    assert not np.array_equal(iso[..., 3] > 0, front[..., 3] > 0)


@pytest.mark.parametrize("angle", [37.0, -120.0])
def test_auto_orient_recovers_arbitrary_yaw(reference_file: Path, tmp_path: Path, angle: float) -> None:
    rotated = _asymmetric_mesh()
    rotated.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(angle), [0, 1, 0]))
    cand_path = tmp_path / "yawed.glb"
    rotated.export(cand_path)

    ref = load_mesh(reference_file)
    views = ("front", "side", "top")
    best = auto_orient(cand_path, ref, views, size=96)
    assert best.mean_iou > 0.95
    assert best.mean_iou > best.axis_aligned_iou
    fixed = apply_orientation(load_mesh(cand_path), best.up, best.front, best.yaw)
    ref_masks, fixed_masks = silhouette_masks(ref, views, 96), silhouette_masks(fixed, views, 96)
    for v in views:
        a, b = ref_masks[v], fixed_masks[v]
        assert (a & b).sum() / (a | b).sum() > 0.95


def test_auto_orient_yaw_can_be_disabled(reference_file: Path) -> None:
    ref = load_mesh(reference_file)
    best = auto_orient(reference_file, ref, ("front",), size=48, yaw_step=0)
    assert best.yaw == 0.0 and best.mean_iou == pytest.approx(1.0)


def _mug(handle_radius: float) -> trimesh.Trimesh:
    body = trimesh.creation.annulus(r_min=0.42, r_max=0.5, height=1.0, sections=48)
    body.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    handle = trimesh.creation.torus(major_radius=handle_radius, minor_radius=0.055, major_sections=32, minor_sections=12)
    handle.apply_translation([0.5 + handle_radius - 0.15, 0.0, 0.0])
    return trimesh.util.concatenate([body, handle])


def test_auto_orient_does_not_trade_size_for_rotation(tmp_path: Path) -> None:
    """A slightly larger handle makes the whole candidate look smaller (models are
    normalised by their largest extent). Turning the handle diagonally used to win
    by shrinking the bounding box; with every view zoomed to the object the
    correct, unrotated orientation wins."""
    ref_path, cand_path = tmp_path / "ref.glb", tmp_path / "cand.glb"
    _mug(0.27).export(ref_path)
    _mug(0.36).export(cand_path)
    best = auto_orient(cand_path, load_mesh(ref_path), ("front", "back", "side", "left", "top", "bottom"), size=96)
    assert (best.up, best.front) == ("+y", "+z")
    assert abs(best.yaw) <= 4.0


def test_decimation_keeps_silhouettes(tmp_path: Path) -> None:
    from src.metrics import compute_silhouette_iou
    from src.orient import decimate_for_silhouettes

    dense = trimesh.creation.icosphere(subdivisions=7)  # 327,680 faces, like an AI-generated mesh
    bump = trimesh.creation.box(extents=[0.6, 0.3, 0.3])
    bump.apply_translation([1.1, 0.0, 0.0])
    path = tmp_path / "dense.glb"
    trimesh.util.concatenate([dense, bump]).export(path)
    mesh = load_mesh(path)
    small = decimate_for_silhouettes(mesh, 2 * 64)  # the grid auto_orient uses for 64 px silhouettes
    assert len(small.faces) < len(mesh.faces) / 3
    views = ("front", "side", "top")
    full, reduced = silhouette_masks(mesh, views, 64), silhouette_masks(small, views, 64)
    for v in views:
        assert compute_silhouette_iou(full[v], reduced[v]) > 0.97
