"""Tests for automatic candidate orientation (src/orient.py) and the iso view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.orient import auto_orient, silhouette_masks  # noqa: E402
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
