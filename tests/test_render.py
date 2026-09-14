"""Tests for the numpy orthographic renderer (src/render.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

trimesh = pytest.importorskip("trimesh")

from src.render import (  # noqa: E402
    RenderError,
    RenderOptions,
    canonical_rotation,
    load_mesh,
    parse_views,
    rasterize,
    render_view,
    render_views,
)


# ---------------------------------------------------------------------------
# rasteriser
# ---------------------------------------------------------------------------
def test_rasterize_square_covers_expected_pixels() -> None:
    # Two triangles forming the square [10, 20) x [10, 20) -> exactly 100 pixels.
    tri = np.array(
        [
            [[10, 10], [20, 10], [20, 20]],
            [[10, 10], [20, 20], [10, 20]],
        ],
        dtype=float,
    )
    depth = np.zeros((2, 3))
    value = np.array([1.0, 1.0])
    z, v = rasterize(tri, depth, value, 32)
    covered = np.isfinite(z)
    assert covered.sum() == 100
    assert covered[10:20, 10:20].all()
    assert not covered[:10].any() and not covered[20:].any()
    assert (v[covered] == 1.0).all()


def test_rasterize_zbuffer_keeps_nearest() -> None:
    far = np.array([[[0, 0], [16, 0], [16, 16]], [[0, 0], [16, 16], [0, 16]]], dtype=float)
    near = np.array([[[4, 4], [12, 4], [12, 12]], [[4, 4], [12, 12], [4, 12]]], dtype=float)
    tri = np.concatenate([far, near])
    depth = np.concatenate([np.full((2, 3), 5.0), np.full((2, 3), 1.0)])
    value = np.array([1.0, 1.0, 2.0, 2.0])
    z, v = rasterize(tri, depth, value, 16)
    assert v[8, 8] == 2.0 and z[8, 8] == 1.0
    assert v[1, 1] == 1.0 and z[1, 1] == 5.0
    # order independent: draw near first, far second
    z2, v2 = rasterize(tri[::-1], depth[::-1], value[::-1], 16)
    assert np.array_equal(v, v2)


def test_rasterize_winding_and_offscreen() -> None:
    cw = np.array([[[10, 10], [20, 20], [20, 10]]], dtype=float)  # clockwise
    ccw = cw[:, ::-1, :]
    z1, _ = rasterize(cw, np.zeros((1, 3)), np.ones(1), 32)
    z2, _ = rasterize(ccw, np.zeros((1, 3)), np.ones(1), 32)
    assert np.array_equal(np.isfinite(z1), np.isfinite(z2))
    assert np.isfinite(z1).sum() > 0
    off = np.array([[[-50, -50], [-40, -50], [-40, -40]]], dtype=float)
    z3, _ = rasterize(off, np.zeros((1, 3)), np.ones(1), 32)
    assert not np.isfinite(z3).any()
    # large triangle covering the whole canvas hits the biggest bucket
    big = np.array([[[-100, -100], [300, -100], [-100, 300]]], dtype=float)
    z4, _ = rasterize(big, np.zeros((1, 3)), np.ones(1), 32)
    assert np.isfinite(z4).all()


# ---------------------------------------------------------------------------
# loading / orientation
# ---------------------------------------------------------------------------
def _handle_mesh() -> trimesh.Trimesh:
    """Tall box with a small cube sticking out of its +X / +Y / +Z corner."""
    box = trimesh.creation.box(extents=[1.0, 2.0, 1.0])
    knob = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
    knob.apply_translation([0.7, 1.2, 0.7])  # protrudes beyond the box on all three axes
    return trimesh.util.concatenate([box, knob])


@pytest.fixture
def mesh_file(tmp_path: Path) -> Path:
    path = tmp_path / "handle.glb"
    _handle_mesh().export(path)
    return path


def test_canonical_rotation_is_right_handed() -> None:
    assert np.allclose(canonical_rotation("+y", "+z"), np.eye(3))
    r = canonical_rotation("+z", "-y")  # Blender convention
    assert np.allclose(np.linalg.det(r), 1.0)
    assert np.allclose(r @ np.array([0, 0, 1.0]), [0, 1, 0])
    assert np.allclose(r @ np.array([0, -1.0, 0]), [0, 0, 1])


def test_load_mesh_normalises(mesh_file: Path) -> None:
    m = load_mesh(mesh_file)
    lo, hi = m.vertices.min(0), m.vertices.max(0)
    assert np.allclose((lo + hi) / 2, 0.0, atol=1e-9)
    assert np.isclose((hi - lo).max(), 1.0)
    assert m.meta["faces"] == len(m.faces) > 0


def test_load_mesh_errors(tmp_path: Path) -> None:
    with pytest.raises(RenderError):
        load_mesh(tmp_path / "missing.obj")
    bad = tmp_path / "bad.obj"
    bad.write_text("this is not a mesh\n")
    with pytest.raises(RenderError):
        load_mesh(bad)


def _rows_of_extreme_column(rgba: np.ndarray, side: str) -> np.ndarray:
    """Row indices of foreground pixels in the left-/right-most foreground column."""
    mask = rgba[..., 3] > 0
    cols = np.nonzero(mask.any(axis=0))[0]
    col = cols.min() if side == "left" else cols.max()
    return np.nonzero(mask[:, col])[0]


def test_view_orientation_follows_third_angle_projection(mesh_file: Path) -> None:
    m = load_mesh(mesh_file)
    opts = RenderOptions(size=128, supersample=1)
    h = 128
    # The knob protrudes at +X, +Y, +Z, so it alone forms the extreme column
    # on the side where +X (or +Z) points, and its rows tell where +Y (or +Z)
    # points.
    #   front: knob on the right, top of the image
    #   side (camera at +X): +Z is on the LEFT, knob at the top
    #   top: +X right, +Z at the bottom (front of the object at the bottom)
    #   back: mirror of front, knob on the left
    front = _rows_of_extreme_column(render_view(m, "front", opts), "right")
    assert (front < h / 2).all()
    side = _rows_of_extreme_column(render_view(m, "side", opts), "left")
    assert (side < h / 2).all()
    top = _rows_of_extreme_column(render_view(m, "top", opts), "right")
    assert (top > h / 2).all()
    back = _rows_of_extreme_column(render_view(m, "back", opts), "left")
    assert (back < h / 2).all()
    # sanity: the extreme column on the other side is the tall box, spanning both halves
    other = _rows_of_extreme_column(render_view(m, "front", opts), "left")
    assert (other < h / 2).any() and (other > h / 2).any()


def test_up_axis_option_changes_which_view_is_tall(mesh_file: Path) -> None:
    tall_y = load_mesh(mesh_file, up="+y", front="+z")
    tall_z = load_mesh(mesh_file, up="+z", front="-y")  # now the long axis is treated as depth
    opts = RenderOptions(size=128, supersample=1)
    a = render_view(tall_y, "front", opts)[..., 3] > 0
    b = render_view(tall_z, "front", opts)[..., 3] > 0
    ha = np.ptp(np.nonzero(a)[0])
    hb = np.ptp(np.nonzero(b)[0])
    assert ha > hb  # the object is no longer tall in the front view


def test_render_views_writes_pngs_and_metadata(mesh_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "renders"
    written = render_views(mesh_file, out, RenderOptions(size=96, views=("front", "top"), supersample=2))
    assert set(written) == {"front", "top"}
    for p in written.values():
        img = Image.open(p)
        assert img.mode == "RGBA" and img.size == (96, 96)
        arr = np.array(img)
        assert (arr[..., 3] == 255).any() and (arr[..., 3] == 0).any()
    meta = json.loads((out / "views.json").read_text())
    assert meta["render"]["views"] == ["front", "top"]
    assert meta["mesh"]["faces"] > 0


def test_supersampling_antialiases_curved_edges(tmp_path: Path) -> None:
    path = tmp_path / "sphere.stl"
    trimesh.creation.icosphere(subdivisions=3).export(path)
    written = render_views(path, tmp_path / "r", RenderOptions(size=64, views=("front",), supersample=2))
    alpha = np.array(Image.open(written["front"]))[..., 3]
    assert ((alpha > 0) & (alpha < 255)).any()


def test_silhouette_style_is_flat_black(mesh_file: Path) -> None:
    m = load_mesh(mesh_file)
    rgba = render_view(m, "front", RenderOptions(size=64, style="silhouette", supersample=1))
    fg = rgba[..., 3] > 0
    assert (rgba[..., :3][fg] == 0).all()


def test_identical_models_render_identically(mesh_file: Path, tmp_path: Path) -> None:
    a = render_views(mesh_file, tmp_path / "a", RenderOptions(size=64))
    b = render_views(mesh_file, tmp_path / "b", RenderOptions(size=64))
    for view in a:
        assert np.array_equal(np.array(Image.open(a[view])), np.array(Image.open(b[view])))


def test_parse_views_and_option_validation() -> None:
    assert parse_views("front, TOP,front") == ("front", "top")
    assert parse_views("all") == ("front", "back", "side", "left", "top", "bottom")
    with pytest.raises(RenderError):
        parse_views("front,diagonal")
    with pytest.raises(RenderError):
        RenderOptions(up="+y", front="+y").validate()
    with pytest.raises(RenderError):
        RenderOptions(style="wireframe").validate()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_render_views_and_compare_models(mesh_file: Path, tmp_path: Path, capsys) -> None:
    from src.cli import main

    out = tmp_path / "views"
    code = main(["render-views", "--model", str(mesh_file), "--output", str(out), "--views", "front,top",
                 "--size", "64", "--log-level", "WARNING"])
    assert code == 0
    assert (out / "front.png").is_file() and (out / "top.png").is_file() and (out / "views.json").is_file()

    variant = tmp_path / "variant.glb"
    box = trimesh.creation.box(extents=[1.0, 2.0, 1.0])
    knob = trimesh.creation.box(extents=[0.6, 0.6, 0.6])
    knob.apply_translation([0.7, 1.2, 0.7])
    trimesh.util.concatenate([box, knob]).export(variant)

    runs = tmp_path / "runs"
    code = main(["compare-models", "--reference", str(mesh_file), "--candidate", str(variant), "--output", str(runs),
                 "--size", "96", "--canvas-size", "96", "--device", "cpu", "--log-level", "WARNING"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "overall_score" in printed and "front.png" in printed
    run = next(runs.glob("run_*"))
    assert (run / "renders" / "reference" / "front.png").is_file()
    assert (run / "renders" / "candidate" / "top.png").is_file()
    assert (run / "metrics.json").is_file() and (run / "models.json").is_file()
    metrics = json.loads((run / "metrics.json").read_text())
    assert 0 < metrics["overall_score"] < 100

    same = main(["compare-models", "--reference", str(mesh_file), "--candidate", str(mesh_file), "--no-save",
                 "--size", "64", "--canvas-size", "64", "--device", "cpu", "--log-level", "WARNING"])
    assert same == 0
    assert "100.00" in capsys.readouterr().out


def test_cli_compare_models_bad_model_exit_code(tmp_path: Path, capsys) -> None:
    from src.cli import main

    code = main(["compare-models", "--reference", str(tmp_path / "nope.glb"), "--candidate", str(tmp_path / "x.glb"),
                 "--no-save", "--log-level", "WARNING"])
    assert code == 4
    assert "not found" in capsys.readouterr().err
