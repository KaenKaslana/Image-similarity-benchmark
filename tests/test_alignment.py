"""Tests for translation alignment (src/alignment.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.alignment import (
    align_candidate,
    estimate_shift_centroid,
    estimate_shift_phase_correlation,
    translate_image,
)
from src.config import ConfigError, PreprocessingConfig, config_from_dict
from src.objects import mug, render_object_views
from src.preprocessing import preprocess_pil
from src.synthetic import draw_shape


def _cfg(**overrides) -> PreprocessingConfig:
    cfg = PreprocessingConfig(canvas_size=128, **overrides)
    cfg.validate()
    return cfg


def _rect_signal(x0: int, y0: int, w: int = 40, h: int = 30, size: int = 128) -> np.ndarray:
    s = np.zeros((size, size), np.float32)
    s[y0 : y0 + h, x0 : x0 + w] = 1.0
    return s


# ---------------------------------------------------------------------------
# Shift estimation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dx,dy", [(12, 7), (-9, 4), (0, 0), (20, -15)])
def test_phase_correlation_recovers_translation(dx: int, dy: int) -> None:
    ref = _rect_signal(40, 45)
    cand = _rect_signal(40 + dx, 45 + dy)
    (ex, ey), response = estimate_shift_phase_correlation(ref, cand)
    assert abs(ex + dx) < 0.6 and abs(ey + dy) < 0.6, (ex, ey)  # correction is the negated motion
    assert response > 0.5


def test_centroid_recovers_translation() -> None:
    ref = _rect_signal(40, 45)
    cand = _rect_signal(52, 52)
    ex, ey = estimate_shift_centroid(ref, cand)
    assert abs(ex + 12) < 1e-6 and abs(ey + 7) < 1e-6
    assert estimate_shift_centroid(np.zeros((8, 8), np.float32), ref) is None


def test_phase_correlation_ignores_missing_part_centroid_does_not() -> None:
    """A mug without its handle, shifted: content-based alignment still lands
    on the body, whereas the centroid is dragged by the missing handle."""
    ref = _rect_signal(30, 40, w=40, h=40)
    ref[52:60, 70:90] = 1.0  # handle sticking out to the right
    cand = _rect_signal(30 + 10, 40 + 6, w=40, h=40)  # body only, moved by (+10, +6)
    (px, py), _ = estimate_shift_phase_correlation(ref, cand)
    cx, cy = estimate_shift_centroid(ref, cand)
    assert abs(px + 10) < 1.0 and abs(py + 6) < 1.0
    assert abs(cx + 10) > 2.0  # centroid pulled toward the handle


# ---------------------------------------------------------------------------
# Applying the shift
# ---------------------------------------------------------------------------
def test_translate_image_moves_pixels_and_mask_and_fills_background() -> None:
    img = preprocess_pil(draw_shape(64, "square", mode="RGBA", seed=None), _cfg())
    moved = translate_image(img, 10, -5, [255, 255, 255])
    assert moved.rgb.shape == img.rgb.shape and moved.mask.shape == img.mask.shape
    ys, xs = np.where(img.mask)
    ys2, xs2 = np.where(moved.mask)
    assert abs((xs2.mean() - xs.mean()) - 10) < 0.5 and abs((ys2.mean() - ys.mean()) + 5) < 0.5
    assert moved.rgb[0, 0].tolist() == [255, 255, 255]
    assert moved.meta["alignment_shift_px"] == [10, -5]
    assert img.meta.get("alignment_shift_px") is None  # original untouched


# ---------------------------------------------------------------------------
# align_candidate end to end
# ---------------------------------------------------------------------------
def test_align_candidate_on_rendered_object() -> None:
    cfg = _cfg(alignment="phase_correlation")
    ref = preprocess_pil(render_object_views(mug(), size=128)["front.png"], cfg)
    cand = preprocess_pil(render_object_views(mug(), size=128, offset=(0.08, -0.05))["front.png"], cfg)
    before = np.logical_and(ref.mask, cand.mask).sum() / np.logical_or(ref.mask, cand.mask).sum()
    aligned, info = align_candidate(ref, cand, cfg)
    after = np.logical_and(ref.mask, aligned.mask).sum() / np.logical_or(ref.mask, aligned.mask).sum()
    assert info.applied and info.signal == "mask" and info.method == "phase_correlation"
    assert info.shift_px[0] < -6 and info.shift_px[1] > 3  # undoes (+8 %, -5 %) of 128 px
    assert after > 0.95 > before + 0.2


def test_align_candidate_without_mask_uses_background_distance() -> None:
    cfg = _cfg(alignment="phase_correlation", mask_mode="alpha")  # RGB inputs -> no mask
    ref = preprocess_pil(draw_shape(96, "circle", mode="RGB"), cfg)
    cand = preprocess_pil(draw_shape(96, "circle", mode="RGB", offset=(9, -6)), cfg)
    aligned, info = align_candidate(ref, cand, cfg)
    assert info.signal == "background_distance" and info.applied
    diff_before = np.abs(ref.rgb.astype(int) - cand.rgb.astype(int)).mean()
    diff_after = np.abs(ref.rgb.astype(int) - aligned.rgb.astype(int)).mean()
    assert diff_after < diff_before * 0.5


def test_alignment_none_and_centroid() -> None:
    ref = preprocess_pil(draw_shape(64, "square", seed=None), _cfg())
    cand = preprocess_pil(draw_shape(64, "square", offset=(6, 3), seed=None), _cfg())
    same, info = align_candidate(ref, cand, _cfg(alignment="none"))
    assert same is cand and not info.applied and info.method == "none"
    moved, info_c = align_candidate(ref, cand, _cfg(alignment="centroid"))
    assert info_c.applied and info_c.method == "centroid" and info_c.response is None
    assert np.logical_and(ref.mask, moved.mask).sum() / np.logical_or(ref.mask, moved.mask).sum() > 0.95


def test_too_large_shift_is_rejected() -> None:
    cfg = _cfg(alignment="centroid", alignment_max_shift=0.05)
    ref = preprocess_pil(draw_shape(64, "square", scale=0.3, offset=(-15, 0), seed=None), cfg)
    cand = preprocess_pil(draw_shape(64, "square", scale=0.3, offset=(15, 0), seed=None), cfg)
    out, info = align_candidate(ref, cand, cfg)
    assert out is cand and not info.applied and "exceeds" in info.note


def test_alignment_config_validation() -> None:
    with pytest.raises(ConfigError, match="alignment must be one of"):
        config_from_dict({"preprocessing": {"alignment": "magic"}})
    with pytest.raises(ConfigError, match="alignment_max_shift"):
        config_from_dict({"preprocessing": {"alignment_max_shift": 0}})
    cfg = config_from_dict({})
    assert cfg.preprocessing.alignment == "phase_correlation"
