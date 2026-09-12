"""Tests for the individual metrics and the weighted combination."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import ConfigError, EdgeConfig, validate_weights
from src.metrics import (
    LPIPSMetric,
    combine_scores,
    compute_edge_similarity,
    compute_silhouette_iou,
    compute_ssim,
    image_to_lpips_tensor,
    iou_to_score,
    lpips_to_score,
    resolve_device,
    ssim_to_score,
)
from src.preprocessing import preprocess_pil
from src.synthetic import adjust_brightness, draw_shape


def _pre(img, cfg):
    return preprocess_pil(img, cfg.preprocessing)


@pytest.fixture(scope="module")
def base(default_config):
    return _pre(draw_shape(256, "circle"), default_config)


# ---------------------------------------------------------------------------
# Score conversions
# ---------------------------------------------------------------------------
def test_ssim_to_score_clips_negative_and_scales() -> None:
    assert ssim_to_score(1.0) == 100.0
    assert ssim_to_score(0.5) == 50.0
    assert ssim_to_score(-0.3) == 0.0
    assert ssim_to_score(1.2) == 100.0


def test_lpips_to_score_uses_exp() -> None:
    assert lpips_to_score(0.0) == 100.0
    assert math.isclose(lpips_to_score(0.13), 100 * math.exp(-0.13))
    assert lpips_to_score(0.5) != 100 * (1 - 0.5)  # explicitly not 1 - d
    assert lpips_to_score(5.0) > 0.0


def test_iou_to_score() -> None:
    assert iou_to_score(None) is None
    assert iou_to_score(0.25) == 25.0


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def test_ssim_identical_is_one(base, default_config) -> None:
    s = compute_ssim(base.rgb, base.rgb, default_config.metrics.ssim)
    assert math.isclose(s, 1.0, abs_tol=1e-9)
    assert ssim_to_score(s) > 99.9


def test_ssim_drops_with_brightness_color_shift_shape(default_config) -> None:
    cfg = default_config
    ref = _pre(draw_shape(256, "circle"), cfg)
    variants = {
        "brightness": _pre(adjust_brightness(draw_shape(256, "circle"), 0.6), cfg),
        "color": _pre(draw_shape(256, "circle", color=(40, 90, 200)), cfg),
        "shift": _pre(draw_shape(256, "circle", offset=(15, 8)), cfg),
        "shape": _pre(draw_shape(256, "triangle"), cfg),
    }
    for name, cand in variants.items():
        s = compute_ssim(ref.rgb, cand.rgb, cfg.metrics.ssim)
        assert s < 0.99, f"{name}: SSIM {s} did not drop"


def test_ssim_rejects_shape_mismatch(default_config) -> None:
    a = np.zeros((8, 8, 3), np.uint8)
    b = np.zeros((9, 8, 3), np.uint8)
    with pytest.raises(ValueError, match="shapes differ"):
        compute_ssim(a, b, default_config.metrics.ssim)


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------
def test_lpips_tensor_normalisation() -> None:
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[0, 0] = 255
    t = image_to_lpips_tensor(rgb, "cpu")
    assert tuple(t.shape) == (1, 3, 4, 4)
    assert float(t.min()) == -1.0 and float(t.max()) == 1.0
    assert t.dtype.is_floating_point


def test_lpips_identical_is_zero(base, lpips_metric: LPIPSMetric) -> None:
    d = lpips_metric.distance(base.rgb, base.rgb)
    assert d < 1e-5
    assert lpips_to_score(d) > 99.9


def test_lpips_drops_with_perturbations(default_config, lpips_metric: LPIPSMetric) -> None:
    cfg = default_config
    ref = _pre(draw_shape(256, "circle"), cfg)
    variants = {
        "brightness": _pre(adjust_brightness(draw_shape(256, "circle"), 0.6), cfg),
        "color": _pre(draw_shape(256, "circle", color=(40, 90, 200)), cfg),
        "shift": _pre(draw_shape(256, "circle", offset=(15, 8)), cfg),
        "shape": _pre(draw_shape(256, "triangle"), cfg),
    }
    for name, cand in variants.items():
        d = lpips_metric.distance(ref.rgb, cand.rgb)
        assert d > 0.02, f"{name}: LPIPS distance {d} did not increase"
        assert lpips_to_score(d) < 99.0


def test_lpips_runs_on_cpu_when_cuda_unavailable(base, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from src.config import LPIPSConfig

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda") == "cpu"  # graceful fallback
    metric = LPIPSMetric(LPIPSConfig(net="alex", device="auto"))
    assert metric.device == "cpu"
    d = metric.distance(base.rgb, base.rgb)
    assert d < 1e-5


def test_lpips_explicit_cpu_device(base) -> None:
    from src.config import LPIPSConfig

    metric = LPIPSMetric(LPIPSConfig(net="alex", device="cpu"))
    assert metric.device == "cpu"
    assert metric.distance(base.rgb, base.rgb) < 1e-5


# ---------------------------------------------------------------------------
# Silhouette IoU
# ---------------------------------------------------------------------------
def test_silhouette_iou_basic() -> None:
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[:5, :] = True
    b[2:7, :] = True
    assert math.isclose(compute_silhouette_iou(a, b), 3 / 7)
    assert compute_silhouette_iou(a, a) == 1.0
    assert compute_silhouette_iou(a, ~a) == 0.0


def test_silhouette_iou_not_available_without_mask() -> None:
    a = np.zeros((10, 10), bool)
    assert compute_silhouette_iou(None, a) is None
    assert compute_silhouette_iou(a, None) is None
    assert compute_silhouette_iou(a, a) is None  # both empty -> undefined


def test_silhouette_drops_when_shape_or_position_changes(default_config) -> None:
    cfg = default_config
    ref = _pre(draw_shape(256, "circle", mode="RGBA"), cfg)
    same = _pre(draw_shape(256, "circle", mode="RGBA"), cfg)
    shifted = _pre(draw_shape(256, "circle", offset=(15, 8), mode="RGBA"), cfg)
    reshaped = _pre(draw_shape(256, "triangle", mode="RGBA"), cfg)
    assert compute_silhouette_iou(ref.mask, same.mask) > 0.99
    assert compute_silhouette_iou(ref.mask, shifted.mask) < 0.9
    assert compute_silhouette_iou(ref.mask, reshaped.mask) < 0.8


# ---------------------------------------------------------------------------
# Edge similarity
# ---------------------------------------------------------------------------
def test_edge_identical_is_100(base, default_config) -> None:
    r = compute_edge_similarity(base.rgb, base.rgb, default_config.metrics.edge)
    assert r.score == 100.0
    assert r.chamfer_ref_to_cand == 0.0 and r.chamfer_cand_to_ref == 0.0


def test_edge_drops_with_shape_change_and_shift(default_config) -> None:
    cfg = default_config
    ref = _pre(draw_shape(256, "circle"), cfg)
    reshaped = _pre(draw_shape(256, "triangle"), cfg)
    shifted = _pre(draw_shape(256, "circle", offset=(15, 8)), cfg)
    slightly_shifted = _pre(draw_shape(256, "circle", offset=(2, 1)), cfg)
    s_reshaped = compute_edge_similarity(ref.rgb, reshaped.rgb, cfg.metrics.edge).score
    s_shifted = compute_edge_similarity(ref.rgb, shifted.rgb, cfg.metrics.edge).score
    s_slight = compute_edge_similarity(ref.rgb, slightly_shifted.rgb, cfg.metrics.edge).score
    assert s_reshaped < 70
    assert s_shifted < 70
    # Chamfer tolerance: a 1-2 px shift should be penalised only mildly.
    assert 80 < s_slight < 100


def test_edge_not_available_for_flat_images() -> None:
    flat = np.full((64, 64, 3), 200, np.uint8)
    r = compute_edge_similarity(flat, flat, EdgeConfig())
    assert r.score is None


def test_edge_zero_when_only_one_image_has_edges(default_config) -> None:
    cfg = default_config
    ref = _pre(draw_shape(256, "circle"), cfg)
    flat = np.full_like(ref.rgb, 255)
    r = compute_edge_similarity(ref.rgb, flat, cfg.metrics.edge)
    assert r.score == 0.0


# ---------------------------------------------------------------------------
# Weights and combination
# ---------------------------------------------------------------------------
WEIGHTS = {"lpips": 0.40, "ssim": 0.30, "silhouette": 0.20, "edge": 0.10}


def test_validate_weights_ok() -> None:
    assert validate_weights(WEIGHTS) == WEIGHTS
    validate_weights({"lpips": 1, "ssim": 0, "silhouette": 0, "edge": 0})


@pytest.mark.parametrize(
    "bad, msg",
    [
        ({"lpips": 0.5, "ssim": 0.3, "silhouette": 0.2, "edge": 0.1}, "sum to 1"),
        ({"lpips": 0.5, "ssim": 0.5, "silhouette": 0.0}, "missing"),
        ({**WEIGHTS, "extra": 0.0}, "unknown"),
        ({"lpips": 1.2, "ssim": -0.2, "silhouette": 0.0, "edge": 0.0}, ">= 0"),
        ({"lpips": "0.4", "ssim": 0.3, "silhouette": 0.2, "edge": 0.1}, "must be a number"),
        ("not a mapping", "mapping"),
    ],
)
def test_validate_weights_rejects(bad, msg: str) -> None:
    with pytest.raises(ConfigError, match=msg):
        validate_weights(bad)


def test_combine_scores_all_available() -> None:
    scores = {"lpips": 87.81, "ssim": 91.0, "silhouette": 88.0, "edge": 84.2}
    pair, eff = combine_scores(scores, WEIGHTS)
    expected = 0.40 * 87.81 + 0.30 * 91.0 + 0.20 * 88.0 + 0.10 * 84.2
    assert math.isclose(pair, expected)
    assert eff == WEIGHTS


def test_combine_scores_renormalises_without_silhouette() -> None:
    scores = {"lpips": 80.0, "ssim": 90.0, "silhouette": None, "edge": 60.0}
    pair, eff = combine_scores(scores, WEIGHTS)
    expected = (0.40 * 80.0 + 0.30 * 90.0 + 0.10 * 60.0) / 0.80
    assert math.isclose(pair, expected)
    assert math.isclose(sum(eff.values()), 1.0)
    assert "silhouette" not in eff
    assert math.isclose(eff["lpips"], 0.5)


def test_combine_scores_two_metrics_missing() -> None:
    scores = {"lpips": 80.0, "ssim": None, "silhouette": None, "edge": 60.0}
    pair, eff = combine_scores(scores, WEIGHTS)
    assert math.isclose(pair, (0.4 * 80 + 0.1 * 60) / 0.5)
    assert set(eff) == {"lpips", "edge"}


def test_combine_scores_nothing_available() -> None:
    pair, eff = combine_scores({k: None for k in WEIGHTS}, WEIGHTS)
    assert pair is None and eff == {}


def test_combine_scores_ignores_zero_weight_metrics() -> None:
    weights = {"lpips": 1.0, "ssim": 0.0, "silhouette": 0.0, "edge": 0.0}
    pair, eff = combine_scores({"lpips": 70.0, "ssim": 10.0, "silhouette": 10.0, "edge": 10.0}, weights)
    assert pair == 70.0 and eff == {"lpips": 1.0}
