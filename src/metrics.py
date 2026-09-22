"""Similarity metrics.

All metric functions take preprocessed ``(H, W, 3)`` uint8 RGB arrays (and
optional boolean masks) of identical shape and return *raw* values plus a
0-100 score. ``None`` is returned for a score when the metric is not
available for the given inputs (for example no reliable mask for silhouette
IoU); callers must not fabricate a value in that case.

Score conversions:

* ``ssim_score       = 100 * clip(ssim, 0, 1)``
* ``lpips_score      = 100 * exp(-lpips_distance)``
* ``silhouette_score = 100 * IoU``
* ``edge_score       = 100 * (1 - symmetric truncated Chamfer distance)``
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .config import EdgeConfig, LPIPSConfig, METRIC_NAMES, SSIMConfig

logger = logging.getLogger(__name__)


def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Image shapes differ: {a.shape} vs {b.shape}; preprocess both to the same canvas first")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB arrays, got shape {a.shape}")
    if a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("Expected uint8 arrays")


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def compute_ssim(ref: np.ndarray, cand: np.ndarray, cfg: SSIMConfig | None = None) -> float:
    """Structural similarity (Wang et al. 2004) on RGB uint8 images.

    ``data_range`` is set explicitly to 255 for 8-bit data and channels are
    averaged via ``channel_axis=-1``.
    """
    cfg = cfg or SSIMConfig()
    _check_pair(ref, cand)
    kwargs: dict[str, Any] = {"channel_axis": -1, "data_range": 255}
    if cfg.gaussian_weights:
        kwargs.update(gaussian_weights=True, sigma=float(cfg.sigma), use_sample_covariance=False)
    else:
        kwargs.update(win_size=int(cfg.win_size))
    value = structural_similarity(ref, cand, **kwargs)
    return float(value)


def ssim_to_score(ssim: float) -> float:
    """Map raw SSIM in ``[-1, 1]`` to a 0-100 score by clipping to ``[0, 1]``."""
    return 100.0 * float(np.clip(ssim, 0.0, 1.0))


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------
def resolve_device(requested: str = "auto") -> str:
    """Return ``"cuda"`` or ``"cpu"`` for the requested device setting.

    ``"auto"`` picks CUDA when available. Requesting ``"cuda"`` without CUDA
    falls back to CPU with a warning instead of failing.
    """
    import torch

    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        logger.warning("CUDA requested but not available; falling back to CPU")
    return "cpu"


def image_to_lpips_tensor(rgb: np.ndarray, device: str) -> "torch.Tensor":  # noqa: F821
    """Convert an ``(H, W, 3)`` uint8 RGB array to a ``(1, 3, H, W)`` float
    tensor normalised to ``[-1, 1]`` as expected by the LPIPS network."""
    import torch

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) array, got {rgb.shape}")
    arr = rgb.astype(np.float32) / 255.0  # [0, 1]
    arr = arr * 2.0 - 1.0  # [-1, 1]
    tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))[None]
    return tensor.to(device)


class LPIPSMetric:
    """Lazy wrapper around the ``lpips`` package.

    The network is only instantiated on first use so that importing this
    module (and running non-LPIPS metrics) does not require loading weights.
    """

    def __init__(self, cfg: LPIPSConfig | None = None) -> None:
        self.cfg = cfg or LPIPSConfig()
        self._model: Any = None
        self._device: str | None = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self.cfg.device)
        return self._device

    def _ensure_model(self) -> Any:
        if self._model is None:
            import lpips
            import torch

            logger.info("Loading LPIPS (%s) on %s", self.cfg.net, self.device)
            with warnings.catch_warnings():
                # torchvision warns about the deprecated `pretrained=` kwarg used inside lpips.
                warnings.simplefilter("ignore", category=UserWarning)
                model = lpips.LPIPS(net=self.cfg.net, verbose=False)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model.to(self.device)
            del torch
        return self._model

    def distance(self, ref: np.ndarray, cand: np.ndarray) -> float:
        """Return the raw LPIPS distance (lower is more similar)."""
        import torch

        _check_pair(ref, cand)
        model = self._ensure_model()
        with torch.no_grad():
            a = image_to_lpips_tensor(ref, self.device)
            b = image_to_lpips_tensor(cand, self.device)
            d = model(a, b)
        return float(d.reshape(-1)[0].item())


def lpips_to_score(distance: float) -> float:
    """``100 * exp(-distance)``. Distance 0 -> 100; larger distances decay."""
    return 100.0 * math.exp(-float(distance))


# ---------------------------------------------------------------------------
# Silhouette IoU
# ---------------------------------------------------------------------------
def compute_silhouette_iou(mask_ref: np.ndarray | None, mask_cand: np.ndarray | None) -> float | None:
    """Intersection-over-union of two boolean foreground masks.

    Returns ``None`` when either mask is missing or when both are empty (the
    union is zero, so IoU is undefined).
    """
    if mask_ref is None or mask_cand is None:
        return None
    if mask_ref.shape != mask_cand.shape:
        raise ValueError(f"Mask shapes differ: {mask_ref.shape} vs {mask_cand.shape}")
    a = mask_ref.astype(bool)
    b = mask_cand.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    inter = int(np.logical_and(a, b).sum())
    return inter / union


def iou_to_score(iou: float | None) -> float | None:
    return None if iou is None else 100.0 * float(iou)


# ---------------------------------------------------------------------------
# Edge similarity (Chamfer-style)
# ---------------------------------------------------------------------------
def extract_edges(rgb: np.ndarray, cfg: EdgeConfig | None = None) -> np.ndarray:
    """Canny edge map (boolean) of an RGB image."""
    cfg = cfg or EdgeConfig()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = int(cfg.blur_kernel)
    if k > 0:
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(gray, int(cfg.canny_low), int(cfg.canny_high))
    return edges > 0


def _distance_to_nearest_edge(edges: np.ndarray) -> np.ndarray:
    """For every pixel, Euclidean distance to the nearest edge pixel.

    ``cv2.distanceTransform`` computes the distance to the nearest *zero*
    pixel, so the edge map is inverted first.
    """
    non_edge = np.where(edges, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(non_edge, cv2.DIST_L2, 5)


@dataclass
class EdgeResult:
    score: float | None
    chamfer_ref_to_cand: float | None
    chamfer_cand_to_ref: float | None
    edge_fraction_ref: float
    edge_fraction_cand: float


def compute_edge_similarity(ref: np.ndarray, cand: np.ndarray, cfg: EdgeConfig | None = None) -> EdgeResult:
    """Symmetric truncated Chamfer edge similarity.

    For every edge pixel in the reference, the distance to the nearest
    candidate edge pixel is looked up via a distance transform (and vice
    versa). Distances are truncated at ``max_distance`` (scaled relative to
    a 512 px canvas) and normalised to ``[0, 1]``. The score is

    ``100 * (1 - 0.5 * (mean_norm_dist_ref->cand + mean_norm_dist_cand->ref))``

    Edge cases: if both images have essentially no edges the metric is not
    available (``score=None``). If exactly one image has edges the score is 0.
    """
    cfg = cfg or EdgeConfig()
    _check_pair(ref, cand)
    e_ref = extract_edges(ref, cfg)
    e_cand = extract_edges(cand, cfg)
    frac_ref = float(e_ref.mean())
    frac_cand = float(e_cand.mean())
    min_frac = float(cfg.min_edge_fraction)

    if frac_ref < min_frac and frac_cand < min_frac:
        logger.debug("Edge metric not available: too few edges (%.5f, %.5f)", frac_ref, frac_cand)
        return EdgeResult(None, None, None, frac_ref, frac_cand)
    if not e_ref.any() or not e_cand.any():
        return EdgeResult(0.0, 1.0, 1.0, frac_ref, frac_cand)

    h = ref.shape[0]
    max_d = float(cfg.max_distance) * (h / 512.0)
    dt_ref = _distance_to_nearest_edge(e_ref)
    dt_cand = _distance_to_nearest_edge(e_cand)
    d_ref_to_cand = float(np.minimum(dt_cand[e_ref] / max_d, 1.0).mean())
    d_cand_to_ref = float(np.minimum(dt_ref[e_cand] / max_d, 1.0).mean())
    score = 100.0 * (1.0 - 0.5 * (d_ref_to_cand + d_cand_to_ref))
    return EdgeResult(float(np.clip(score, 0.0, 100.0)), d_ref_to_cand, d_cand_to_ref, frac_ref, frac_cand)


# ---------------------------------------------------------------------------
# Weighted combination
# ---------------------------------------------------------------------------
def apply_score_floor(score: float | None, floor: float, gamma: float = 1.0) -> float | None:
    """Calibrate a 0-100 score: ``100 * clip((score - floor) / (100 - floor), 0, 1) ** gamma``.

    Every metric has a "chance level" that two unrelated objects reach anyway
    (shared background for SSIM/LPIPS, two centred blobs overlapping for IoU).
    Subtracting it makes the combined score start near 0 for unrelated
    objects instead of ~40. ``gamma < 1`` then lifts the mid range, so that a
    rough but recognisable replica is not punished as hard as a linear scale
    would; the end points (floor -> 0, 100 -> 100) do not move.
    """
    if score is None:
        return None
    if floor <= 0 and gamma == 1.0:
        return float(score)
    x = min(1.0, max(0.0, (float(score) - floor) / (100.0 - floor)))
    return 100.0 * x ** float(gamma)


def combine_scores(
    scores: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> tuple[float | None, dict[str, float]]:
    """Weighted average of available metric scores with weight renormalisation.

    Args:
        scores: metric name -> 0-100 score or ``None`` (not available).
        weights: metric name -> weight (validated to sum to 1).

    Returns:
        ``(pair_score, effective_weights)``. ``effective_weights`` contains the
        renormalised weight actually used for each available metric (and is
        empty, with ``pair_score=None``, when nothing is available).
    """
    available = {k: float(v) for k, v in scores.items() if v is not None and k in weights and weights[k] > 0}
    total = sum(weights[k] for k in available)
    if not available or total <= 0:
        return None, {}
    if abs(total - 1.0) < 1e-9:  # everything available: keep the exact configured weights
        effective = {k: float(weights[k]) for k in METRIC_NAMES if k in available}
    else:
        effective = {k: weights[k] / total for k in METRIC_NAMES if k in available}
    pair_score = sum(effective[k] * available[k] for k in effective)
    return float(pair_score), effective
