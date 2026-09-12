"""Translation alignment of a candidate image to its reference.

The benchmark measures *object* similarity, so a difference in where the
object sits inside the frame should not cost points. Before the metrics are
computed the candidate is shifted (translation only, never scaled or rotated)
so that it overlaps the reference as well as possible.

Methods:

* ``none``               - keep images as they are.
* ``centroid``           - align the centroids of the foreground signal.
* ``phase_correlation``  - OpenCV phase correlation on the foreground signal
  (content based: a missing or extra part does not drag the main body away,
  unlike bounding-box based cropping). Recommended default.

The "foreground signal" is the reliable foreground mask when both images
have one, otherwise the per-pixel distance from the background colour.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .config import ALIGNMENT_METHODS, PreprocessingConfig
from .preprocessing import PreprocessedImage

logger = logging.getLogger(__name__)

__all__ = ["ALIGNMENT_METHODS", "AlignmentInfo", "align_candidate", "translate_image"]


@dataclass
class AlignmentInfo:
    """What the alignment step did for one pair."""

    method: str
    shift_px: tuple[int, int]  # (dx, dy) applied to the candidate, in canvas pixels
    shift_fraction: tuple[float, float]  # same shift relative to the canvas size
    response: float | None  # phase-correlation peak response (None for other methods)
    signal: str  # "mask" or "background_distance"
    applied: bool
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shift_px"] = list(self.shift_px)
        d["shift_fraction"] = list(self.shift_fraction)
        return d


# ---------------------------------------------------------------------------
# Foreground signal
# ---------------------------------------------------------------------------
def foreground_signal(img: PreprocessedImage, background: list[int], use_mask: bool) -> np.ndarray:
    """Float32 ``(H, W)`` map in ``[0, 1]``: 1 = clearly object, 0 = background."""
    if use_mask and img.mask is not None:
        return img.mask.astype(np.float32)
    bg = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    diff = np.abs(img.rgb.astype(np.float32) - bg).max(axis=-1) / 255.0
    return diff.astype(np.float32)


def _signals(ref: PreprocessedImage, cand: PreprocessedImage, background: list[int]) -> tuple[np.ndarray, np.ndarray, str]:
    use_mask = ref.has_mask and cand.has_mask
    return (
        foreground_signal(ref, background, use_mask),
        foreground_signal(cand, background, use_mask),
        "mask" if use_mask else "background_distance",
    )


# ---------------------------------------------------------------------------
# Shift estimation
# ---------------------------------------------------------------------------
def _centroid(signal: np.ndarray) -> tuple[float, float] | None:
    total = float(signal.sum())
    if total <= 0:
        return None
    ys, xs = np.indices(signal.shape)
    return float((xs * signal).sum() / total), float((ys * signal).sum() / total)


def estimate_shift_centroid(sig_ref: np.ndarray, sig_cand: np.ndarray) -> tuple[float, float] | None:
    """Shift ``(dx, dy)`` that moves the candidate centroid onto the reference centroid."""
    c_ref, c_cand = _centroid(sig_ref), _centroid(sig_cand)
    if c_ref is None or c_cand is None:
        return None
    return c_ref[0] - c_cand[0], c_ref[1] - c_cand[1]


def estimate_shift_phase_correlation(sig_ref: np.ndarray, sig_cand: np.ndarray) -> tuple[tuple[float, float], float]:
    """Shift ``(dx, dy)`` to apply to the candidate so it overlaps the reference.

    Uses ``cv2.phaseCorrelate`` with a Hanning window. OpenCV returns the
    translation of the second image relative to the first, so the correction
    to apply to the candidate is its negation.
    """
    window = cv2.createHanningWindow((sig_ref.shape[1], sig_ref.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(sig_ref, sig_cand, window)
    return (-float(dx), -float(dy)), float(response)


# ---------------------------------------------------------------------------
# Applying the shift
# ---------------------------------------------------------------------------
def translate_image(img: PreprocessedImage, dx: int, dy: int, background: list[int]) -> PreprocessedImage:
    """Return a copy of ``img`` shifted by integer ``(dx, dy)`` pixels.

    Uncovered areas are filled with the background colour (mask: ``False``).
    """
    h, w = img.rgb.shape[:2]
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    rgb = cv2.warpAffine(
        img.rgb, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(c) for c in background),
    )
    mask = None
    if img.mask is not None:
        m = cv2.warpAffine(
            img.mask.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask = m.astype(bool)
    meta = dict(img.meta)
    meta["alignment_shift_px"] = [int(dx), int(dy)]
    return PreprocessedImage(rgb=rgb, mask=mask, mask_source=img.mask_source, mask_reliable=img.mask_reliable, meta=meta)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def align_candidate(
    ref: PreprocessedImage,
    cand: PreprocessedImage,
    cfg: PreprocessingConfig,
) -> tuple[PreprocessedImage, AlignmentInfo]:
    """Shift ``cand`` so that its object overlaps the one in ``ref``.

    Returns the (possibly unchanged) candidate and an :class:`AlignmentInfo`.
    The shift is rejected (image left untouched, ``applied=False``) when it
    exceeds ``alignment_max_shift`` of the canvas, which almost always means
    the estimate is unreliable (e.g. two unrelated images).
    """
    method = cfg.alignment
    size = int(cfg.canvas_size)
    if method == "none":
        return cand, AlignmentInfo("none", (0, 0), (0.0, 0.0), None, "n/a", False)

    sig_ref, sig_cand, signal_name = _signals(ref, cand, cfg.background_color)
    response: float | None = None
    if method == "centroid":
        est = estimate_shift_centroid(sig_ref, sig_cand)
        if est is None:
            return cand, AlignmentInfo(method, (0, 0), (0.0, 0.0), None, signal_name, False, "empty foreground signal")
        fdx, fdy = est
    elif method == "phase_correlation":
        (fdx, fdy), response = estimate_shift_phase_correlation(sig_ref, sig_cand)
    else:
        raise ValueError(f"unknown alignment method {method!r}")

    dx, dy = int(round(fdx)), int(round(fdy))
    frac = (dx / size, dy / size)
    max_shift = float(cfg.alignment_max_shift)
    if abs(frac[0]) > max_shift or abs(frac[1]) > max_shift:
        note = f"estimated shift {frac} exceeds alignment_max_shift={max_shift}; not applied"
        logger.warning("%s: %s", ref.meta.get("name", "<pair>"), note)
        return cand, AlignmentInfo(method, (dx, dy), frac, response, signal_name, False, note)
    if dx == 0 and dy == 0:
        return cand, AlignmentInfo(method, (0, 0), (0.0, 0.0), response, signal_name, True, "already aligned")

    # Sanity check: only apply the shift if it actually increases the overlap of
    # the foreground signals. For two unrelated objects the estimate can be a
    # large, meaningless shift that would make things worse.
    overlap_before = _overlap(sig_ref, sig_cand)
    overlap_after = _overlap(sig_ref, _shift_signal(sig_cand, dx, dy))
    if overlap_after < overlap_before:
        note = f"shift ({dx}, {dy}) would reduce overlap ({overlap_before:.3f} -> {overlap_after:.3f}); not applied"
        logger.info("%s: %s", ref.meta.get("name", "<pair>"), note)
        return cand, AlignmentInfo(method, (dx, dy), frac, response, signal_name, False, note)

    logger.debug("%s: aligning candidate by (%d, %d) px via %s", ref.meta.get("name", "<pair>"), dx, dy, method)
    return translate_image(cand, dx, dy, cfg.background_color), AlignmentInfo(method, (dx, dy), frac, response, signal_name, True)


def _shift_signal(signal: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    h, w = signal.shape
    return cv2.warpAffine(signal, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised overlap of two non-negative signals (1 = identical, 0 = disjoint)."""
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)
