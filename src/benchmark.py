"""Benchmark orchestration: discover, pair, preprocess, score and save.

The central class is :class:`BenchmarkRunner`. A run produces a directory

    outputs/run_YYYYMMDD_HHMMSS/
        preprocessed/reference/*.png
        preprocessed/candidate/*.png
        comparisons/*.png
        metrics.json
        metrics.csv
        report.png  (+ report_page02.png ... for many pairs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import reporting
from .alignment import align_candidate
from .config import BenchmarkConfig, METRIC_NAMES
from .metrics import (
    LPIPSMetric,
    combine_scores,
    compute_edge_similarity,
    compute_silhouette_iou,
    compute_ssim,
    iou_to_score,
    lpips_to_score,
    ssim_to_score,
)
from .preprocessing import ImageLoadError, PreprocessedImage, preprocess_image, save_preprocessed

logger = logging.getLogger(__name__)


class PairingError(RuntimeError):
    """Raised when reference/candidate folders cannot be paired."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ImagePair:
    """A reference/candidate pair matched by (case-insensitive) file name."""

    name: str
    reference: Path
    candidate: Path


@dataclass
class PairResult:
    """All metrics for one image pair. ``None`` means *not available*."""

    name: str
    reference_path: str
    candidate_path: str
    ssim: float | None = None
    ssim_score: float | None = None
    lpips_distance: float | None = None
    lpips_score: float | None = None
    silhouette_iou: float | None = None
    silhouette_score: float | None = None
    edge_score: float | None = None
    edge_chamfer_ref_to_cand: float | None = None
    edge_chamfer_cand_to_ref: float | None = None
    pair_score: float | None = None
    effective_weights: dict[str, float] = field(default_factory=dict)
    unavailable_metrics: list[str] = field(default_factory=list)
    mask_source_reference: str = "none"
    mask_source_candidate: str = "none"
    alignment: dict[str, Any] | None = None
    error: str | None = None
    preprocessed_reference: str | None = None
    preprocessed_candidate: str | None = None
    comparison_image: str | None = None

    # In-memory only; excluded from serialisation.
    ref_image: PreprocessedImage | None = field(default=None, repr=False, compare=False)
    cand_image: PreprocessedImage | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.error is None and self.pair_score is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (raw, unrounded floats)."""
        return {
            "reference_path": self.reference_path,
            "candidate_path": self.candidate_path,
            "ssim": self.ssim,
            "ssim_score": self.ssim_score,
            "lpips_distance": self.lpips_distance,
            "lpips_score": self.lpips_score,
            "silhouette_iou": self.silhouette_iou,
            "silhouette_score": self.silhouette_score,
            "edge_score": self.edge_score,
            "edge_chamfer_ref_to_cand": self.edge_chamfer_ref_to_cand,
            "edge_chamfer_cand_to_ref": self.edge_chamfer_cand_to_ref,
            "pair_score": self.pair_score,
            "effective_weights": self.effective_weights,
            "unavailable_metrics": self.unavailable_metrics,
            "mask_source_reference": self.mask_source_reference,
            "mask_source_candidate": self.mask_source_candidate,
            "alignment": self.alignment,
            "error": self.error,
            "preprocessed_reference": self.preprocessed_reference,
            "preprocessed_candidate": self.preprocessed_candidate,
            "comparison_image": self.comparison_image,
        }


@dataclass
class BenchmarkResult:
    """Result of a full run."""

    pairs: list[PairResult]
    overall_score: float | None
    run_dir: Path | None
    config: dict[str, Any]
    skipped_unmatched: dict[str, list[str]] = field(default_factory=dict)
    group_scores: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def valid_pairs(self) -> list[PairResult]:
        return [p for p in self.pairs if p.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": {p.name: p.to_dict() for p in self.pairs},
            "overall_score": self.overall_score,
            "group_scores": self.group_scores,
            "num_pairs": len(self.pairs),
            "num_valid_pairs": len(self.valid_pairs),
            "skipped_unmatched": self.skipped_unmatched,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "configuration": self.config,
        }


# ---------------------------------------------------------------------------
# Discovery and pairing
# ---------------------------------------------------------------------------
def discover_images(folder: str | Path, extensions: Sequence[str]) -> dict[str, Path]:
    """Return ``{lower-case file name: path}`` for supported images in ``folder``.

    Raises:
        PairingError: if the folder is missing or two files differ only by case.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise PairingError(f"Input folder does not exist or is not a directory: {folder}")
    exts = {e.lower() for e in extensions}
    found: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        key = path.name.lower()
        if key in found:
            raise PairingError(
                f"Ambiguous pairing in {folder}: '{found[key].name}' and '{path.name}' differ only by case"
            )
        found[key] = path
    return found


def pair_images(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    extensions: Sequence[str],
    skip_unmatched: bool = False,
) -> tuple[list[ImagePair], dict[str, list[str]]]:
    """Pair images in two folders by case-insensitive file name.

    Returns:
        ``(pairs, skipped)`` where ``skipped`` lists unmatched names per side.

    Raises:
        PairingError: if no images are found, or if some are unmatched and
            ``skip_unmatched`` is ``False``.
    """
    refs = discover_images(reference_dir, extensions)
    cands = discover_images(candidate_dir, extensions)
    if not refs:
        raise PairingError(f"No supported images ({', '.join(extensions)}) found in reference folder {reference_dir}")
    if not cands:
        raise PairingError(f"No supported images ({', '.join(extensions)}) found in candidate folder {candidate_dir}")

    common = sorted(set(refs) & set(cands))
    only_ref = sorted(set(refs) - set(cands))
    only_cand = sorted(set(cands) - set(refs))
    skipped = {
        "reference_without_candidate": [refs[k].name for k in only_ref],
        "candidate_without_reference": [cands[k].name for k in only_cand],
    }

    if only_ref or only_cand:
        lines = []
        if only_ref:
            lines.append("  reference images with no candidate: " + ", ".join(skipped["reference_without_candidate"]))
        if only_cand:
            lines.append("  candidate images with no reference: " + ", ".join(skipped["candidate_without_reference"]))
        message = "Unmatched images (pairing is by case-insensitive file name):\n" + "\n".join(lines)
        if not skip_unmatched:
            raise PairingError(message + "\nSet input.skip_unmatched: true (or pass --skip-unmatched) to ignore them.")
        logger.warning("%s\nThese files will be skipped.", message)

    if not common:
        raise PairingError("No matching file names between reference and candidate folders")

    pairs = [ImagePair(name=refs[k].name, reference=refs[k], candidate=cands[k]) for k in common]
    logger.info("Found %d image pair(s)", len(pairs))
    return pairs, skipped


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class BenchmarkRunner:
    """Compute all metrics for image pairs under a given configuration."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._lpips: LPIPSMetric | None = None

    @property
    def lpips(self) -> LPIPSMetric:
        if self._lpips is None:
            self._lpips = LPIPSMetric(self.config.metrics.lpips)
        return self._lpips

    # -- scoring ----------------------------------------------------------
    def score_preprocessed(self, name: str, ref: PreprocessedImage, cand: PreprocessedImage) -> PairResult:
        """Compute every metric for two already-preprocessed images."""
        result = PairResult(
            name=name,
            reference_path=str(ref.meta.get("path", "")),
            candidate_path=str(cand.meta.get("path", "")),
            mask_source_reference=ref.mask_source if ref.mask_reliable else "none",
            mask_source_candidate=cand.mask_source if cand.mask_reliable else "none",
            ref_image=ref,
            cand_image=cand,
        )
        mcfg = self.config.metrics

        result.ssim = compute_ssim(ref.rgb, cand.rgb, mcfg.ssim)
        result.ssim_score = ssim_to_score(result.ssim)

        result.lpips_distance = self.lpips.distance(ref.rgb, cand.rgb)
        result.lpips_score = lpips_to_score(result.lpips_distance)

        mask_ref = ref.mask if ref.has_mask else None
        mask_cand = cand.mask if cand.has_mask else None
        result.silhouette_iou = compute_silhouette_iou(mask_ref, mask_cand)
        result.silhouette_score = iou_to_score(result.silhouette_iou)
        if result.silhouette_score is None:
            logger.info("%s: silhouette IoU not available (no reliable foreground mask on both images)", name)

        edge = compute_edge_similarity(ref.rgb, cand.rgb, mcfg.edge)
        result.edge_score = edge.score
        result.edge_chamfer_ref_to_cand = edge.chamfer_ref_to_cand
        result.edge_chamfer_cand_to_ref = edge.chamfer_cand_to_ref
        if result.edge_score is None:
            logger.info("%s: edge score not available (no edges detected in either image)", name)

        scores = {
            "lpips": result.lpips_score,
            "ssim": result.ssim_score,
            "silhouette": result.silhouette_score,
            "edge": result.edge_score,
        }
        result.pair_score, result.effective_weights = combine_scores(scores, self.config.weights)
        result.unavailable_metrics = [m for m in METRIC_NAMES if scores[m] is None]
        return result

    def compare_pair(
        self,
        reference: str | Path,
        candidate: str | Path,
        name: str | None = None,
        run_dir: Path | None = None,
    ) -> PairResult:
        """Preprocess and score one pair. Errors are captured in ``result.error``."""
        reference, candidate = Path(reference), Path(candidate)
        name = name or reference.name
        pcfg = self.config.preprocessing
        ocfg = self.config.output
        try:
            ref = preprocess_image(reference, pcfg)
            cand = preprocess_image(candidate, pcfg)
        except ImageLoadError as exc:
            logger.error("%s: %s", name, exc)
            return PairResult(name=name, reference_path=str(reference), candidate_path=str(candidate), error=str(exc))

        cand, align_info = align_candidate(ref, cand, pcfg)
        result = self.score_preprocessed(name, ref, cand)
        result.alignment = align_info.to_dict()
        if align_info.applied and align_info.shift_px != (0, 0):
            logger.info(
                "%s: candidate aligned by (%+d, %+d) px via %s (%.1f%%, %.1f%% of canvas)",
                name, align_info.shift_px[0], align_info.shift_px[1], align_info.method,
                100 * align_info.shift_fraction[0], 100 * align_info.shift_fraction[1],
            )

        if run_dir is not None:
            if ocfg.save_preprocessed:
                p_ref = save_preprocessed(ref, run_dir / "preprocessed" / "reference", name)
                p_cand = save_preprocessed(cand, run_dir / "preprocessed" / "candidate", name)
                result.preprocessed_reference = str(p_ref.relative_to(run_dir))
                result.preprocessed_candidate = str(p_cand.relative_to(run_dir))
            if ocfg.save_comparisons:
                comp = reporting.save_comparison_image(result, run_dir / "comparisons")
                result.comparison_image = str(comp.relative_to(run_dir))

        logger.info(
            "%s: SSIM=%.4f (%.1f)  LPIPS=%.4f (%.1f)  IoU=%s  Edge=%s  -> pair_score=%.2f%s",
            name,
            result.ssim,
            result.ssim_score,
            result.lpips_distance,
            result.lpips_score,
            "n/a" if result.silhouette_iou is None else f"{result.silhouette_iou:.4f}",
            "n/a" if result.edge_score is None else f"{result.edge_score:.1f}",
            result.pair_score if result.pair_score is not None else float("nan"),
            f"  [unavailable: {', '.join(result.unavailable_metrics)}]" if result.unavailable_metrics else "",
        )
        return result

    # -- full run ---------------------------------------------------------
    @staticmethod
    def create_run_dir(output_root: str | Path) -> Path:
        """Create ``output_root/run_YYYYMMDD_HHMMSS`` (suffixed if it exists)."""
        output_root = Path(output_root)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"run_{stamp}"
        counter = 1
        while run_dir.exists():
            counter += 1
            run_dir = output_root / f"run_{stamp}_{counter}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def run(
        self,
        reference_dir: str | Path,
        candidate_dir: str | Path,
        output_root: str | Path | None,
        skip_unmatched: bool | None = None,
        run_dir: str | Path | None = None,
    ) -> BenchmarkResult:
        """Run the full benchmark over two folders.

        Args:
            reference_dir: folder of reference images.
            candidate_dir: folder of candidate images.
            output_root: where to create the ``run_*`` directory. ``None``
                disables all file output.
            skip_unmatched: override ``input.skip_unmatched`` from the config.
            run_dir: write into this existing directory instead of creating a
                new ``run_*`` folder under ``output_root`` (used by
                ``compare-models``, which puts the renders there first).

        Raises:
            PairingError: on unmatched images (unless skipping) or empty folders.
        """
        if skip_unmatched is None:
            skip_unmatched = self.config.input.skip_unmatched
        pairs, skipped = pair_images(reference_dir, candidate_dir, self.config.input.extensions, skip_unmatched)

        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        elif output_root is not None:
            run_dir = self.create_run_dir(output_root)
        if run_dir is not None:
            logger.info("Run directory: %s", run_dir)

        results = [self.compare_pair(p.reference, p.candidate, p.name, run_dir) for p in pairs]
        overall = compute_overall_score(results)
        groups = compute_group_scores(results, self.config.output.group_separator)
        bench = BenchmarkResult(
            pairs=results,
            overall_score=overall,
            run_dir=run_dir,
            config=self.config.to_dict(),
            skipped_unmatched=skipped,
            group_scores=groups,
        )
        for group, info in groups.items():
            logger.info("Group %s: score %.2f over %d pair(s)", group, info["score"], info["num_pairs"])

        failed = [r.name for r in results if not r.ok]
        if failed:
            logger.warning("%d pair(s) failed and were excluded from the overall score: %s", len(failed), failed)
        if overall is None:
            logger.error("No valid pairs; overall score is not available")
        else:
            logger.info("Overall score: %.2f over %d valid pair(s)", overall, len(bench.valid_pairs))

        if run_dir is not None:
            reporting.save_metrics_json(bench, run_dir / "metrics.json")
            reporting.save_metrics_csv(bench, run_dir / "metrics.csv")
            reporting.save_report(bench, run_dir, pairs_per_page=self.config.output.report_pairs_per_page)
        return bench


def compute_overall_score(results: Sequence[PairResult]) -> float | None:
    """Mean of all valid pair scores; ``None`` when there are none."""
    valid = [r.pair_score for r in results if r.ok and r.pair_score is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def group_name(file_name: str, separator: str) -> str | None:
    """Return the object/group prefix of ``mug_front.png`` -> ``mug``.

    ``None`` when grouping is disabled or the stem contains no separator.
    """
    if not separator:
        return None
    stem = Path(file_name).stem
    if separator not in stem:
        return None
    prefix = stem.split(separator, 1)[0]
    return prefix or None


def compute_group_scores(results: Sequence[PairResult], separator: str) -> dict[str, dict[str, Any]]:
    """Average valid pair scores per object prefix.

    Returns ``{group: {"score": mean, "num_pairs": n, "pairs": [names]}}``
    sorted by group name. Pairs without a separator in their name are not
    grouped; when no pair has one, the result is empty.
    """
    buckets: dict[str, list[PairResult]] = {}
    for r in results:
        g = group_name(r.name, separator)
        if g is None or not r.ok or r.pair_score is None:
            continue
        buckets.setdefault(g, []).append(r)
    return {
        g: {
            "score": float(np.mean([r.pair_score for r in rs])),
            "num_pairs": len(rs),
            "pairs": [r.name for r in rs],
        }
        for g, rs in sorted(buckets.items())
    }
