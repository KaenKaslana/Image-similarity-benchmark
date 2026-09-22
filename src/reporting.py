"""Output writers: metrics.json, metrics.csv, comparison images, report.png.

Numbers in the JSON/CSV files are written unrounded; rounding to two decimals
only happens in the rendered images and console output.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from .benchmark import BenchmarkResult, PairResult

logger = logging.getLogger(__name__)

CSV_COLUMNS: tuple[str, ...] = (
    "name",
    "ssim",
    "ssim_score",
    "lpips_distance",
    "lpips_score",
    "silhouette_iou",
    "silhouette_score",
    "edge_score",
    "pair_score",
    "unavailable_metrics",
    "mask_source_reference",
    "mask_source_candidate",
    "alignment_shift_px",
    "error",
    "reference_path",
    "candidate_path",
)


def _fmt(value: float | None, digits: int = 2, na: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return na
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Tabular outputs
# ---------------------------------------------------------------------------
def save_metrics_json(result: "BenchmarkResult", path: str | Path) -> Path:
    """Write the full result (raw floats) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", path)
    return path


def save_metrics_csv(result: "BenchmarkResult", path: str | Path) -> Path:
    """Write one row per pair plus a trailing ``__overall__`` row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for p in result.pairs:
            row: dict[str, Any] = {c: getattr(p, c, None) for c in CSV_COLUMNS}
            row["unavailable_metrics"] = ";".join(p.unavailable_metrics)
            if p.alignment and p.alignment.get("applied"):
                dx, dy = p.alignment.get("shift_px", (0, 0))
                row["alignment_shift_px"] = f"{dx};{dy}"
            else:
                row["alignment_shift_px"] = ""
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
        for group, info in result.group_scores.items():
            writer.writerow({"name": f"__group__:{group}", "pair_score": info["score"]})
        writer.writerow({"name": "__overall__", "pair_score": "" if result.overall_score is None else result.overall_score})
    logger.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def difference_image(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Per-pixel absolute difference magnitude, ``(H, W)`` float in ``[0, 1]``."""
    diff = np.abs(ref.astype(np.int16) - cand.astype(np.int16)).mean(axis=-1)
    return diff.astype(np.float32) / 255.0


def _metrics_text(p: "PairResult") -> str:
    lines = [
        f"SSIM: {_fmt(p.ssim, 4)}  (score {_fmt(p.ssim_score)})",
        f"LPIPS dist: {_fmt(p.lpips_distance, 4)}  (score {_fmt(p.lpips_score)})",
        f"Silhouette IoU: {_fmt(p.silhouette_iou, 4)}  (score {_fmt(p.silhouette_score)})",
        f"Edge score: {_fmt(p.edge_score)}",
        f"Pair score: {_fmt(p.pair_score)}",
    ]
    raw = {"ssim": p.ssim_score, "lpips": p.lpips_score, "silhouette": p.silhouette_score, "edge": p.edge_score}
    if any(p.calibrated_scores.get(k) != v for k, v in raw.items()):
        cal = p.calibrated_scores
        lines.insert(
            4,
            "after floors: ssim {} lpips {} sil {} edge {}".format(
                _fmt(cal.get("ssim"), 0), _fmt(cal.get("lpips"), 0), _fmt(cal.get("silhouette"), 0), _fmt(cal.get("edge"), 0)
            ),
        )
    if p.unavailable_metrics:
        lines.append("not available: " + ", ".join(p.unavailable_metrics))
    if p.alignment and p.alignment.get("method", "none") != "none":
        dx, dy = p.alignment.get("shift_px", (0, 0))
        state = "applied" if p.alignment.get("applied") else "NOT applied"
        lines.append(f"aligned: shift ({dx:+d}, {dy:+d}) px, {state}")
    if p.error:
        lines.append(f"ERROR: {p.error}")
    return "\n".join(lines)


def _draw_pair_row(axes: Sequence[Any], p: "PairResult", label: str | None = None) -> None:
    """Draw reference | candidate | difference | text into four axes.

    ``label`` (default: the file name) is shown as the row title, e.g. the
    view name inside a per-object report.
    """
    ax_ref, ax_cand, ax_diff, ax_txt = axes
    for ax in (ax_ref, ax_cand, ax_diff, ax_txt):
        ax.set_xticks([])
        ax.set_yticks([])
    if p.ref_image is not None and p.cand_image is not None:
        ax_ref.imshow(p.ref_image.rgb)
        ax_cand.imshow(p.cand_image.rgb)
        ax_diff.imshow(difference_image(p.ref_image.rgb, p.cand_image.rgb), cmap="magma", vmin=0.0, vmax=1.0)
    else:
        for ax in (ax_ref, ax_cand, ax_diff):
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
    label = label or p.name
    ax_ref.set_title(f"{label}  |  reference", fontsize=10, loc="left", fontweight="bold")
    ax_cand.set_title("candidate", fontsize=9)
    ax_diff.set_title("|difference|", fontsize=9)
    ax_txt.axis("off")
    ax_txt.text(0.0, 0.95, _metrics_text(p), fontsize=9, family="monospace", va="top", ha="left", transform=ax_txt.transAxes)


def view_name(file_name: str, separator: str) -> str:
    """``mug_front.png`` -> ``front`` (or the whole stem when there is no separator)."""
    stem = Path(file_name).stem
    if separator and separator in stem:
        return stem.split(separator, 1)[1] or stem
    return stem


def save_comparison_image(p: "PairResult", directory: str | Path) -> Path:
    """Save a single-row comparison figure for one pair."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{Path(p.name).stem}.png"
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    try:
        _draw_pair_row(axes, p)
        fig.tight_layout()
        fig.savefig(out, dpi=110)
    finally:
        plt.close(fig)
    return out


def _render_pages(
    pairs: Sequence["PairResult"],
    title: str,
    directory: Path,
    base_name: str,
    pairs_per_page: int,
    labels: Sequence[str] | None = None,
) -> list[Path]:
    """Render ``pairs`` as one or more PNG pages named ``base_name[_pageNN].png``."""
    directory.mkdir(parents=True, exist_ok=True)
    labels = list(labels) if labels is not None else [p.name for p in pairs]
    items = list(zip(pairs, labels))
    pages = [items[i : i + pairs_per_page] for i in range(0, len(items), pairs_per_page)]
    written: list[Path] = []
    for page_idx, page in enumerate(pages, start=1):
        rows = len(page)
        fig, axes = plt.subplots(
            rows, 4, figsize=(15, 3.6 * rows + 0.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]}, squeeze=False
        )
        try:
            for r, (p, label) in enumerate(page):
                _draw_pair_row(axes[r], p, label=label)
            page_title = title + (f"  - page {page_idx}/{len(pages)}" if len(pages) > 1 else "")
            fig.suptitle(page_title, fontsize=13, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            name = f"{base_name}.png" if page_idx == 1 else f"{base_name}_page{page_idx:02d}.png"
            out = directory / name
            fig.savefig(out, dpi=100)
            written.append(out)
        finally:
            plt.close(fig)
    return written


def _score_color(score: float | None) -> tuple[float, float, float, float]:
    """Light red -> yellow -> green background for a 0-100 score (grey for n/a)."""
    if score is None:
        return (0.92, 0.92, 0.92, 1.0)
    r, g, b, _ = plt.get_cmap("RdYlGn")(float(np.clip(score, 0.0, 100.0)) / 100.0)
    return (0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b, 1.0)


def save_summary_report(result: "BenchmarkResult", path: str | Path) -> Path:
    """One-page table: one row per object, one column per view, plus object
    and overall scores. Used as ``report.png`` when the run contains groups."""
    from .benchmark import group_name

    path = Path(path)
    separator = str(result.config.get("output", {}).get("group_separator", "_"))
    views: list[str] = []
    table: dict[str, dict[str, float | None]] = {}
    for p in result.pairs:
        g = group_name(p.name, separator)
        if g is None:
            continue
        v = view_name(p.name, separator)
        if v not in views:
            views.append(v)
        table.setdefault(g, {})[v] = p.pair_score if p.ok else None

    groups = list(result.group_scores)
    col_labels = ["object"] + views + ["object score"]
    cell_text: list[list[str]] = []
    cell_colors: list[list[tuple[float, float, float, float]]] = []
    for g in groups:
        row_scores = [table.get(g, {}).get(v) for v in views]
        obj_score = result.group_scores[g]["score"]
        cell_text.append([g] + [_fmt(s) for s in row_scores] + [_fmt(obj_score)])
        cell_colors.append([(1, 1, 1, 1)] + [_score_color(s) for s in row_scores] + [_score_color(obj_score)])
    cell_text.append(["overall"] + [""] * len(views) + [_fmt(result.overall_score)])
    cell_colors.append([(0.97, 0.97, 0.97, 1)] * (len(views) + 1) + [_score_color(result.overall_score)])

    n_rows, n_cols = len(cell_text), len(col_labels)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * n_cols), 0.36 * (n_rows + 1) + 0.9))
    try:
        ax.axis("off")
        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            cellLoc="center",
            loc="upper center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0 or c == 0 or r == n_rows:
                cell.set_text_props(fontweight="bold")
            if c == n_cols - 1 and r > 0:
                cell.set_text_props(fontweight="bold")
        ax.set_title(
            f"Image similarity benchmark - overall score: {_fmt(result.overall_score)}"
            f"  ({len(result.valid_pairs)}/{len(result.pairs)} valid pairs, {len(groups)} objects)\n"
            "pair scores per view (0-100); object score = mean of its views; overall = mean of all valid pairs",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)
    return path


def save_report(result: "BenchmarkResult", run_dir: str | Path, pairs_per_page: int = 6) -> list[Path]:
    """Write the report images for a run.

    * With object groups (``mug_front.png`` style names): one
      ``report_<object>.png`` per object showing its views (front / side /
      top ...) as rows, and ``report.png`` as a one-page score table.
    * Without groups (plain ``front.png`` names, i.e. a single object):
      ``report.png`` shows every view as a row, paged by ``pairs_per_page``.

    Returns the list of written files.
    """
    from .benchmark import group_name

    run_dir = Path(run_dir)
    pairs = list(result.pairs)
    if not pairs:
        return []
    separator = str(result.config.get("output", {}).get("group_separator", "_"))
    written: list[Path] = []

    if result.group_scores:
        for group, info in result.group_scores.items():
            group_pairs = [p for p in pairs if group_name(p.name, separator) == group]
            labels = [view_name(p.name, separator) for p in group_pairs]
            per_view = "  ".join(f"{lab} {_fmt(p.pair_score)}" for p, lab in zip(group_pairs, labels))
            title = f"{group}  -  object score {_fmt(info['score'])}     ({per_view})"
            written += _render_pages(group_pairs, title, run_dir, f"report_{group}", max(pairs_per_page, 1), labels)
        ungrouped = [p for p in pairs if group_name(p.name, separator) is None]
        if ungrouped:
            title = f"ungrouped pairs  ({len(ungrouped)}; they count toward the overall score only)"
            written += _render_pages(ungrouped, title, run_dir, "report_ungrouped", pairs_per_page)
        written.append(save_summary_report(result, run_dir / "report.png"))
    else:
        title = (
            f"Image similarity benchmark - overall score: {_fmt(result.overall_score)}"
            f"  ({len(result.valid_pairs)}/{len(pairs)} valid pairs)"
        )
        written += _render_pages(pairs, title, run_dir, "report", pairs_per_page, [view_name(p.name, "") for p in pairs])

    logger.info("Wrote report(s): %s", ", ".join(w.name for w in written))
    return written


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def format_summary_table(result: "BenchmarkResult") -> str:
    """Human-readable table for the terminal."""
    header = f"{'name':<24} {'SSIM':>7} {'LPIPS':>7} {'IoU':>7} {'Edge':>7} {'Pair':>7}"
    lines = [header, "-" * len(header)]
    for p in result.pairs:
        if p.error:
            lines.append(f"{p.name[:24]:<24} ERROR: {p.error}")
            continue
        lines.append(
            f"{p.name[:24]:<24} {_fmt(p.ssim, 3):>7} {_fmt(p.lpips_distance, 3):>7} "
            f"{_fmt(p.silhouette_iou, 3):>7} {_fmt(p.edge_score, 1):>7} {_fmt(p.pair_score, 2):>7}"
        )
    lines.append("-" * len(header))
    for group, info in result.group_scores.items():
        label = f"[{group}] ({info['num_pairs']} views)"
        lines.append(f"{label[:24]:<24} {'':>7} {'':>7} {'':>7} {'':>7} {_fmt(info['score']):>7}")
    if result.group_scores:
        lines.append("-" * len(header))
    lines.append(f"{'overall_score':<24} {'':>7} {'':>7} {'':>7} {'':>7} {_fmt(result.overall_score):>7}")
    return "\n".join(lines)
