"""Collect every ``outputs/run_*`` into one results table (Markdown + CSV).

Usage (from the project root)::

    python scripts/summarize_runs.py                 # writes outputs/results.md and outputs/results.csv
    python scripts/summarize_runs.py --output-root other/dir

Each row is one run: what was compared (from ``models.json``), how the
candidate was generated (provider / mode / model version / credits), which
config was used (crop mode + weights), the automatic orientation, per-view
pair scores and the overall score. Runs without ``metrics.json`` are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import describe_model  # noqa: E402

COLUMNS = [
    "run", "reference", "candidate", "generation", "credits", "config", "orientation",
    "front", "back", "side", "left", "top", "bottom", "overall",
]


def _config_name(cfg: dict) -> str:
    """``shape`` / ``default`` for the shipped configs, otherwise crop mode + weights."""
    crop = (cfg.get("preprocessing") or {}).get("crop_mode", "?")
    w = cfg.get("weights") or {}
    if crop == "foreground_bbox" and w.get("silhouette", 0) >= 0.5:
        floors = cfg.get("score_floors") or {}
        return "shape" if any(floors.values()) else "shape-v1 (no floors)"
    if crop == "none" and abs(w.get("lpips", 0) - 0.4) < 1e-6 and abs(w.get("ssim", 0) - 0.3) < 1e-6:
        return "default"
    weights = "/".join(f"{k[:3]}{v:.2f}".rstrip("0").rstrip(".") for k, v in w.items())
    return f"{crop} {weights}"


def summarize_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = json.loads((run_dir / "models.json").read_text(encoding="utf-8")) if (run_dir / "models.json").is_file() else {}
    sf = models.get("sketchfab") or {}
    gen = models.get("generation")

    def side(name: str) -> str:
        entry = models.get(name) or {}
        if not entry.get("model"):
            return "-"
        path = Path(entry["model"])
        info = sf.get(name) if isinstance(sf, dict) else None
        generation = gen if (name == "candidate" and gen) else None
        return describe_model(path, info, generation)

    orient = (models.get("candidate") or {}).get("auto_orient")
    if orient:
        orientation = f"auto up={orient['up']} front={orient['front']} yaw={orient.get('yaw', 0):+.0f}"
    else:
        c = models.get("candidate") or {}
        orientation = f"manual up={c.get('up', '?')} front={c.get('front', '?')}"
    pairs = metrics.get("pairs") or {}
    per_view = {name.rsplit(".", 1)[0]: p.get("pair_score") for name, p in pairs.items()}
    credits = (gen or {}).get("meta", {}).get("consumed_credit") if gen else None
    row = {
        "run": run_dir.name,
        "reference": side("reference"),
        "candidate": side("candidate"),
        "generation": (
            "-".join(x for x in (gen.get("provider"), gen.get("mode"),
                                 re.sub(r"-\\d{8}$", "", str(gen.get("meta", {}).get("model_version_used") or gen.get("meta", {}).get("model_version") or ""))) if x)
            if gen else "-"
        ),
        "credits": credits if credits is not None else "-",
        "config": _config_name(metrics.get("configuration") or {}),
        "orientation": orientation,
        **{v: per_view.get(v) for v in ("front", "back", "side", "left", "top", "bottom")},
        "overall": metrics.get("overall_score"),
    }
    return row


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v).replace("|", "/")


def write_tables(rows: list[dict], output_root: Path) -> tuple[Path, Path]:
    md = output_root / "results.md"
    csv_path = output_root / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in COLUMNS})
    try:
        shown = output_root.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        shown = output_root
    lines = ["# Results", "", f"{len(rows)} run(s) under `{shown}`. Scores are 0-100 (higher = more similar).", "",
             "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[k]) for k in COLUMNS) + " |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md, csv_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    ap.add_argument("--sort", choices=["time", "score"], default="time")
    args = ap.parse_args()
    rows = [r for r in (summarize_run(d) for d in sorted(args.output_root.glob("run_*")) if d.is_dir()) if r]
    if args.sort == "score":
        rows.sort(key=lambda r: (r["overall"] is None, -(r["overall"] or 0)))
    md, csv_path = write_tables(rows, args.output_root)
    for line in md.read_text(encoding="utf-8").splitlines()[4:]:
        print(line)
    print(f"\nWrote {md} and {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
