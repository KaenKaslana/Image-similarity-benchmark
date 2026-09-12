"""Command-line interface.

Examples (run from the project root)::

    python -m src.cli compare --reference data/reference --candidate data/candidate --output outputs
    python -m src.cli compare-pair --reference data/reference/front.png --candidate data/candidate/front.png --output outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .benchmark import BenchmarkResult, BenchmarkRunner, PairingError, compute_overall_score
from .config import ALIGNMENT_METHODS, BenchmarkConfig, ConfigError, CROP_MODES, DEVICES, load_config
from .reporting import format_summary_table, save_metrics_csv, save_metrics_json, save_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

logger = logging.getLogger("src.cli")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference", required=True, type=Path, help="Reference folder (or file for compare-pair)")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate folder (or file for compare-pair)")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs", help="Output root (default: outputs/)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML config (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)} if present, else built-in defaults)",
    )
    parser.add_argument("--crop-mode", choices=CROP_MODES, default=None, help="Override preprocessing.crop_mode")
    parser.add_argument(
        "--alignment",
        choices=ALIGNMENT_METHODS,
        default=None,
        help="Override preprocessing.alignment (translation alignment of the candidate before scoring)",
    )
    parser.add_argument("--canvas-size", type=int, default=None, help="Override preprocessing.canvas_size")
    parser.add_argument("--device", choices=DEVICES, default=None, help="Override metrics.lpips.device")
    parser.add_argument("--no-save", action="store_true", help="Do not write any files; print results only")
    parser.add_argument("--log-level", default=None, help="Override output.log_level (DEBUG/INFO/WARNING/ERROR)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Pairwise image similarity benchmark (SSIM, LPIPS, silhouette IoU, edge similarity).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cmp = sub.add_parser("compare", help="Compare two folders of images paired by file name")
    _add_common_args(p_cmp)
    p_cmp.add_argument(
        "--skip-unmatched", action="store_true", help="Skip images without a partner instead of aborting"
    )

    p_pair = sub.add_parser("compare-pair", help="Compare a single reference/candidate image pair")
    _add_common_args(p_pair)
    return parser


def load_effective_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Load the YAML config and apply CLI overrides."""
    config_path = args.config
    if config_path is None and DEFAULT_CONFIG.is_file():
        config_path = DEFAULT_CONFIG
    cfg = load_config(config_path)
    if args.crop_mode is not None:
        cfg.preprocessing.crop_mode = args.crop_mode
    if args.alignment is not None:
        cfg.preprocessing.alignment = args.alignment
    if args.canvas_size is not None:
        cfg.preprocessing.canvas_size = int(args.canvas_size)
    if args.device is not None:
        cfg.metrics.lpips.device = args.device
    if getattr(args, "skip_unmatched", False):
        cfg.input.skip_unmatched = True
    if args.log_level is not None:
        cfg.output.log_level = args.log_level
    cfg.validate()
    return cfg


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = load_effective_config(args)
    setup_logging(cfg.output.log_level)
    runner = BenchmarkRunner(cfg)
    output_root = None if args.no_save else args.output
    result = runner.run(args.reference, args.candidate, output_root)
    print()
    print(format_summary_table(result))
    if result.run_dir is not None:
        print(f"\nOutputs written to: {result.run_dir}")
    return 0 if result.overall_score is not None else 1


def cmd_compare_pair(args: argparse.Namespace) -> int:
    cfg = load_effective_config(args)
    setup_logging(cfg.output.log_level)
    if not args.reference.is_file():
        raise PairingError(f"Reference image not found: {args.reference}")
    if not args.candidate.is_file():
        raise PairingError(f"Candidate image not found: {args.candidate}")
    runner = BenchmarkRunner(cfg)
    run_dir = None if args.no_save else BenchmarkRunner.create_run_dir(args.output)
    name = args.reference.name
    if args.reference.name.lower() != args.candidate.name.lower():
        logger.warning(
            "File names differ (%s vs %s); comparing anyway because compare-pair was requested explicitly",
            args.reference.name,
            args.candidate.name,
        )
    pair = runner.compare_pair(args.reference, args.candidate, name=name, run_dir=run_dir)
    result = BenchmarkResult(
        pairs=[pair], overall_score=compute_overall_score([pair]), run_dir=run_dir, config=cfg.to_dict()
    )
    if run_dir is not None:
        save_metrics_json(result, run_dir / "metrics.json")
        save_metrics_csv(result, run_dir / "metrics.csv")
        save_report(result, run_dir, pairs_per_page=cfg.output.report_pairs_per_page)
    print()
    print(json.dumps({"pair": pair.to_dict(), "overall_score": result.overall_score}, indent=2))
    if run_dir is not None:
        print(f"\nOutputs written to: {run_dir}")
    return 0 if pair.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            return cmd_compare(args)
        if args.command == "compare-pair":
            return cmd_compare_pair(args)
        parser.error(f"unknown command {args.command}")
        return 2
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except PairingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
