"""Command-line interface.

Examples (run from the project root)::

    python -m src.cli compare --reference data/reference --candidate data/candidate --output outputs
    python -m src.cli compare-pair --reference data/reference/front.png --candidate data/candidate/front.png --output outputs
    python -m src.cli render-views --model models/mug.glb --output renders/mug
    python -m src.cli compare-models --reference models/a.glb --candidate https://sketchfab.com/3d-models/xxx-<uid>
    python -m src.cli fetch-sketchfab https://sketchfab.com/3d-models/xxx-<uid> --models-dir models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from .benchmark import BenchmarkResult, BenchmarkRunner, PairingError, compute_overall_score
from .config import ALIGNMENT_METHODS, BenchmarkConfig, ConfigError, CROP_MODES, DEVICES, load_config
from .render import AXIS_NAMES, DEFAULT_VIEWS, RenderError, RenderOptions, STYLES, VIEWS, parse_views, render_views
from .reporting import format_summary_table, save_metrics_csv, save_metrics_json, save_report
from .sketchfab import SketchfabError, download_model, is_sketchfab_reference

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

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

    p_render = sub.add_parser(
        "render-views",
        help="Render orthographic views (front/side/top ...) of a 3D model or Sketchfab URL to PNG",
    )
    p_render.add_argument("--model", required=True, help="Mesh file (glb/gltf/obj/stl/ply ...) or Sketchfab model URL/uid")
    p_render.add_argument("--output", required=True, type=Path, help="Folder that receives <view>.png + views.json")
    _add_render_args(p_render)
    _add_sketchfab_args(p_render)
    p_render.add_argument("--log-level", default="INFO")

    p_models = sub.add_parser(
        "compare-models",
        help="Render two 3D models (files or Sketchfab URLs) with identical settings and score their views",
    )
    p_models.add_argument("--reference", required=True, help="Reference mesh file or Sketchfab model URL/uid")
    p_models.add_argument("--candidate", required=True, help="Candidate mesh file or Sketchfab model URL/uid")
    p_models.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs", help="Output root (default: outputs/)")
    p_models.add_argument("--config", type=Path, default=None, help="YAML config for the image benchmark")
    p_models.add_argument("--crop-mode", choices=CROP_MODES, default=None, help="Override preprocessing.crop_mode")
    p_models.add_argument("--alignment", choices=ALIGNMENT_METHODS, default=None, help="Override preprocessing.alignment")
    p_models.add_argument("--canvas-size", type=int, default=None, help="Override preprocessing.canvas_size")
    p_models.add_argument("--device", choices=DEVICES, default=None, help="Override metrics.lpips.device")
    p_models.add_argument("--no-save", action="store_true", help="Do not write any files; print results only")
    p_models.add_argument("--log-level", default=None, help="Override output.log_level")
    _add_render_args(p_models)
    _add_sketchfab_args(p_models)

    p_fetch = sub.add_parser("fetch-sketchfab", help="Download a Sketchfab model (needs an API token) and print its path")
    p_fetch.add_argument("reference", help="Sketchfab model URL, 'sketchfab:<uid>' or bare uid")
    _add_sketchfab_args(p_fetch)
    p_fetch.add_argument("--force", action="store_true", help="Re-download even if the model is cached")
    p_fetch.add_argument("--log-level", default="INFO")
    return parser


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("rendering (applied identically to both models)")
    g.add_argument(
        "--views",
        default=",".join(DEFAULT_VIEWS),
        help=f"Comma-separated views or 'all' (default: %(default)s; available: {', '.join(VIEWS)})",
    )
    g.add_argument("--size", type=int, default=512, help="Render resolution in pixels (default: %(default)s)")
    g.add_argument("--up", choices=AXIS_NAMES, default="+y", help="Model axis pointing up (glTF: +y, Blender/STL often: +z)")
    g.add_argument(
        "--front",
        choices=AXIS_NAMES,
        default=None,
        help="Model axis pointing towards the viewer of the front view (default: +z for +y-up, -y for +z-up)",
    )
    g.add_argument("--fill", type=float, default=0.85, help="Fraction of the canvas covered by the largest extent")
    g.add_argument("--style", choices=STYLES, default="shaded", help="shaded (headlight, default) or silhouette (flat black)")
    g.add_argument("--supersample", type=int, default=2, help="Anti-aliasing factor (default: %(default)s)")


def _add_sketchfab_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("sketchfab")
    g.add_argument("--token", default=None, help="Sketchfab API token (default: $SKETCHFAB_API_TOKEN)")
    g.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Where downloaded models are cached (default: models/)",
    )


def render_options_from_args(args: argparse.Namespace) -> RenderOptions:
    front = args.front
    if front is None:
        front = {"+y": "+z", "-y": "-z", "+z": "-y", "-z": "+y", "+x": "+z", "-x": "+z"}[args.up]
    opts = RenderOptions(
        size=int(args.size),
        views=parse_views(args.views),
        up=args.up,
        front=front,
        fill=float(args.fill),
        style=args.style,
        supersample=int(args.supersample),
    )
    opts.validate()
    return opts


def resolve_model(reference: str, args: argparse.Namespace) -> tuple[Path, dict | None]:
    """Turn a CLI model argument into a local mesh path.

    Local files are used as-is; Sketchfab URLs / uids are downloaded into
    ``--models-dir``. Returns ``(path, sketchfab metadata or None)``.
    """
    path = Path(reference).expanduser()
    if path.is_file():
        return path, None
    if is_sketchfab_reference(reference):
        mesh_path, info = download_model(reference, args.models_dir, token=args.token)
        return mesh_path, info.to_dict()
    raise RenderError(f"model not found: {reference!r} (not a local file and not a Sketchfab URL/uid)")


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


def cmd_render_views(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)
    opts = render_options_from_args(args)
    model_path, info = resolve_model(args.model, args)
    written = render_views(model_path, args.output, opts)
    if info is not None:
        with open(Path(args.output) / "sketchfab.json", "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
    print()
    for view, path in written.items():
        print(f"{view:>8}: {path}")
    return 0


def cmd_compare_models(args: argparse.Namespace) -> int:
    cfg = load_effective_config(args)
    setup_logging(cfg.output.log_level)
    opts = render_options_from_args(args)

    ref_path, ref_info = resolve_model(args.reference, args)
    cand_path, cand_info = resolve_model(args.candidate, args)

    with tempfile.TemporaryDirectory(prefix="imgsim_render_") as tmp:
        if args.no_save:
            run_dir = None
            render_root = Path(tmp)
        else:
            run_dir = BenchmarkRunner.create_run_dir(args.output)
            render_root = run_dir / "renders"
        ref_dir, cand_dir = render_root / "reference", render_root / "candidate"

        logger.info("Rendering reference model %s", ref_path)
        render_views(ref_path, ref_dir, opts)
        logger.info("Rendering candidate model %s", cand_path)
        render_views(cand_path, cand_dir, opts)

        runner = BenchmarkRunner(cfg)
        result = runner.run(ref_dir, cand_dir, None, run_dir=run_dir)

        if run_dir is not None:
            models_meta = {
                "reference": {"model": str(ref_path), "sketchfab": ref_info},
                "candidate": {"model": str(cand_path), "sketchfab": cand_info},
                "render": opts.to_dict(),
            }
            with open(run_dir / "models.json", "w", encoding="utf-8") as fh:
                json.dump(models_meta, fh, indent=2)

    print()
    print(format_summary_table(result))
    if result.run_dir is not None:
        print(f"\nOutputs written to: {result.run_dir}")
    return 0 if result.overall_score is not None else 1


def cmd_fetch_sketchfab(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)
    path, info = download_model(args.reference, args.models_dir, token=args.token, force=args.force)
    print(json.dumps({"path": str(path), **info.to_dict()}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            return cmd_compare(args)
        if args.command == "compare-pair":
            return cmd_compare_pair(args)
        if args.command == "render-views":
            return cmd_render_views(args)
        if args.command == "compare-models":
            return cmd_compare_models(args)
        if args.command == "fetch-sketchfab":
            return cmd_fetch_sketchfab(args)
        parser.error(f"unknown command {args.command}")
        return 2
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except PairingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    except (RenderError, SketchfabError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
