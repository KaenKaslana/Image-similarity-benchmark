"""Command line: ``python -m src.agentbench {run,check} CONFIG``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import AgentConfigError, load_bench_config
from .runner import TaskContext, describe_error, first_message, reference_images, run_benchmark
from .tools import McpConnection


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.agentbench",
                                description="Run an agent benchmark described by a JSON/YAML config file.")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_ in (("run", "run the tasks and score the results"),
                        ("check", "validate the config, list reference images and (for MCP) the server's tools; "
                                  "no model calls")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("config", help="benchmark config (.json / .yaml)")
        s.add_argument("--task", action="append", default=None, help="only this task id (repeatable)")
        s.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                       help="override a config value, e.g. --set llm.model=deepseek-chat --set limits.max_turns=10")
        s.add_argument("--log-level", default="INFO")
    return p


async def check(cfg, task_ids) -> int:
    ok = True
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False, default=str))
    missing = getattr(cfg, "missing_env", [])
    if missing:
        print(f"\nWARNING: environment variables not set: {', '.join(missing)}")
    if cfg.llm and not cfg.llm.api_key:
        print("WARNING: llm.api_key is empty" + (" (the anthropic SDK then tries ANTHROPIC_API_KEY / ant auth)"
                                                 if cfg.llm.api_format == "anthropic" else ""))
    for t in cfg.tasks:
        if task_ids and t.id not in task_ids:
            continue
        print(f"\n[{t.id}]")
        imgs = []
        try:
            imgs = reference_images(cfg, t)
            print(f"  reference images: {len(imgs)}")
            for p in imgs:
                print(f"    {p.relative_to(cfg.workspace_root)}")
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
            ok = False
        for label, ref in (("reference_model", t.reference_model), ("reference_views", t.reference_views)):
            if ref:
                path = cfg.resolve(ref)
                exists = path.exists()
                ok &= exists
                print(f"  {label}: {path} {'' if exists else '(MISSING)'}")
                try:
                    path.relative_to(cfg.workspace_root)
                    print("    note: inside the workspace; MCP servers that mount the workspace can read it")
                except ValueError:
                    pass
        if not (t.reference_model or t.reference_views):
            print("  no reference_model / reference_views: the result will not be scored")
        ctx = TaskContext(cfg, t.id, cfg.output_root / "<run>" / t.id, cfg.output_root / "<run>" / t.id / t.output_model)
        print("  first message:\n    " + first_message(t, imgs, ctx, t.attach_images)[0]["text"].replace("\n", "\n    "))
    if cfg.runtime.type == "mcp":
        print(f"\nConnecting to MCP server ({cfg.runtime.mcp.transport}) ...")
        try:
            async with McpConnection(cfg.runtime.mcp) as conn:
                hidden = [n for n in conn.all_tools if not conn.allowed(n)]
                print(f"  {conn.server_info.get('name')} {conn.server_info.get('version') or ''}: {len(conn.tools)} tools"
                      + (f" ({len(hidden)} excluded: {', '.join(hidden)})" if hidden else ""))
                for tool in conn.tools:
                    print(f"    {tool.name}: {(tool.description or '').strip().splitlines()[0][:100] if tool.description else ''}")
        except BaseException as exc:  # noqa: BLE001 - report any connection problem
            if isinstance(exc, KeyboardInterrupt):
                raise
            print(f"  ERROR: cannot connect: {describe_error(exc)}")
            ok = False
    print("\nOK" if ok else "\nPROBLEMS FOUND")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S", force=True)
    for noisy in ("httpx", "httpcore", "mcp", "anthropic", "PIL", "matplotlib", "trimesh"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        cfg = load_bench_config(args.config, args.set)
    except AgentConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.command == "check":
        return asyncio.run(check(cfg, args.task))
    missing = getattr(cfg, "missing_env", [])
    if missing:
        logging.warning("environment variables not set: %s", ", ".join(missing))
    try:
        run_dir, outcomes = asyncio.run(run_benchmark(cfg, args.task))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; partial results are kept in the run folder", file=sys.stderr)
        return 130
    print()
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Results: {run_dir}")
    return 0 if all(o.status == "completed" for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
