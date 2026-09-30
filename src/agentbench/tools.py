"""Tools the agent can call: this project's built-in tools and MCP server tools.

Built-in tools run on the machine of the runner and are confined to the
workspace; evaluation reference files (``reference_model`` / ``reference_views``)
are never readable through them, even if they sit inside the workspace.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Awaitable, Callable

from PIL import Image

from .config import BenchConfig, McpConfig
from .llm import ToolSpec

logger = logging.getLogger(__name__)

MAX_IMAGE_SIDE = 1568  # larger images are downscaled before they go to the model
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


class ToolError(RuntimeError):
    """A tool call failed; the message is returned to the model as an error result."""


async def in_daemon_thread(fn: Callable[..., Any], *args: Any) -> Any:
    """Run a blocking call in a daemon thread.

    Unlike ``asyncio.to_thread`` a call that hangs (a stalled HTTP request) does
    not keep the program alive: when ``asyncio.wait_for`` gives up on it, the
    loop can finish and the process exits without waiting for the thread.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    def deliver(result: Any, error: BaseException | None) -> None:
        if fut.done():  # cancelled by a timeout
            return
        if error is not None:
            fut.set_exception(error)
        else:
            fut.set_result(result)

    def target() -> None:
        try:
            result, error = fn(*args), None
        except BaseException as exc:  # noqa: BLE001 - handed to the awaiting coroutine
            result, error = None, exc
        try:
            loop.call_soon_threadsafe(deliver, result, error)
        except RuntimeError:  # loop already closed: nobody is waiting any more
            pass

    threading.Thread(target=target, daemon=True, name=f"agentbench-{getattr(fn, '__name__', 'call')}").start()
    return await fut


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_block_from_bytes(data: bytes, media_type: str = "image/png") -> dict[str, Any]:
    """Normalise an image for the model: downscale, keep PNG/JPEG."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return {"type": "image", "media_type": media_type, "data": base64.b64encode(data).decode("ascii")}
    changed = False
    if max(img.size) > MAX_IMAGE_SIDE:
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        changed = True
    if media_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        changed = True
    if changed:
        buf = io.BytesIO()
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGBA")
        img.save(buf, format="PNG")
        data, media_type = buf.getvalue(), "image/png"
    return {"type": "image", "media_type": media_type, "data": base64.b64encode(data).decode("ascii")}


def image_block(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}.get(suffix, "image/png")
    return image_block_from_bytes(path.read_bytes(), media)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + f"\n... [{len(text) - limit} characters truncated] ...\n" + text[-limit // 2:]


@dataclass
class TaskContext:
    """What the tools need to know about the task being run."""

    cfg: BenchConfig
    task_id: str
    task_dir: Path  # <output_root>/<run>/<task_id>, inside the workspace
    output_model: Path  # where the agent must save its model (host path)
    hidden: set[Path] = field(default_factory=set)  # evaluation references, never readable
    generate_calls: int = 0
    generated: list[dict[str, Any]] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.cfg.workspace_root

    def path(self, rel: str | None, must_exist: bool = False) -> Path:
        if not rel:
            rel = "."
        p = Path(os.path.expanduser(str(rel)))
        p = (p if p.is_absolute() else self.root / p).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise ToolError(f"path {rel!r} is outside the workspace ({self.root})") from None
        for h in self.hidden:
            if p == h or h in p.parents:
                raise ToolError(f"path {rel!r} is not accessible to the agent")
        if must_exist and not p.exists():
            raise ToolError(f"no such file or directory: {rel}")
        return p

    def rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(p)


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------
@dataclass
class BuiltinTool:
    spec: ToolSpec
    fn: Callable[[TaskContext, dict[str, Any]], list[dict[str, Any]]]


def _list_files(ctx: TaskContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    base = ctx.path(args.get("path"), must_exist=True)
    limit = int(args.get("max_entries", 300))
    lines: list[str] = []
    entries = [base] if base.is_file() else sorted(base.rglob("*"))
    for p in entries:
        try:
            ctx.path(str(p))
        except ToolError:
            continue
        if p.is_dir():
            continue
        lines.append(f"{ctx.rel(p)}  ({p.stat().st_size} bytes)")
        if len(lines) >= limit:
            lines.append(f"... (stopped at {limit} entries)")
            break
    return [text_block("\n".join(lines) or "(empty)")]


def _read_file(ctx: TaskContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    p = ctx.path(args.get("path"), must_exist=True)
    if p.is_dir():
        raise ToolError("path is a directory; use list_files")
    if p.suffix.lower() in IMAGE_SUFFIXES:
        return _read_image(ctx, args)
    data = p.read_bytes()[: ctx.cfg.limits.max_tool_output_chars * 4]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"{args.get('path')} is a binary file ({p.stat().st_size} bytes)") from None
    return [text_block(truncate(text, ctx.cfg.limits.max_tool_output_chars))]


def _write_file(ctx: TaskContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    p = ctx.path(args.get("path"))
    content = args.get("content")
    if not isinstance(content, str):
        raise ToolError("content must be a string")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return [text_block(f"wrote {len(content)} characters to {ctx.rel(p)}")]


def _read_image(ctx: TaskContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    p = ctx.path(args.get("path"), must_exist=True)
    if p.suffix.lower() not in IMAGE_SUFFIXES:
        raise ToolError(f"not an image file: {args.get('path')}")
    with Image.open(p) as im:
        size = im.size
    return [text_block(f"{ctx.rel(p)} ({size[0]}x{size[1]})"), image_block(p)]


def _run_python(ctx: TaskContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code must be a non-empty string")
    timeout = min(float(args.get("timeout_seconds") or ctx.cfg.limits.tool_timeout_seconds), ctx.cfg.limits.tool_timeout_seconds)
    scripts = ctx.task_dir / "scripts"  # every script the agent ran is kept for the record
    scripts.mkdir(parents=True, exist_ok=True)
    script = scripts / f"run_{len(list(scripts.glob('run_*.py'))) + 1:03d}.py"
    script.write_text(code, encoding="utf-8")
    env = {**os.environ, **ctx.cfg.runtime.env,
           "WORKSPACE": str(ctx.root), "OUTPUT_DIR": str(ctx.task_dir), "OUTPUT_MODEL": str(ctx.output_model)}
    try:
        proc = subprocess.run(
            [ctx.cfg.runtime.python or sys.executable, str(script)],
            cwd=ctx.root, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        raise ToolError(f"timed out after {timeout:.0f}s\n{truncate(out, 4000)}") from None
    limit = ctx.cfg.limits.max_tool_output_chars
    text = f"exit code {proc.returncode}  (script saved as {ctx.rel(script)})\n"
    if proc.stdout:
        text += "--- stdout ---\n" + truncate(proc.stdout, limit // 2) + "\n"
    if proc.stderr:
        text += "--- stderr ---\n" + truncate(proc.stderr, limit // 2) + "\n"
    if proc.returncode != 0:
        raise ToolError(text)
    return [text_block(text)]


def _render_views(ctx: TaskContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    from ..render import RenderError, RenderOptions, load_mesh, parse_views, render_view

    p = ctx.path(args.get("path") or ctx.rel(ctx.output_model), must_exist=True)
    try:
        views = parse_views(args.get("views") or "front,side,top,iso")
        opts = RenderOptions(size=int(args.get("size") or 384), views=views, up=args.get("up") or "+y",
                             front=args.get("front") or ("-y" if (args.get("up") or "+y") == "+z" else "+z"))
        opts.validate()
        mesh = load_mesh(p, opts.up, opts.front)
    except RenderError as exc:
        raise ToolError(str(exc)) from None
    out_dir = ctx.task_dir / "tool_renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = [text_block(f"{ctx.rel(p)}: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices, "
                         f"extents {tuple(round(e, 4) for e in mesh.original_extents)} (x, y, z); "
                         f"views rendered with +Y up, front view looking at the model's +Z side")]
    for v in views:
        rgba = render_view(mesh, v, opts)
        img = Image.fromarray(rgba, mode="RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        buf = io.BytesIO()
        bg.convert("RGB").save(buf, format="PNG")
        (out_dir / f"{p.stem}_{v}.png").write_bytes(buf.getvalue())
        blocks += [text_block(f"view: {v}"), image_block_from_bytes(buf.getvalue())]
    return blocks


def _generate_3d(ctx: TaskContext, args: dict[str, Any]) -> list[dict[str, Any]]:
    from ..generate import GenerationError, GenerationRequest, get_provider, tripo_balance

    g = ctx.cfg.runtime.generate
    if ctx.generate_calls >= g.max_calls:
        raise ToolError(f"generate_3d may be called at most {g.max_calls} time(s) per task (credit limit)")
    prompt, image = args.get("prompt"), args.get("image")
    if image and not g.allow_image:
        raise ToolError("image-to-3D is disabled for this benchmark; pass a text prompt instead")
    req = GenerationRequest(
        image=ctx.path(image, must_exist=True) if image else None,
        prompt=prompt if not image else None,
        texture=g.texture, model_version=g.model_version, options=dict(g.options),
        timeout=g.timeout_seconds,
    )
    try:
        req.validate()
        provider = get_provider(g.provider, g.api_key or None)
        ctx.generate_calls += 1
        before = tripo_balance(provider.api_key) if provider.name == "tripo" else None
        model = provider.generate(req, ctx.task_dir / "generated")
        if before is not None:
            after = tripo_balance(provider.api_key)
            if after is not None:
                model.meta["credits_used"], model.meta["credits_left"] = before - after, after
    except GenerationError as exc:
        raise ToolError(str(exc)) from None
    ctx.output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model.path, ctx.output_model)
    info = model.to_dict()
    ctx.generated.append(info)
    return [text_block(f"generated {ctx.rel(model.path)} in {model.elapsed_seconds:.0f}s and copied it to "
                       f"{ctx.rel(ctx.output_model)}\n{json.dumps(info.get('meta', {}), ensure_ascii=False)[:1500]}")]


def _schema(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


BUILTINS: dict[str, BuiltinTool] = {
    "list_files": BuiltinTool(ToolSpec(
        "list_files", "List files under a folder of the workspace (recursive). Paths are relative to the workspace root.",
        _schema({"path": {"type": "string", "description": "Folder or file, relative to the workspace (default '.')"},
                 "max_entries": {"type": "integer", "description": "Stop after this many files (default 300)"}}),
    ), _list_files),
    "read_file": BuiltinTool(ToolSpec(
        "read_file", "Read a UTF-8 text file from the workspace (images are returned as images).",
        _schema({"path": {"type": "string"}}, ["path"]),
    ), _read_file),
    "write_file": BuiltinTool(ToolSpec(
        "write_file", "Create or overwrite a UTF-8 text file in the workspace.",
        _schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    ), _write_file),
    "read_image": BuiltinTool(ToolSpec(
        "read_image", "Look at an image file in the workspace (e.g. a reference image).",
        _schema({"path": {"type": "string", "description": "Image path relative to the workspace"}}, ["path"]),
    ), _read_image),
    "run_python": BuiltinTool(ToolSpec(
        "run_python",
        "Run a Python script (working directory = workspace root). numpy, trimesh, PIL and scipy are available. "
        "Environment variables: OUTPUT_MODEL (where the final .glb must be written), OUTPUT_DIR, WORKSPACE. "
        "Returns stdout/stderr; a non-zero exit code is reported as an error.",
        _schema({"code": {"type": "string", "description": "Complete Python source"},
                 "timeout_seconds": {"type": "number"}}, ["code"]),
    ), _run_python),
    "render_views": BuiltinTool(ToolSpec(
        "render_views",
        "Render orthographic views of a mesh file (glb/obj/stl/...) with the benchmark's renderer, to check a model. "
        "Default: the task's output model, views front,side,top,iso.",
        _schema({"path": {"type": "string", "description": "Mesh path relative to the workspace (default: output model)"},
                 "views": {"type": "string", "description": "Comma-separated: front,back,side,left,top,bottom,iso"},
                 "up": {"type": "string", "description": "Up axis of the file, +y (glTF default) or +z"},
                 "size": {"type": "integer"}}),
    ), _render_views),
    "generate_3d": BuiltinTool(ToolSpec(
        "generate_3d",
        "Generate a 3D model with an AI text-to-3D service from a detailed prompt (paid; limited number of calls). "
        "The result is saved as the task's output model automatically.",
        _schema({"prompt": {"type": "string", "description": "Detailed description of the object's parts, proportions and style"},
                 "image": {"type": "string", "description": "Image path for image-to-3D (only if enabled)"}}),
    ), _generate_3d),
}


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------
class McpConnection:
    """One MCP client session (Streamable HTTP, SSE or stdio)."""

    def __init__(self, cfg: McpConfig) -> None:
        self.cfg = cfg
        self.stack = AsyncExitStack()
        self.session = None
        self.tools: list[ToolSpec] = []
        self.all_tools: list[str] = []
        self.server_info: dict[str, Any] = {}

    async def __aenter__(self) -> "McpConnection":
        try:
            from mcp import ClientSession
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ToolError('runtime.type "mcp" needs the MCP client library: pip install "mcp>=1.9,<2"') from exc
        c = self.cfg
        try:
            if c.transport == "stdio":
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                params = StdioServerParameters(command=c.command, args=list(c.args), env={**os.environ, **c.env})
                read, write = await self.stack.enter_async_context(stdio_client(params))
            elif c.transport == "sse":
                from mcp.client.sse import sse_client

                read, write = await self.stack.enter_async_context(sse_client(c.url, headers=c.headers or None))
            else:
                from mcp.client.streamable_http import streamablehttp_client

                read, write, _ = await self.stack.enter_async_context(
                    streamablehttp_client(c.url, headers=c.headers or None))
            self.session = await self.stack.enter_async_context(ClientSession(read, write))
            init = await self.session.initialize()
            info = getattr(init, "serverInfo", None)
            self.server_info = {"name": getattr(info, "name", None), "version": getattr(info, "version", None),
                                "instructions": getattr(init, "instructions", None)}
            listed = await self.session.list_tools()
        except BaseException:
            await self.stack.aclose()
            raise
        self.all_tools = [t.name for t in listed.tools]
        self.tools = [ToolSpec(t.name, t.description or "", t.inputSchema or {}) for t in listed.tools
                      if self.allowed(t.name)]
        return self

    def allowed(self, name: str) -> bool:
        if self.cfg.include_tools is not None and not any(fnmatch(name, p) for p in self.cfg.include_tools):
            return False
        return not any(fnmatch(name, p) for p in self.cfg.exclude_tools)

    async def __aexit__(self, *exc) -> None:
        await self.stack.aclose()

    async def call(self, name: str, args: dict[str, Any], timeout: float) -> tuple[list[dict[str, Any]], bool]:
        res = await self.session.call_tool(name, args, read_timeout_seconds=timedelta(seconds=timeout))
        blocks: list[dict[str, Any]] = []
        for c in res.content:
            kind = getattr(c, "type", "")
            if kind == "text":
                blocks.append(text_block(c.text))
            elif kind == "image":
                blocks.append(image_block_from_bytes(base64.b64decode(c.data), c.mimeType or "image/png"))
            elif kind == "resource":
                r = c.resource
                if getattr(r, "text", None) is not None:
                    blocks.append(text_block(f"[resource {r.uri}]\n{r.text}"))
                else:
                    blocks.append(text_block(f"[binary resource {r.uri}, {getattr(r, 'mimeType', '')}]"))
            else:
                blocks.append(text_block(json.dumps(c.model_dump(mode="json"), ensure_ascii=False)[:4000]))
        if getattr(res, "structuredContent", None) and not blocks:
            blocks.append(text_block(json.dumps(res.structuredContent, ensure_ascii=False)))
        return blocks, bool(res.isError)


# ---------------------------------------------------------------------------
# Toolbox: what one task can call
# ---------------------------------------------------------------------------
class Toolbox:
    def __init__(self, ctx: TaskContext, builtin_names: list[str], mcp: McpConnection | None = None) -> None:
        self.ctx = ctx
        self.mcp = mcp
        self.builtins = {n: BUILTINS[n] for n in builtin_names}
        mcp_names = {t.name for t in (mcp.tools if mcp else [])}
        clash = mcp_names & set(self.builtins)
        if clash:
            logger.warning("MCP tools %s shadow built-in tools of the same name; using the MCP versions", sorted(clash))
            for n in clash:
                self.builtins.pop(n)

    @property
    def specs(self) -> list[ToolSpec]:
        return [b.spec for b in self.builtins.values()] + (list(self.mcp.tools) if self.mcp else [])

    def source(self, name: str) -> str:
        return "builtin" if name in self.builtins else "mcp"

    async def call(self, name: str, args: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        timeout = self.ctx.cfg.limits.tool_timeout_seconds
        try:
            if name in self.builtins:
                fn = self.builtins[name].fn
                # generate_3d polls a remote service for minutes; give it its own limit
                limit = self.ctx.cfg.runtime.generate.timeout_seconds + 60 if name == "generate_3d" else timeout + 5
                return await asyncio.wait_for(in_daemon_thread(fn, self.ctx, args), limit), False
            if self.mcp and any(t.name == name for t in self.mcp.tools):
                return await self.mcp.call(name, args, self.mcp.cfg.timeout_seconds)
            return [text_block(f"unknown tool {name!r}; available: {', '.join(s.name for s in self.specs)}")], True
        except ToolError as exc:
            return [text_block(str(exc))], True
        except asyncio.TimeoutError:
            return [text_block(f"tool {name} timed out")], True
        except Exception as exc:  # report every failure to the model instead of aborting the run
            logger.debug("tool %s failed", name, exc_info=True)
            return [text_block(f"{type(exc).__name__}: {exc}")], True
