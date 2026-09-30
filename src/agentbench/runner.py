"""Run every task of a benchmark config headlessly and record what happened.

Per run::

    <workspace>/<output_dir>/<name>_<YYYYMMDD_HHMMSS>/
    ├── config.json            resolved config (API keys redacted)
    ├── summary.json / .md     one row per task: status, score, turns, tokens, time
    └── <task_id>/
        ├── model.glb          the agent's model (the file that is scored)
        ├── prompt.md          system prompt + first user message as sent
        ├── transcript.jsonl   every event: model turns (text, reasoning, tool calls,
        │                      token usage, latency), tool results, evaluations, errors
        ├── conversation.md    the same, readable: thinking, tool calls, results, images
        ├── images/            every image the model saw (tool screenshots, renders)
        ├── scripts/           code the agent ran with run_python
        ├── eval/, eval_2/ ... scoring output of each evaluation (metrics.json, report.png)
        └── task.json          summary of the task
"""

from __future__ import annotations

import asyncio
import base64
import glob
import json
import logging
import os
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import BenchConfig, TaskConfig
from .llm import AssistantTurn, LLMClient, LLMError, ToolResult, make_client
from .tools import McpConnection, TaskContext, Toolbox, image_block, in_daemon_thread, text_block, truncate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
class Recorder:
    """Append-only transcript (JSONL + Markdown); images are saved as files."""

    def __init__(self, task_dir: Path) -> None:
        self.dir = task_dir
        self.images = task_dir / "images"
        self.t0 = time.monotonic()
        self.n_images = 0
        self._jsonl = open(task_dir / "transcript.jsonl", "a", encoding="utf-8")
        self._md = open(task_dir / "conversation.md", "a", encoding="utf-8")

    def close(self) -> None:
        self._jsonl.close()
        self._md.close()

    def event(self, kind: str, **data: Any) -> None:
        rec = {"ts": datetime.now().isoformat(timespec="milliseconds"),
               "elapsed": round(time.monotonic() - self.t0, 3), "type": kind, **data}
        self._jsonl.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._jsonl.flush()

    def md(self, text: str) -> None:
        self._md.write(text.rstrip() + "\n\n")
        self._md.flush()

    def save_images(self, blocks: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        """Replace image data by saved file paths (for the transcript)."""
        out = []
        for b in blocks:
            if b.get("type") == "image":
                self.images.mkdir(exist_ok=True)
                self.n_images += 1
                ext = {"image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(b["media_type"], ".png")
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:40]
                path = self.images / f"{self.n_images:03d}_{safe}{ext}"
                path.write_bytes(base64.b64decode(b["data"]))
                out.append({"type": "image", "path": path.relative_to(self.dir).as_posix(), "media_type": b["media_type"]})
            else:
                out.append(b)
        return out

    def md_blocks(self, blocks: list[dict[str, Any]], limit: int = 4000) -> str:
        parts = []
        for b in blocks:
            if b["type"] == "image":
                parts.append(f"![{b['path']}]({b['path']})")
            elif b.get("text"):
                parts.append("```text\n" + truncate(b["text"], limit).replace("```", "ʼʼʼ") + "\n```")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompts (built in; the config only supplies the task prompt)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a 3D modeling agent in an automated benchmark. You run headless: nobody will answer questions, \
so make reasonable decisions yourself and keep working until the model is finished.

Goal: build a 3D model that matches the task description and the reference images as closely as possible \
(overall shape, proportions, parts and their placement). The model is scored automatically: six orthographic \
views of your model (front, back, both sides, top, bottom) are rendered and their silhouettes and edges are \
compared with those of a hidden reference model. Colours, materials and textures are not scored; a face count \
far away from the reference costs a few points. Use +Y as up and make the front of the object face +Z \
(Blender: +Z up and -Y front; its glTF exporter converts this automatically).

Workspace: {workspace}
Save the final model as ONE .glb file at: {output_model}
Overwrite that file whenever you improve the model; only that file is scored.

{runtime_notes}
When the model is saved and you are done, reply with a short summary of what you built and do not call a tool."""


def runtime_notes(cfg: BenchConfig, tools: Toolbox | None, mcp: McpConnection | None, ctx: TaskContext) -> str:
    notes: list[str] = []
    names = {s.name for s in tools.specs} if tools else set()
    if cfg.runtime.type == "builtin":
        notes.append("Build the model with run_python, e.g. with trimesh (creation.box / cylinder / icosphere / "
                     "revolve / extrude_polygon, transformations, concatenate) and export it with "
                     "mesh.export(os.environ['OUTPUT_MODEL']).")
    if mcp is not None:
        server = mcp.server_info.get("name") or "the MCP server"
        notes.append(f"Your modeling tools come from {server}. Files that the MCP server reads or writes must use "
                     f"the server's paths: the workspace is {mcp_path(cfg, ctx.root, ctx)} there, so save the model to "
                     f"{mcp_path(cfg, ctx.output_model, ctx)}. The helper tools "
                     f"({', '.join(sorted(n for n in names if tools.source(n) == 'builtin')) or 'none'}) run on "
                     f"the benchmark host and take paths relative to the workspace.")
        if "execute_blender_code" in names:
            notes.append("The server controls a headless Blender. First clear the default scene with "
                         "bpy.ops.wm.read_factory_settings(use_empty=True). Export with "
                         "bpy.ops.export_scene.gltf(filepath=<output path>, export_format='GLB'). "
                         "get_viewport_screenshot renders the current scene so you can check your work.")
        disabled = [n for n in mcp.all_tools if not mcp.allowed(n)]
        if disabled:
            notes.append(f"{len(disabled)} of the server's tools are disabled for this benchmark "
                         f"({', '.join(disabled[:8])}{' ...' if len(disabled) > 8 else ''}); ignore any mention of "
                         "them below and work with the tools you have.")
        if mcp.server_info.get("instructions"):
            notes.append("Server instructions:\n" + str(mcp.server_info["instructions"]).strip())
    if "render_views" in names:
        notes.append("render_views renders your saved model the same way the scorer does; use it to compare "
                     "with the reference images before you finish.")
    if "generate_3d" in names:
        g = cfg.runtime.generate
        notes.append(f"generate_3d creates a model with an AI text-to-3D service ({g.provider}); it is paid and "
                     f"limited to {g.max_calls} call(s), so write one detailed prompt (parts, proportions, style).")
    return "\n\n".join(notes) + ("\n" if notes else "")


def mcp_path(cfg: BenchConfig, host_path: Path, ctx: TaskContext) -> str:
    """How the MCP server sees a host path inside the workspace."""
    m = cfg.runtime.mcp
    if not m or not m.workspace_path:
        return str(host_path)
    rel = host_path.resolve().relative_to(ctx.root)
    return m.workspace_path.rstrip("/") + "/" + rel.as_posix() if rel.parts else m.workspace_path


def reference_images(cfg: BenchConfig, task: TaskConfig) -> list[Path]:
    found: list[Path] = []
    for pattern in task.reference_images:
        matches = sorted(glob.glob(str(cfg.workspace_root / pattern), recursive=True))
        if not matches:
            raise FileNotFoundError(f"task {task.id}: reference image pattern {pattern!r} matched nothing "
                                    f"under {cfg.workspace_root}")
        found.extend(Path(m).resolve() for m in matches if Path(m).is_file())
    return list(dict.fromkeys(found))


def first_message(task: TaskConfig, images: list[Path], ctx: TaskContext, attach: bool) -> list[dict[str, Any]]:
    text = task.prompt.strip()
    if images:
        how = "they are attached below" if attach else "look at them with read_image"
        text += (f"\n\nReference images ({len(images)}, paths relative to the workspace; {how}):\n"
                 + "\n".join(f"- {ctx.rel(p)}" for p in images))
    blocks = [text_block(text)]
    if attach:
        for p in images:
            blocks += [text_block(ctx.rel(p)), image_block(p)]
    return blocks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(cfg: BenchConfig, task: TaskConfig, model: Path, out_dir: Path) -> dict[str, Any]:
    """Score ``model`` against the task's reference; returns a compact summary."""
    from ..cli import SHAPE_CONFIG, run_model_comparison
    from ..benchmark import BenchmarkRunner
    from ..config import load_config
    from ..render import RenderOptions, load_mesh, parse_views, render_views

    ev = cfg.evaluation
    bench_cfg = load_config(cfg.resolve(ev.benchmark_config) if ev.benchmark_config else SHAPE_CONFIG)
    if ev.mesh_weight is not None:
        bench_cfg.mesh_complexity.weight = float(ev.mesh_weight)
    if ev.rig_weight is not None:
        bench_cfg.rig.weight = float(ev.rig_weight)
    bench_cfg.validate()
    front = ev.front or {"+y": "+z", "-y": "-z", "+z": "-y", "-z": "+y", "+x": "+z", "-x": "+z"}[ev.up]
    opts = RenderOptions(size=int(ev.size), views=parse_views(ev.views), up=ev.up, front=front)
    opts.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    if task.reference_model:
        result = run_model_comparison(bench_cfg, opts, cfg.resolve(task.reference_model), model, out_dir,
                                      ev.auto_orient, None, None)
    else:
        ref_dir = cfg.resolve(task.reference_views)
        views = tuple(p.stem.lower() for p in sorted(ref_dir.iterdir()) if p.stem.lower() in
                      ("front", "back", "side", "left", "top", "bottom", "iso"))
        if not views:
            raise FileNotFoundError(f"{ref_dir} contains no <view>.png images (front, side, top ...)")
        opts.views = views
        render_views(load_mesh(model, opts.up, opts.front), out_dir / "renders" / "candidate", opts)
        bench_cfg.input.skip_unmatched = True
        result = BenchmarkRunner(bench_cfg).run(ref_dir, out_dir / "renders" / "candidate", None, run_dir=out_dir)
    summary: dict[str, Any] = {
        "overall_score": result.overall_score,
        "shape_score": result.shape_score,
        "views": {p.name.rsplit(".", 1)[0]: p.pair_score for p in result.pairs},
        "failed_views": {p.name: p.error for p in result.pairs if p.error},
        "eval_dir": out_dir.name,
    }
    if result.mesh:
        summary["faces"] = {"candidate": result.mesh["candidate"]["faces"], "reference": result.mesh["reference"]["faces"],
                            "mesh_score": result.mesh["score"]}
    if result.rig and result.rig.get("applicable"):
        summary["rig_score"] = result.rig["score"]
    report = out_dir / "report.png"
    summary["report"] = report.name if report.is_file() else None
    return summary


def feedback_message(ev: dict[str, Any], round_no: int, rounds: int, ctx: TaskContext, out_dir: Path,
                     images: bool) -> list[dict[str, Any]]:
    views = ", ".join(f"{k} {v:.1f}" if v is not None else f"{k} n/a" for k, v in ev["views"].items())
    text = (f"Automatic evaluation of your current model ({ctx.rel(ctx.output_model)}): "
            f"overall score {ev['overall_score']:.1f} / 100")
    if ev.get("shape_score") is not None:
        text += f" (shape {ev['shape_score']:.1f})"
    text += f".\nPer view: {views}."
    if ev.get("faces"):
        text += f"\nFaces: yours {ev['faces']['candidate']:,} vs reference {ev['faces']['reference']:,}."
    text += (f"\n\nThis is feedback round {round_no} of {rounds}. Improve the model where the views score low, "
             f"save it to the same path, then reply with a short summary.")
    blocks = [text_block(text)]
    report = out_dir / "report.png"
    if images and report.is_file():
        blocks += [text_block("Evaluation report (reference | your model | difference, one row per view):"),
                   image_block(report)]
    return blocks


def describe_error(exc: BaseException) -> str:
    """``Type: message`` of the first leaf exception (MCP clients raise ExceptionGroups)."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------
@dataclass
class TaskOutcome:
    id: str
    status: str = "pending"
    score: float | None = None
    shape_score: float | None = None
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    model_path: str | None = None
    final_message: str = ""
    error: str | None = None
    generated: list[dict[str, Any]] = field(default_factory=list)

    def add_usage(self, usage: dict[str, Any]) -> None:
        self.input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        self.input_tokens += int(usage.get("cache_read_input_tokens") or 0) + int(usage.get("cache_creation_input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)


class TaskRunner:
    def __init__(self, cfg: BenchConfig, task: TaskConfig, run_dir: Path, llm_factory=make_client) -> None:
        self.cfg, self.task = cfg, task
        self.task_dir = run_dir / task.id
        self.task_dir.mkdir(parents=True, exist_ok=True)
        hidden = set()
        for ref in (task.reference_model, task.reference_views):
            if ref:
                hidden.add(cfg.resolve(ref))
        self.ctx = TaskContext(cfg, task.id, self.task_dir, self.task_dir / task.output_model, hidden)
        self.rec = Recorder(self.task_dir)
        self.out = TaskOutcome(task.id)
        self.llm_factory = llm_factory
        self.deadline = time.monotonic() + cfg.limits.timeout_seconds
        self._evaluated_mtime: float | None = None

    # -- helpers ----------------------------------------------------------------
    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    async def evaluate(self) -> dict[str, Any] | None:
        """Score the current model file (once per version of the file)."""
        model = self.ctx.output_model
        if not model.is_file():
            return None
        mtime = model.stat().st_mtime
        if mtime == self._evaluated_mtime:
            return self.out.evaluations[-1] if self.out.evaluations else None
        self._evaluated_mtime = mtime
        can_score = self.cfg.evaluation.enabled and (self.task.reference_model or self.task.reference_views)
        if not can_score:
            return None
        n = len(self.out.evaluations) + 1
        out_dir = self.task_dir / ("eval" if n == 1 else f"eval_{n}")
        t0 = time.monotonic()
        try:
            ev = await asyncio.to_thread(evaluate_model, self.cfg, self.task, model, out_dir)
        except Exception as exc:
            logger.exception("[%s] evaluation failed", self.task.id)
            ev = {"overall_score": None, "error": f"{type(exc).__name__}: {exc}", "eval_dir": out_dir.name}
        ev["seconds"] = round(time.monotonic() - t0, 1)
        self.out.evaluations.append(ev)
        self.out.score, self.out.shape_score = ev.get("overall_score"), ev.get("shape_score")
        self.rec.event("evaluation", **ev)
        if ev.get("overall_score") is not None:
            views = ", ".join(f"{k} {v:.1f}" for k, v in ev["views"].items() if v is not None)
            self.rec.md(f"## Evaluation {n}\n\n**Score {ev['overall_score']:.2f}** (shape {ev.get('shape_score') or 0:.2f}); "
                        f"views: {views}" + (f"\n\n![report]({out_dir.name}/report.png)" if ev.get("report") else ""))
            logger.info("[%s] evaluation %d: score %.2f", self.task.id, n, ev["overall_score"])
        else:
            self.rec.md(f"## Evaluation {n}\n\nFailed: {ev.get('error')}")
        return ev

    def _write_prompt(self, system: str, user: list[dict[str, Any]]) -> None:
        text = "# System prompt\n\n" + system + "\n\n# First user message\n\n" + "\n\n".join(
            b["text"] if b["type"] == "text" else "[image]" for b in user)
        (self.task_dir / "prompt.md").write_text(text, encoding="utf-8")

    # -- entry point --------------------------------------------------------------
    async def run(self) -> TaskOutcome:
        t0 = time.monotonic()
        self.rec.md(f"# Task `{self.task.id}`\n\nRuntime: `{self.cfg.runtime.type}`"
                    + (f", model `{self.cfg.llm.model}` ({self.cfg.llm.api_format})" if self.cfg.llm else ""))
        self.rec.event("task_start", task=asdict(self.task), runtime=self.cfg.runtime.type,
                       model=self.cfg.llm.model if self.cfg.llm else None)
        try:
            images = reference_images(self.cfg, self.task)
            if self.cfg.runtime.type == "command":
                await self._run_command(images)
            elif self.cfg.runtime.type == "mcp":
                async with McpConnection(self.cfg.runtime.mcp) as conn:
                    self.rec.event("mcp_connected", server=conn.server_info, tools=[t.name for t in conn.tools])
                    self.rec.md(f"MCP server: {conn.server_info.get('name')} {conn.server_info.get('version') or ''}, "
                                f"{len(conn.tools)} tools: " + ", ".join(f"`{t.name}`" for t in conn.tools))
                    await self._run_agent(images, Toolbox(self.ctx, self.cfg.runtime.tools, conn), conn)
            else:
                await self._run_agent(images, Toolbox(self.ctx, self.cfg.runtime.tools), None)
        except Exception as exc:
            logger.exception("[%s] task failed", self.task.id)
            self.out.status, self.out.error = "error", describe_error(exc)
            self.rec.event("error", error=self.out.error)
            self.rec.md(f"**Error:** {self.out.error}")
        except BaseException as exc:  # Ctrl-C / cancellation: keep what was recorded
            self.out.status, self.out.error = "interrupted", type(exc).__name__
            raise
        finally:
            if self.out.status not in ("interrupted",):
                await self.evaluate()  # score whatever model exists at the end
            if self.ctx.output_model.is_file():
                self.out.model_path = self.ctx.rel(self.ctx.output_model)
                if self.out.status in ("completed", "pending"):
                    self.out.status = "completed"
            elif self.out.status in ("completed", "pending"):
                self.out.status = "no_model"
            self.out.generated = self.ctx.generated
            self.out.elapsed_seconds = round(time.monotonic() - t0, 1)
            self.rec.event("task_end", **asdict(self.out))
            (self.task_dir / "task.json").write_text(json.dumps(asdict(self.out), indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
            self.rec.md(f"## Result\n\nStatus **{self.out.status}**, score "
                        f"{'n/a' if self.out.score is None else f'{self.out.score:.2f}'}, {self.out.turns} turns, "
                        f"{sum(self.out.tool_calls.values())} tool calls, {self.out.elapsed_seconds:.0f}s")
            self.rec.close()
        return self.out

    # -- LLM agent loop ----------------------------------------------------------------
    async def _run_agent(self, images: list[Path], tools: Toolbox, mcp: McpConnection | None) -> None:
        cfg = self.cfg
        llm: LLMClient = self.llm_factory(cfg.llm)
        tool_names = {s.name for s in tools.specs}
        workspace = str(self.ctx.root)
        if mcp is not None:
            output_model = mcp_path(cfg, self.ctx.output_model, self.ctx)
            if cfg.runtime.mcp.workspace_path:
                workspace = f"{workspace} on the benchmark host, {cfg.runtime.mcp.workspace_path} for the MCP server"
        else:
            output_model = self.ctx.rel(self.ctx.output_model) + " (relative to the workspace)"
            if "run_python" in tool_names:
                output_model += "; inside run_python it is also os.environ['OUTPUT_MODEL']"
        system = SYSTEM_PROMPT.format(workspace=workspace, output_model=output_model,
                                      runtime_notes=runtime_notes(cfg, tools, mcp, self.ctx))
        attach = self.task.attach_images or "read_image" not in tool_names
        user = first_message(self.task, images, self.ctx, attach)
        self._write_prompt(system, user)
        saved_user = self.rec.save_images(user, "reference")
        self.rec.event("system_prompt", text=system, tools=[asdict(s) for s in tools.specs])
        self.rec.event("user", content=saved_user)
        self.rec.md("## System prompt\n\n```text\n" + system + "\n```")
        self.rec.md("## User\n\n" + self.rec.md_blocks(saved_user))
        llm.start(system, user)

        feedback_left = cfg.evaluation.feedback_rounds
        while True:
            if self.out.turns >= cfg.limits.max_turns:
                self.out.status = "max_turns"
                self.rec.event("stop", reason="max_turns")
                return
            if self.remaining() <= 0:
                self.out.status = "timeout"
                self.rec.event("stop", reason="timeout")
                return
            self.out.turns += 1
            t_call = time.monotonic()
            try:
                turn: AssistantTurn | None = await self._call_llm(llm, tools)
            except LLMError as exc:
                self.out.status, self.out.error = "llm_error", str(exc)
                self.rec.event("llm_error", error=str(exc), turn=self.out.turns)
                self.rec.md(f"**Model API error:** {exc}")
                return
            if turn is None:  # the task's time ran out while waiting for the model
                self.out.status = "timeout"
                self.rec.event("stop", reason="timeout during model call")
                return
            self.out.add_usage(turn.usage)
            self._record_turn(turn, time.monotonic() - t_call)
            if turn.refusal:
                self.out.status = "refused"
                self.out.final_message = turn.text
                return

            if not turn.tool_calls:
                self.out.final_message = turn.text
                if feedback_left > 0 and self.ctx.output_model.is_file():
                    ev = await self.evaluate()
                    if ev and ev.get("overall_score") is not None:
                        round_no = cfg.evaluation.feedback_rounds - feedback_left + 1
                        feedback_left -= 1
                        msg = feedback_message(ev, round_no, cfg.evaluation.feedback_rounds, self.ctx,
                                               self.task_dir / ev["eval_dir"], cfg.evaluation.feedback_images)
                        saved = self.rec.save_images(msg, "feedback")
                        self.rec.event("feedback", round=round_no, content=saved)
                        self.rec.md(f"## Feedback {round_no}\n\n" + self.rec.md_blocks(saved))
                        llm.add_user(msg)
                        continue
                self.out.status = "completed"
                return

            results: list[ToolResult] = []
            for call in turn.tool_calls:
                self.out.tool_calls[call.name] = self.out.tool_calls.get(call.name, 0) + 1
                t_tool = time.monotonic()
                if call.parse_error:
                    blocks, is_error = [text_block(f"invalid tool arguments ({call.parse_error}); send valid JSON")], True
                else:
                    blocks, is_error = await tools.call(call.name, call.arguments)
                if not blocks:
                    blocks = [text_block("(no output)")]
                if is_error:
                    self.out.tool_errors += 1
                results.append(ToolResult(call.id, call.name, blocks, is_error))
                saved = self.rec.save_images(blocks, call.name)
                self.rec.event("tool_result", call_id=call.id, name=call.name, source=tools.source(call.name),
                               is_error=is_error, seconds=round(time.monotonic() - t_tool, 3), content=saved)
                self.rec.md(f"**Result of `{call.name}`**" + (" (error)" if is_error else "")
                            + f" — {time.monotonic() - t_tool:.1f}s\n\n" + self.rec.md_blocks(saved))
            llm.add_tool_results(results)

    async def _call_llm(self, llm: LLMClient, tools: Toolbox) -> AssistantTurn | None:
        """One model call with a deadline of ``llm.timeout_seconds`` per request.

        A request that gets no answer in time is abandoned and sent again (up to
        ``llm.max_retries`` times). Returns None when the task's own time runs out.
        """
        per_request = self.cfg.llm.timeout_seconds
        for attempt in range(self.cfg.llm.max_retries + 1):
            limit = min(per_request, self.remaining())
            if limit <= 0:
                return None
            try:
                return await asyncio.wait_for(in_daemon_thread(llm.complete, tools.specs), limit)
            except asyncio.TimeoutError:
                llm.abandon()
                if self.remaining() <= 0:
                    return None
                self.rec.event("llm_timeout", turn=self.out.turns, attempt=attempt + 1, seconds=round(limit, 1))
                self.rec.md(f"*Model gave no answer within {limit:.0f}s (attempt {attempt + 1}); sending the request again.*")
                logger.warning("[%s] model gave no answer within %.0fs; retrying", self.task.id, limit)
        raise LLMError(f"no answer from the model within {per_request:.0f}s, "
                       f"{self.cfg.llm.max_retries + 1} attempts")

    def _record_turn(self, turn: AssistantTurn, seconds: float) -> None:
        self.rec.event("assistant", turn=self.out.turns, text=turn.text, reasoning=turn.reasoning,
                       tool_calls=[asdict(c) for c in turn.tool_calls], stop_reason=turn.stop_reason,
                       usage=turn.usage, refusal=turn.refusal, model=turn.model, seconds=round(seconds, 3))
        md = [f"## Turn {self.out.turns} — assistant ({seconds:.1f}s, stop: {turn.stop_reason})"]
        if turn.reasoning:
            md.append("<details open><summary>Thinking</summary>\n\n" + turn.reasoning.strip() + "\n\n</details>")
        if turn.text.strip():
            md.append(turn.text.strip())
        for c in turn.tool_calls:
            args = json.dumps(c.arguments, ensure_ascii=False, indent=2)
            code = c.arguments.get("code") if isinstance(c.arguments.get("code"), str) else None
            if code:
                rest = {k: v for k, v in c.arguments.items() if k != "code"}
                md.append(f"**Tool call `{c.name}`** {json.dumps(rest, ensure_ascii=False) if rest else ''}\n\n```python\n{code}\n```")
            else:
                md.append(f"**Tool call `{c.name}`**\n\n```json\n{args}\n```")
        if turn.refusal:
            md.append(f"**Refused:** {turn.refusal}")
        self.rec.md("\n\n".join(md))
        logger.info("[%s] turn %d: %d tool call(s)%s", self.task.id, self.out.turns, len(turn.tool_calls),
                    "" if turn.tool_calls else " - finished")

    # -- external command --------------------------------------------------------------
    def _command_prompt(self, images: list[Path], extra: str = "") -> str:
        text = first_message(self.task, images, self.ctx, attach=False)[0]["text"].replace(
            "look at them with read_image", "open the image files")
        return (f"{text}\n\nRequirements (automated, headless benchmark; nobody will answer questions):\n"
                f"- Save the final model as ONE .glb file at: {self.ctx.output_model}\n"
                f"- Use +Y as up and make the front of the object face +Z.\n"
                f"- The model is scored by comparing six orthographic silhouettes with a hidden reference model.\n"
                + (f"\n{extra}\n" if extra else ""))

    def _command_env(self) -> dict[str, str]:
        env = {**os.environ, **self.cfg.runtime.env,
               "WORKSPACE": str(self.ctx.root), "OUTPUT_DIR": str(self.task_dir),
               "OUTPUT_MODEL": str(self.ctx.output_model), "TASK_ID": self.task.id}
        llm = self.cfg.llm
        if llm is not None:
            env.update({"LLM_API_FORMAT": llm.api_format, "LLM_MODEL": llm.model,
                        "LLM_BASE_URL": llm.base_url or "", "LLM_API_KEY": llm.api_key or ""})
            if llm.api_format == "anthropic":
                env.update({k: v for k, v in (("ANTHROPIC_API_KEY", llm.api_key), ("ANTHROPIC_BASE_URL", llm.base_url),
                                              ("ANTHROPIC_MODEL", llm.model)) if v})
            else:
                env.update({k: v for k, v in (("OPENAI_API_KEY", llm.api_key), ("OPENAI_BASE_URL", llm.base_url),
                                              ("OPENAI_MODEL", llm.model)) if v})
        return env

    async def _run_command(self, images: list[Path]) -> None:
        rounds = self.cfg.evaluation.feedback_rounds
        extra = ""
        for attempt in range(rounds + 1):
            prompt = self._command_prompt(images, extra)
            prompt_file = self.task_dir / ("prompt.md" if attempt == 0 else f"prompt_{attempt + 1}.md")
            prompt_file.write_text(prompt, encoding="utf-8")
            values = {
                "prompt": prompt, "prompt_file": str(prompt_file), "workspace": str(self.ctx.root),
                "output_dir": str(self.task_dir), "output_model": str(self.ctx.output_model),
                "task_id": self.task.id, "images": " ".join(str(p) for p in images),
                "model": self.cfg.llm.model if self.cfg.llm else "",
                "base_url": (self.cfg.llm.base_url or "") if self.cfg.llm else "",
            }
            cmd = self.cfg.runtime.command
            self.rec.event("user", content=[text_block(prompt)], attempt=attempt + 1)
            self.rec.md(f"## Prompt (attempt {attempt + 1})\n\n```text\n{prompt}\n```")
            code = await self._exec(cmd, values, attempt + 1)
            self.out.turns += 1
            if code is None:
                self.out.status = "timeout"
                return
            if code != 0:
                self.out.status, self.out.error = "command_failed", f"exit code {code}"
                return
            if attempt < rounds:
                ev = await self.evaluate()
                if not ev or ev.get("overall_score") is None:
                    break
                fb = feedback_message(ev, attempt + 1, rounds, self.ctx, self.task_dir / ev["eval_dir"], False)
                extra = fb[0]["text"] + f"\nThe full evaluation report is {self.task_dir / ev['eval_dir'] / 'report.png'}."
        self.out.status = "completed"

    async def _exec(self, cmd: list[str] | str, values: dict[str, str], attempt: int) -> int | None:
        env = self._command_env()
        if isinstance(cmd, str):
            line = fill(cmd, values, quote=True)
            proc = await asyncio.create_subprocess_shell(line, cwd=self.ctx.root, env=env,
                                                         stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            shown = line
        else:
            argv = [fill(a, values) for a in cmd]
            proc = await asyncio.create_subprocess_exec(*argv, cwd=self.ctx.root, env=env,
                                                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            shown = shlex.join(argv)
        shown = shown.replace(values["prompt"], "<prompt>") if values["prompt"] else shown
        self.rec.event("command_start", command=shown, attempt=attempt)
        self.rec.md(f"## Command (attempt {attempt})\n\n```bash\n{shown}\n```")
        suffix = "" if attempt == 1 else f"_{attempt}"
        md_lines: list[str] = []

        async def pump(stream, name: str) -> None:
            with open(self.task_dir / f"{name}{suffix}.log", "w", encoding="utf-8") as log:
                while True:
                    raw = await stream.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", "replace").rstrip("\n")
                    log.write(line + "\n")
                    log.flush()
                    data: Any = None
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                        except ValueError:
                            data = None
                    self.rec.event(f"command_{name}", line=None if data is not None else line, json=data)
                    if name == "stdout" and data is None:
                        md_lines.append(line)

        try:
            await asyncio.wait_for(asyncio.gather(pump(proc.stdout, "stdout"), pump(proc.stderr, "stderr"), proc.wait()),
                                   max(self.remaining(), 1))
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            self.rec.event("stop", reason="timeout", attempt=attempt)
            return None
        self.rec.event("command_end", exit_code=proc.returncode, attempt=attempt)
        tail = "\n".join(md_lines[-200:])
        self.rec.md(f"Exit code {proc.returncode}. Output (last lines; full output in stdout{suffix}.log):\n\n```text\n"
                    f"{truncate(tail, 12000)}\n```")
        if md_lines:
            self.out.final_message = md_lines[-1]
        return proc.returncode


def fill(template: str, values: dict[str, str], quote: bool = False) -> str:
    """Replace ``{name}`` placeholders; other braces (JSON, Python code) stay as they are."""
    pattern = re.compile(r"\{(" + "|".join(map(re.escape, values)) + r")\}")
    return pattern.sub(lambda m: shlex.quote(values[m.group(1)]) if quote else values[m.group(1)], template)


# ---------------------------------------------------------------------------
# Whole run
# ---------------------------------------------------------------------------
def create_run_dir(cfg: BenchConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = cfg.output_root / f"{cfg.name}_{stamp}"
    run_dir, n = base, 1
    while run_dir.exists():
        n += 1
        run_dir = Path(f"{base}_{n}")
    run_dir.mkdir(parents=True)
    return run_dir


def write_summary(cfg: BenchConfig, run_dir: Path, outcomes: list[TaskOutcome], started: datetime) -> None:
    scores = [o.score for o in outcomes if o.score is not None]
    data = {
        "name": cfg.name, "description": cfg.description, "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"), "runtime": cfg.runtime.type,
        "llm": {"api_format": cfg.llm.api_format, "model": cfg.llm.model, "base_url": cfg.llm.base_url} if cfg.llm else None,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "tasks": [asdict(o) for o in outcomes],
    }
    (run_dir / "summary.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    lines = [f"# {cfg.name}", "",
             f"Runtime `{cfg.runtime.type}`" + (f", model `{cfg.llm.model}` ({cfg.llm.api_format})" if cfg.llm else "")
             + f", started {data['started']}", "",
             "| task | status | score | shape | turns | tool calls (errors) | tokens in / out | time |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for o in outcomes:
        lines.append(f"| [{o.id}]({o.id}/conversation.md) | {o.status} | {fmt(o.score)} | {fmt(o.shape_score)} | "
                     f"{o.turns} | {sum(o.tool_calls.values())} ({o.tool_errors}) | "
                     f"{o.input_tokens:,} / {o.output_tokens:,} | {o.elapsed_seconds:.0f}s |")
    lines += ["", f"Mean score: {fmt(data['mean_score'])} over {len(scores)} scored task(s)."]
    for o in outcomes:
        if o.error:
            lines.append(f"\n- `{o.id}`: {o.error}")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_benchmark(cfg: BenchConfig, task_ids: list[str] | None = None, llm_factory=make_client) -> tuple[Path, list[TaskOutcome]]:
    tasks = [t for t in cfg.tasks if not task_ids or t.id in task_ids]
    unknown = sorted(set(task_ids or []) - {t.id for t in cfg.tasks})
    if unknown:
        raise ValueError(f"unknown task id(s): {unknown}")
    run_dir = create_run_dir(cfg)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    started = datetime.now()
    outcomes: list[TaskOutcome] = []
    logger.info("Run directory: %s", run_dir)
    try:
        for t in tasks:
            logger.info("=== task %s ===", t.id)
            outcomes.append(await TaskRunner(cfg, t, run_dir, llm_factory).run())
            write_summary(cfg, run_dir, outcomes, started)
    finally:
        write_summary(cfg, run_dir, outcomes, started)
    return run_dir, outcomes
