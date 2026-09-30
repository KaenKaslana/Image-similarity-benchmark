"""Benchmark-run configuration (JSON or YAML) for the headless agent runner.

One file describes a whole benchmark run::

    {
      "name": "mug-blender-mcp",
      "runtime":   {"type": "mcp", "mcp": {"url": "http://localhost:8000/mcp"}},
      "llm":       {"api_format": "openai", "base_url": "https://...", "api_key": "${LLM_API_KEY}", "model": "..."},
      "workspace": {"root": "../../workspaces/example", "output_dir": "runs"},
      "tasks": [
        {"id": "mug", "prompt": "...", "reference_images": ["refs/mug/*.png"],
         "reference_model": "answers/mug.glb"}
      ]
    }

Rules:

* ``${VAR}`` / ``${VAR:-default}`` in any string is replaced by the environment
  variable (a ``.env`` next to the config or in the project root is read first,
  without overriding variables that are already set). Keep API keys out of the file.
* Relative paths are resolved against the directory of the config file, except
  ``workspace.output_dir`` and ``tasks[].reference_images``, which are relative
  to ``workspace.root`` (the agent sees them that way).
* Unknown keys are errors, like the image benchmark's YAML config.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RUNTIME_TYPES = ("mcp", "builtin", "command")
API_FORMATS = ("openai", "openai_responses", "anthropic")
MCP_TRANSPORTS = ("streamable_http", "sse", "stdio")
BUILTIN_TOOLS = ("list_files", "read_file", "write_file", "read_image", "run_python", "render_views", "generate_3d")
DEFAULT_TOOLS = {
    "mcp": ["list_files", "read_image", "render_views"],
    "builtin": ["list_files", "read_file", "write_file", "read_image", "run_python", "render_views"],
    "command": [],
}
_UNSET = object()


class AgentConfigError(ValueError):
    """Raised when the benchmark-run configuration is invalid."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class McpConfig:
    """How to reach the MCP server (network URL or a local stdio command)."""

    url: str | None = None
    transport: str = "streamable_http"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    # How the MCP server sees workspace.root, e.g. "/app/workspaces/example" when
    # the server runs in Docker with ./workspaces mounted at /app/workspaces.
    # Default: the same absolute path as on this machine.
    workspace_path: str | None = None
    timeout_seconds: float = 300.0
    # Tool filters (fnmatch patterns on tool names), e.g. hide asset downloaders
    # so the agent has to model: "exclude_tools": ["*sketchfab*", "*tripo*"]
    include_tools: list[str] | None = None
    exclude_tools: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.transport not in MCP_TRANSPORTS:
            raise AgentConfigError(f"runtime.mcp.transport must be one of {MCP_TRANSPORTS}, got {self.transport!r}")
        if self.transport == "stdio" and not self.command:
            raise AgentConfigError("runtime.mcp.command is required for the stdio transport")
        if self.transport != "stdio" and not self.url:
            raise AgentConfigError(f"runtime.mcp.url is required for the {self.transport} transport")
        if self.timeout_seconds <= 0:
            raise AgentConfigError("runtime.mcp.timeout_seconds must be positive")


@dataclass
class GenerateToolConfig:
    """Settings of the optional ``generate_3d`` tool (paid AI generation service)."""

    provider: str = "tripo"
    api_key: str | None = None
    model_version: str | None = "v3.1-20260211"
    texture: bool = False
    allow_image: bool = False  # text-to-3D only unless enabled (image-to-3D costs more)
    max_calls: int = 1
    options: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 1800.0

    def validate(self) -> None:
        from ..generate import PROVIDERS

        if self.provider not in PROVIDERS:
            raise AgentConfigError(f"runtime.generate.provider must be one of {PROVIDERS}")
        if self.max_calls < 0:
            raise AgentConfigError("runtime.generate.max_calls must be >= 0")


@dataclass
class RuntimeConfig:
    """Which environment the agent works in.

    * ``mcp``: tools come from an MCP server (e.g. the headless Blender container),
      plus the host-side helper tools listed in ``tools``.
    * ``builtin``: only this project's own tools (files, Python with trimesh, renderer).
    * ``command``: run any other agent CLI; the runner only fills in the prompt and
      paths, records its output and scores the model it leaves behind.
    """

    type: str = "mcp"
    tools: list[str] | None = None
    mcp: McpConfig | None = None
    command: list[str] | str | None = None
    env: dict[str, str] = field(default_factory=dict)
    python: str | None = None  # interpreter for run_python (default: this one)
    generate: GenerateToolConfig = field(default_factory=GenerateToolConfig)

    def validate(self) -> None:
        if self.type not in RUNTIME_TYPES:
            raise AgentConfigError(f"runtime.type must be one of {RUNTIME_TYPES}, got {self.type!r}")
        if self.tools is None:
            self.tools = list(DEFAULT_TOOLS[self.type])
        unknown = [t for t in self.tools if t not in BUILTIN_TOOLS]
        if unknown:
            raise AgentConfigError(f"runtime.tools: unknown tool(s) {unknown}; available: {', '.join(BUILTIN_TOOLS)}")
        if self.type == "mcp":
            if self.mcp is None:
                raise AgentConfigError("runtime.mcp is required when runtime.type is 'mcp'")
            self.mcp.validate()
        if self.type == "command":
            if not self.command:
                raise AgentConfigError("runtime.command is required when runtime.type is 'command'")
            if self.tools:
                raise AgentConfigError("runtime.tools does not apply to runtime.type 'command'")
        if self.type == "builtin" and not self.tools:
            raise AgentConfigError("runtime.tools must not be empty for runtime.type 'builtin'")
        if "generate_3d" in self.tools:
            self.generate.validate()


@dataclass
class LLMConfig:
    """The model API. ``openai`` = any OpenAI-compatible /chat/completions endpoint
    (OpenAI, DeepSeek, Qwen/DashScope, Kimi, GLM, OpenRouter, vLLM, Ollama ...);
    ``openai_responses`` = OpenAI's /responses endpoint (reasoning models that do not
    allow tools with reasoning on /chat/completions); ``anthropic`` = the Claude Messages API."""

    api_format: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    model: str = ""
    max_tokens: int | None = 16000
    temperature: float | None = None
    # anthropic only: passed as the `thinking` parameter. Default: adaptive thinking
    # with summarised text so the reasoning can be recorded. null = omit the parameter.
    thinking: Any = _UNSET
    supports_images: bool = True
    # openai only: send reasoning_content back with earlier assistant turns
    # (needed by some thinking models, e.g. DeepSeek/Kimi in thinking mode with tools).
    echo_reasoning: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 600.0  # deadline per request; a request without an answer is sent again
    max_retries: int = 3

    def validate(self) -> None:
        if self.api_format not in API_FORMATS:
            raise AgentConfigError(f"llm.api_format must be one of {API_FORMATS}, got {self.api_format!r}")
        if not self.model:
            raise AgentConfigError("llm.model is required")
        if self.api_format != "anthropic" and not self.base_url:
            raise AgentConfigError(f"llm.base_url is required for api_format {self.api_format!r}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise AgentConfigError("llm.max_tokens must be positive (or null)")
        if self.thinking is _UNSET:
            self.thinking = {"type": "adaptive", "display": "summarized"} if self.api_format == "anthropic" else None

    def to_dict(self, redact: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if redact and d.get("api_key"):
            d["api_key"] = "***"
        return d


@dataclass
class WorkspaceConfig:
    root: str = "."
    output_dir: str = "runs"


@dataclass
class LimitsConfig:
    max_turns: int = 40  # model calls per task (feedback rounds included)
    timeout_seconds: float = 1800.0  # wall clock per task
    tool_timeout_seconds: float = 300.0
    max_tool_output_chars: int = 20000

    def validate(self) -> None:
        if self.max_turns < 1 or self.timeout_seconds <= 0 or self.tool_timeout_seconds <= 0:
            raise AgentConfigError("limits: max_turns >= 1 and positive timeouts are required")


@dataclass
class EvaluationConfig:
    """Scoring of the model the agent leaves at the task's output path."""

    enabled: bool = True
    benchmark_config: str | None = None  # image benchmark YAML; default configs/shape.yaml
    auto_orient: bool = True
    views: str | list[str] = "all"
    up: str = "+y"
    front: str | None = None
    size: int = 512
    mesh_weight: float | None = None
    rig_weight: float | None = None
    # After the agent finishes, send it the score and ask for an improved model.
    feedback_rounds: int = 0
    feedback_images: bool = True

    def validate(self) -> None:
        if self.feedback_rounds < 0:
            raise AgentConfigError("evaluation.feedback_rounds must be >= 0")


@dataclass
class TaskConfig:
    id: str
    prompt: str
    reference_images: list[str] = field(default_factory=list)  # globs, relative to workspace.root
    reference_model: str | None = None  # for scoring; relative to the config file, hidden from the agent
    reference_views: str | None = None  # alternative: folder of <view>.png to score renders against
    output_model: str = "model.glb"  # file name inside the task's output folder
    attach_images: bool = False  # also put the reference images into the first message

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", self.id or ""):
            raise AgentConfigError(f"task id {self.id!r} must be non-empty and use only letters, digits, '_', '-', '.'")
        if not self.prompt or not str(self.prompt).strip():
            raise AgentConfigError(f"task {self.id}: prompt is required")
        if isinstance(self.reference_images, str):
            self.reference_images = [self.reference_images]
        if Path(self.output_model).name != self.output_model:
            raise AgentConfigError(f"task {self.id}: output_model must be a plain file name")


@dataclass
class BenchConfig:
    name: str = "agent-bench"
    description: str = ""
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    llm: LLMConfig | None = None
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tasks: list[TaskConfig] = field(default_factory=list)
    # set by the loader
    config_path: Path | None = None
    base_dir: Path = field(default_factory=Path.cwd)

    # -- paths --------------------------------------------------------------
    def resolve(self, path: str | Path) -> Path:
        p = Path(os.path.expanduser(str(path)))
        return (p if p.is_absolute() else self.base_dir / p).resolve()

    @property
    def workspace_root(self) -> Path:
        return self.resolve(self.workspace.root)

    @property
    def output_root(self) -> Path:
        p = Path(os.path.expanduser(self.workspace.output_dir))
        return (p if p.is_absolute() else self.workspace_root / p).resolve()

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", self.name or ""):
            raise AgentConfigError("name must use only letters, digits, '_', '-', '.'")
        self.runtime.validate()
        if self.runtime.type != "command" and self.llm is None:
            raise AgentConfigError("llm is required unless runtime.type is 'command'")
        if self.llm is not None:
            self.llm.validate()
        self.limits.validate()
        self.evaluation.validate()
        if not self.tasks:
            raise AgentConfigError("tasks must contain at least one task")
        seen: set[str] = set()
        for t in self.tasks:
            t.validate()
            if t.id in seen:
                raise AgentConfigError(f"duplicate task id {t.id!r}")
            seen.add(t.id)
        if not self.workspace_root.is_dir():
            raise AgentConfigError(f"workspace.root does not exist: {self.workspace_root}")
        if self.runtime.type == "mcp" and self.runtime.mcp.workspace_path:
            try:
                self.output_root.relative_to(self.workspace_root)
            except ValueError:
                raise AgentConfigError(
                    "workspace.output_dir must be inside workspace.root when runtime.mcp.workspace_path is set "
                    "(the MCP server can only write where the workspace is mounted)"
                ) from None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "description": self.description,
            "runtime": asdict(self.runtime),
            "llm": self.llm.to_dict() if self.llm else None,
            "workspace": asdict(self.workspace),
            "limits": asdict(self.limits),
            "evaluation": asdict(self.evaluation),
            "tasks": [asdict(t) for t in self.tasks],
            "config_path": str(self.config_path) if self.config_path else None,
            "resolved": {"workspace_root": str(self.workspace_root), "output_root": str(self.output_root)},
        }
        gen = d["runtime"].get("generate") or {}
        if gen.get("api_key"):
            gen["api_key"] = "***"
        return d


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: Path) -> None:
    """Minimal ``.env`` reader: KEY=VALUE lines; existing variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ and value:
            os.environ[key] = value


def expand_env(value: Any, missing: list[str]) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            missing.append(name)
            return ""

        return _ENV_RE.sub(sub, value)
    if isinstance(value, list):
        return [expand_env(v, missing) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v, missing) for k, v in value.items()}
    return value


def _build(cls, data: Any, where: str):
    """Build dataclass ``cls`` from a mapping, rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, Mapping):
        raise AgentConfigError(f"{where} must be an object")
    names = {f.name for f in fields(cls) if f.name not in ("config_path", "base_dir")}
    unknown = sorted(set(data) - names)
    if unknown:
        raise AgentConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(names)}")
    kwargs: dict[str, Any] = {}
    nested = {
        (RuntimeConfig, "mcp"): McpConfig,
        (RuntimeConfig, "generate"): GenerateToolConfig,
        (BenchConfig, "runtime"): RuntimeConfig,
        (BenchConfig, "llm"): LLMConfig,
        (BenchConfig, "workspace"): WorkspaceConfig,
        (BenchConfig, "limits"): LimitsConfig,
        (BenchConfig, "evaluation"): EvaluationConfig,
    }
    for key, value in data.items():
        sub = nested.get((cls, key))
        if sub is not None and value is not None:
            kwargs[key] = _build(sub, value, f"{where}.{key}".lstrip("."))
        elif cls is BenchConfig and key == "tasks":
            if not isinstance(value, list):
                raise AgentConfigError("tasks must be a list")
            tasks = []
            for i, t in enumerate(value):
                if isinstance(t, Mapping) and "id" not in t:
                    t = {"id": f"task{i + 1:02d}", **t}
                tasks.append(_build(TaskConfig, t, f"tasks[{i}]"))
            kwargs[key] = tasks
        else:
            kwargs[key] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise AgentConfigError(f"{where}: {exc}") from exc


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """``llm.model=gpt-x`` style overrides; values are parsed as JSON when possible."""
    for item in overrides or []:
        if "=" not in item:
            raise AgentConfigError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        node = data
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return data


def load_bench_config(path: str | Path, overrides: list[str] | None = None) -> BenchConfig:
    path = Path(path).resolve()
    if not path.is_file():
        raise AgentConfigError(f"config file not found: {path}")
    load_dotenv(path.parent / ".env")
    load_dotenv(PROJECT_ROOT / ".env")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as exc:
        raise AgentConfigError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentConfigError(f"{path}: top level must be an object")
    data.pop("$schema", None)
    data = apply_overrides(data, overrides)
    missing: list[str] = []
    data = expand_env(data, missing)
    cfg = _build(BenchConfig, data, "")
    cfg.config_path = path
    cfg.base_dir = path.parent
    cfg.missing_env = sorted(set(missing))  # type: ignore[attr-defined]
    cfg.validate()
    return cfg
