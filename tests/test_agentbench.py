"""Tests for the config-driven agent runner (src/agentbench) with scripted models.

No real model API is called: the OpenAI-compatible client gets a fake
``urlopen`` and the Anthropic client a fake SDK object. The MCP test starts a
tiny FastMCP server over stdio (skipped when the ``mcp`` package is missing).
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from src.agentbench.config import AgentConfigError, load_bench_config
from src.agentbench.llm import AnthropicClient, OpenAIChatClient, OpenAIResponsesClient, ToolResult, ToolSpec
from src.agentbench.runner import run_benchmark
from src.agentbench.tools import TaskContext, ToolError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _mug(path: Path) -> None:
    body = trimesh.creation.annulus(r_min=0.4, r_max=0.5, height=1.0, sections=32)
    handle = trimesh.creation.torus(major_radius=0.25, minor_radius=0.05, major_sections=24, minor_sections=8)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    handle.apply_translation([0.6, 0, 0])
    m = trimesh.util.concatenate([body, handle])
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    m.export(path)


@pytest.fixture
def bench(tmp_path: Path):
    """A benchmark folder with a workspace, a hidden answer and a config writer."""
    ws = tmp_path / "ws"
    (ws / "refs").mkdir(parents=True)
    Image.new("RGB", (64, 48), (200, 30, 30)).save(ws / "refs" / "front.png")
    Image.new("RGB", (64, 48), (30, 200, 30)).save(ws / "refs" / "side.png")
    _mug(tmp_path / "answers" / "mug.glb")

    def write(runtime: dict, llm: dict | None = None, **extra) -> Path:
        cfg = {
            "name": "t",
            "runtime": runtime,
            "workspace": {"root": "ws", "output_dir": "runs"},
            "evaluation": {"size": 96, "views": "front,side,top", "auto_orient": False},
            "tasks": [{"id": "mug", "prompt": "Make a mug.", "reference_images": ["refs/*.png"],
                       "reference_model": "answers/mug.glb"}],
            **extra,
        }
        if llm is not None:
            cfg["llm"] = llm
        p = tmp_path / "bench.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return p

    return tmp_path, write


OPENAI_LLM = {"api_format": "openai", "base_url": "http://fake/v1", "api_key": "${TEST_AGENT_KEY}", "model": "m"}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class ScriptedOpenAI:
    """Fake /chat/completions: returns the scripted assistant messages in order."""

    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.requests: list[dict] = []

    def __call__(self, req, timeout=None):
        assert req.full_url == "http://fake/v1/chat/completions"
        assert req.get_header("Authorization") == "Bearer sk-test"
        self.requests.append(json.loads(req.data.decode()))
        msg = self.script.pop(0)
        body = {"model": "m", "choices": [{"message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        return _Resp(json.dumps(body).encode())


def call(i: int, name: str, args: dict) -> dict:
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


BUILD_MUG = textwrap.dedent("""
    import os, numpy as np, trimesh
    body = trimesh.creation.annulus(r_min=0.4, r_max=0.5, height=1.0, sections=32)
    body.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    body.export(os.environ["OUTPUT_MODEL"])
    print("saved", os.environ["OUTPUT_MODEL"])
""")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_env_paths_and_overrides(bench, monkeypatch):
    root, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "sk-test")
    cfg = load_bench_config(write({"type": "builtin"}, OPENAI_LLM), ["llm.model=other", "limits.max_turns=5"])
    assert cfg.llm.api_key == "sk-test" and cfg.llm.model == "other" and cfg.limits.max_turns == 5
    assert cfg.workspace_root == (root / "ws").resolve()
    assert cfg.output_root == (root / "ws" / "runs").resolve()
    assert cfg.runtime.tools == ["list_files", "read_file", "write_file", "read_image", "run_python", "render_views"]
    assert cfg.to_dict()["llm"]["api_key"] == "***"


@pytest.mark.parametrize("patch, message", [
    ({"runtime": {"type": "nope"}}, "runtime.type"),
    ({"runtime": {"type": "builtin", "tools": ["rm_rf"]}}, "unknown tool"),
    ({"runtime": {"type": "mcp"}}, "runtime.mcp is required"),
    ({"llm": {**OPENAI_LLM, "temprature": 1}}, "unknown key"),
    ({"tasks": []}, "at least one task"),
    ({"tasks": [{"id": "a b", "prompt": "x"}]}, "task id"),
])
def test_config_errors(bench, patch, message):
    _, write = bench
    base = {"runtime": {"type": "builtin"}, "llm": OPENAI_LLM}
    base.update(patch)
    runtime, llm = base.pop("runtime"), base.pop("llm")
    with pytest.raises(AgentConfigError, match=message):
        load_bench_config(write(runtime, llm, **base))


def test_config_yaml(tmp_path):
    (tmp_path / "w").mkdir()
    p = tmp_path / "b.yaml"
    p.write_text("name: y\nruntime: {type: command, command: 'echo {prompt}'}\nworkspace: {root: w}\n"
                 "tasks:\n  - prompt: hello\n", encoding="utf-8")
    cfg = load_bench_config(p)
    assert cfg.tasks[0].id == "task01" and cfg.llm is None


def test_workspace_sandbox(bench, monkeypatch):
    root, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "k")
    (root / "ws" / "secret").mkdir()
    (root / "ws" / "secret" / "answer.glb").write_bytes(b"x")
    cfg = load_bench_config(write({"type": "builtin"}, OPENAI_LLM))
    ctx = TaskContext(cfg, "mug", root / "ws" / "runs" / "x", root / "ws" / "runs" / "x" / "model.glb",
                      hidden={(root / "ws" / "secret").resolve()})
    assert ctx.path("refs/front.png") == (root / "ws" / "refs" / "front.png").resolve()
    with pytest.raises(ToolError, match="outside the workspace"):
        ctx.path("../answers/mug.glb")
    with pytest.raises(ToolError, match="not accessible"):
        ctx.path("secret/answer.glb")


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
def test_openai_adapter_messages(bench, monkeypatch):
    _, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "sk-test")
    cfg = load_bench_config(write({"type": "builtin"}, {**OPENAI_LLM, "echo_reasoning": True}))
    fake = ScriptedOpenAI([{"content": "", "reasoning_content": "hmm", "tool_calls": [call(1, "read_image", {"path": "a.png"})]}])
    client = OpenAIChatClient(cfg.llm, urlopen=fake)
    client.start("sys", [{"type": "text", "text": "hi"}])
    turn = client.complete([ToolSpec("read_image", "d", {"properties": {"path": {"type": "string"}}})])
    assert turn.reasoning == "hmm" and turn.tool_calls[0].arguments == {"path": "a.png"}
    body = fake.requests[0]
    assert body["tools"][0]["function"]["parameters"]["type"] == "object"
    assert body["max_tokens"] == 16000
    img = {"type": "image", "media_type": "image/png", "data": "AAAA"}
    client.add_tool_results([ToolResult("c1", "read_image", [{"type": "text", "text": "ok"}, img])])
    assert client.messages[2]["reasoning_content"] == "hmm"  # echoed
    assert client.messages[3] == {"role": "tool", "tool_call_id": "c1",
                                  "content": "ok\n[1 image(s) returned; shown in the next message]"}
    assert client.messages[4]["role"] == "user"
    assert client.messages[4]["content"][-1]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_openai_switches_to_max_completion_tokens(bench, monkeypatch):
    import urllib.error

    _, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "sk-test")
    cfg = load_bench_config(write({"type": "builtin"}, OPENAI_LLM))
    ok = ScriptedOpenAI([{"content": "hi"}, {"content": "again"}])
    bodies = []

    def fake(req, timeout=None):
        body = json.loads(req.data.decode())
        bodies.append(body)
        if "max_tokens" in body:
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(
                b'{"error": {"message": "Unsupported parameter: \'max_tokens\' is not supported with this model. '
                b'Use \'max_completion_tokens\' instead."}}'))
        return ok(req, timeout)

    client = OpenAIChatClient(cfg.llm, urlopen=fake)
    client.start("sys", [{"type": "text", "text": "hi"}])
    assert client.complete([]).text == "hi"
    assert client.complete([]).text == "again"
    assert [("max_tokens" in b, b.get("max_completion_tokens")) for b in bodies] == [(True, None), (False, 16000), (False, 16000)]


def test_openai_responses_adapter(bench, monkeypatch):
    _, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "sk-test")
    cfg = load_bench_config(write({"type": "builtin"}, {**OPENAI_LLM, "api_format": "openai_responses"}))
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "check the image"}],
                 "encrypted_content": "ENC"}
    fn = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_image", "arguments": '{"path": "a.png"}'}
    replies = [
        {"output": [reasoning, fn], "status": "completed", "usage": {"input_tokens": 5, "output_tokens": 3}},
        {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}],
         "status": "completed"},
    ]
    bodies = []

    def fake(req, timeout=None):
        assert req.full_url == "http://fake/v1/responses"
        bodies.append(json.loads(req.data.decode()))
        return _Resp(json.dumps(replies.pop(0)).encode())

    client = OpenAIResponsesClient(cfg.llm, urlopen=fake)
    client.start("sys", [{"type": "text", "text": "hi"}])
    turn = client.complete([ToolSpec("read_image", "d", {})])
    assert turn.reasoning == "check the image" and turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].arguments == {"path": "a.png"} and turn.stop_reason == "tool_calls"
    assert bodies[0]["instructions"] == "sys" and bodies[0]["store"] is False and bodies[0]["max_output_tokens"] == 16000
    assert bodies[0]["tools"][0] == {"type": "function", "name": "read_image", "description": "d",
                                     "parameters": {"type": "object", "properties": {}}}
    img = {"type": "image", "media_type": "image/png", "data": "AAAA"}
    client.add_tool_results([ToolResult("call_1", "read_image", [{"type": "text", "text": "ok"}, img])])
    assert client.complete([]).text == "done"
    items = bodies[1]["input"]
    assert items[1] == reasoning and items[2] == fn  # passed back unchanged
    assert items[3]["type"] == "function_call_output" and items[3]["call_id"] == "call_1"
    assert items[4]["content"][-1] == {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}


def test_anthropic_adapter_echoes_thinking():
    anthropic = pytest.importorskip("anthropic")
    from anthropic.types import Message

    final = Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "content": [{"type": "thinking", "thinking": "plan the mug", "signature": "sig=="},
                    {"type": "text", "text": "Looking."},
                    {"type": "tool_use", "id": "tu_1", "name": "read_image", "input": {"path": "a.png"}}],
        "stop_reason": "tool_use", "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    seen = {}

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return final

    class Messages:
        def stream(self, **kwargs):
            seen.update(kwargs)
            return Stream()

    class Fake:
        messages = Messages()

    from src.agentbench.config import LLMConfig

    cfg = LLMConfig(api_format="anthropic", model="claude-opus-5")
    cfg.validate()
    client = AnthropicClient(cfg, client=Fake())
    client.start("sys", [{"type": "text", "text": "hi"}])
    turn = client.complete([ToolSpec("read_image", "d", {})])
    assert seen["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert seen["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    assert turn.reasoning == "plan the mug" and turn.text == "Looking." and turn.tool_calls[0].id == "tu_1"
    assert client.messages[1]["content"][0] == {"type": "thinking", "thinking": "plan the mug", "signature": "sig=="}
    client.add_tool_results([ToolResult("tu_1", "read_image",
                                        [{"type": "image", "media_type": "image/png", "data": "AAAA"}], False)])
    res = client.messages[2]["content"][0]
    assert res["type"] == "tool_result" and res["tool_use_id"] == "tu_1"
    assert res["content"][0]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_builtin_run_with_feedback(bench, monkeypatch):
    root, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "sk-test")
    cfg = load_bench_config(write({"type": "builtin"}, OPENAI_LLM),
                            ["evaluation.feedback_rounds=1"])
    fake = ScriptedOpenAI([
        {"content": "Let me look.", "reasoning_content": "first the references",
         "tool_calls": [call(1, "read_image", {"path": "refs/front.png"}),
                        call(2, "read_file", {"path": "../answers/mug.glb"})]},
        {"content": "", "tool_calls": [call(3, "run_python", {"code": BUILD_MUG})]},
        {"content": "", "tool_calls": [call(4, "render_views", {"views": "front,top"})]},
        {"content": "Built a cylindrical mug body."},
        {"content": "Kept the model."},  # after feedback
    ])
    run_dir, outcomes = asyncio.run(run_benchmark(cfg, llm_factory=lambda c: OpenAIChatClient(c, urlopen=fake)))
    out = outcomes[0]
    assert out.status == "completed", out.error
    assert out.turns == 5 and out.tool_errors == 1  # reading the hidden answer failed
    assert out.tool_calls == {"read_image": 1, "read_file": 1, "run_python": 1, "render_views": 1}
    assert out.input_tokens == 500 and out.output_tokens == 50
    assert out.score is not None and 0 < out.score < 100
    assert len(out.evaluations) == 1  # model unchanged after feedback -> not re-scored

    # the model saw the image as a user message after the tool result
    second = fake.requests[1]["messages"]
    assert second[-1]["role"] == "user" and second[-1]["content"][-1]["type"] == "image_url"
    assert "outside the workspace" in second[-2]["content"]
    # feedback message with the score and the report image
    fb = fake.requests[4]["messages"][-1]
    assert "overall score" in fb["content"][0]["text"] and fb["content"][-1]["type"] == "image_url"

    task_dir = run_dir / "mug"
    for name in ("model.glb", "prompt.md", "transcript.jsonl", "conversation.md", "task.json",
                 "eval/metrics.json", "eval/report.png", "scripts/run_001.py"):
        assert (task_dir / name).is_file(), name
    events = [json.loads(line) for line in (task_dir / "transcript.jsonl").read_text().splitlines()]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "task_start" and kinds[-1] == "task_end"
    assert {"system_prompt", "user", "assistant", "tool_result", "evaluation", "feedback"} <= set(kinds)
    assert events[kinds.index("assistant")]["reasoning"] == "first the references"
    assert any(e["type"] == "tool_result" and e["content"][-1].get("path", "").startswith("images/") for e in events)
    md = (task_dir / "conversation.md").read_text()
    assert "Thinking" in md and "first the references" in md and "```python" in md
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["tasks"][0]["status"] == "completed" and summary["mean_score"] == pytest.approx(out.score)
    assert json.loads((run_dir / "config.json").read_text())["llm"]["api_key"] == "***"


def test_llm_error_is_recorded(bench, monkeypatch):
    import urllib.error

    _, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "sk-test")
    cfg = load_bench_config(write({"type": "builtin"}, OPENAI_LLM))

    def failing(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b'{"error":"bad key"}'))

    run_dir, outcomes = asyncio.run(run_benchmark(cfg, llm_factory=lambda c: OpenAIChatClient(c, urlopen=failing)))
    assert outcomes[0].status == "llm_error" and "401" in outcomes[0].error
    assert "llm_error" in (run_dir / "summary.md").read_text()


def test_max_turns(bench, monkeypatch):
    _, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "sk-test")
    cfg = load_bench_config(write({"type": "builtin"}, OPENAI_LLM), ["limits.max_turns=2"])
    fake = ScriptedOpenAI([{"content": "", "tool_calls": [call(i, "list_files", {})]} for i in range(3)])
    _, outcomes = asyncio.run(run_benchmark(cfg, llm_factory=lambda c: OpenAIChatClient(c, urlopen=fake)))
    assert outcomes[0].status == "max_turns" and outcomes[0].turns == 2


def test_command_runtime(bench):
    _, write = bench
    script = ("import sys, trimesh; trimesh.creation.box().export(sys.argv[1]); "
              "print('{\"type\": \"event\"}'); print('done', sys.argv[2][:5])")
    cfg = load_bench_config(write({"type": "command", "command": [sys.executable, "-c", script, "{output_model}", "{prompt}"]}))
    run_dir, outcomes = asyncio.run(run_benchmark(cfg))
    out = outcomes[0]
    assert out.status == "completed" and out.score is not None and out.final_message == "done Make "
    events = [json.loads(line) for line in (run_dir / "mug" / "transcript.jsonl").read_text().splitlines()]
    assert any(e["type"] == "command_stdout" and e["json"] == {"type": "event"} for e in events)
    assert "Save the final model" in (run_dir / "mug" / "prompt.md").read_text()


MCP_SERVER = textwrap.dedent("""
    import sys, trimesh
    from mcp.server.fastmcp import FastMCP, Image
    from PIL import Image as PILImage
    import io

    mcp = FastMCP("fake-modeler")

    @mcp.tool()
    def make_box(path: str, size: float = 1.0) -> str:
        \"\"\"Create a box and save it as glb at path.\"\"\"
        trimesh.creation.box(extents=[size, size, size]).export(path)
        return f"saved {path}"

    @mcp.tool()
    def screenshot() -> Image:
        \"\"\"Grab a screenshot.\"\"\"
        buf = io.BytesIO()
        PILImage.new("RGB", (32, 32), (0, 0, 255)).save(buf, format="PNG")
        return Image(data=buf.getvalue(), format="png")

    @mcp.tool()
    def download_asset(uid: str) -> str:
        \"\"\"Hidden by exclude_tools.\"\"\"
        return "nope"

    mcp.run()
""")


def test_mcp_stdio_runtime(bench, monkeypatch):
    pytest.importorskip("mcp")
    root, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "sk-test")
    server = root / "server.py"
    server.write_text(MCP_SERVER, encoding="utf-8")
    cfg = load_bench_config(write(
        {"type": "mcp", "tools": ["read_image"],
         "mcp": {"transport": "stdio", "command": sys.executable, "args": [str(server)], "exclude_tools": ["download_*"]}},
        OPENAI_LLM))
    out_path = {}

    class Script(ScriptedOpenAI):
        def __call__(self, req, timeout=None):
            body = json.loads(req.data.decode())
            if not self.requests:
                names = sorted(t["function"]["name"] for t in body["tools"])
                assert names == ["make_box", "read_image", "screenshot"]
                system = body["messages"][0]["content"]
                path = system.split("ONE .glb file at: ")[1].split("\n")[0].strip()
                out_path["p"] = path
                self.script = [
                    {"content": "", "tool_calls": [call(1, "make_box", {"path": path, "size": 0.8}),
                                                   call(2, "screenshot", {})]},
                    {"content": "done"},
                ]
            return super().__call__(req, timeout)

    fake = Script([])
    run_dir, outcomes = asyncio.run(run_benchmark(cfg, llm_factory=lambda c: OpenAIChatClient(c, urlopen=fake)))
    out = outcomes[0]
    assert out.status == "completed", out.error
    assert Path(out_path["p"]) == run_dir / "mug" / "model.glb"
    assert out.tool_calls == {"make_box": 1, "screenshot": 1} and out.score is not None
    # the MCP screenshot was saved and passed to the model
    assert list((run_dir / "mug" / "images").glob("*screenshot.png"))
    assert fake.requests[1]["messages"][-1]["content"][-1]["type"] == "image_url"


def test_hung_model_request_is_abandoned_and_retried(bench, monkeypatch):
    import threading
    import time

    _, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "sk-test")
    cfg = load_bench_config(write({"type": "builtin"}, {**OPENAI_LLM, "timeout_seconds": 0.5, "max_retries": 1}),
                            ["evaluation.enabled=false"])
    release = threading.Event()
    calls = []

    def fake(req, timeout=None):
        calls.append(time.monotonic())
        if len(calls) == 1:  # the first request hangs, then answers late
            release.wait(10)
            msg = {"content": "late", "tool_calls": [call(9, "list_files", {})]}
        else:
            msg = {"content": "on time"}
        body = {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {}}
        return _Resp(json.dumps(body).encode())

    clients = []

    def factory(c):
        clients.append(OpenAIChatClient(c, urlopen=fake))
        return clients[0]

    t0 = time.monotonic()
    run_dir, outcomes = asyncio.run(run_benchmark(cfg, llm_factory=factory))
    assert time.monotonic() - t0 < 5  # did not wait for the hung request
    out = outcomes[0]
    assert out.final_message == "on time" and out.status == "no_model" and out.turns == 1
    release.set()  # the late answer arrives now and must be discarded
    time.sleep(0.2)
    assert [m["content"] for m in clients[0].messages if m["role"] == "assistant"] == ["on time"]
    kinds = [json.loads(l)["type"] for l in (run_dir / "mug" / "transcript.jsonl").read_text().splitlines()]
    assert "llm_timeout" in kinds


def test_model_that_never_answers_ends_as_llm_error(bench, monkeypatch):
    import threading

    _, write = bench
    monkeypatch.setenv("TEST_AGENT_KEY", "sk-test")
    cfg = load_bench_config(write({"type": "builtin"}, {**OPENAI_LLM, "timeout_seconds": 0.3, "max_retries": 1}),
                            ["evaluation.enabled=false"])
    never = threading.Event()

    def hang(req, timeout=None):
        never.wait(5)
        raise TimeoutError("released")

    _, outcomes = asyncio.run(run_benchmark(cfg, llm_factory=lambda c: OpenAIChatClient(c, urlopen=hang)))
    assert outcomes[0].status == "llm_error" and "2 attempts" in outcomes[0].error
    never.set()  # let the abandoned threads finish before pytest closes its log capture
    for t in threading.enumerate():
        if t.name.startswith("agentbench-"):
            t.join(5)
