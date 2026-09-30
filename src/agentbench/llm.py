"""Model API adapters with one provider-neutral interface for the agent loop.

Content blocks exchanged with the runner are plain dicts:

    {"type": "text", "text": "..."}
    {"type": "image", "media_type": "image/png", "data": "<base64>"}

Each adapter keeps the conversation in its provider's native format, so
provider-specific pieces (Claude thinking signatures, ``reasoning_content``)
are sent back unchanged. The history is append-only.

* :class:`OpenAIChatClient` - any OpenAI-compatible ``/chat/completions`` endpoint,
  over plain HTTPS (no SDK dependency). Records ``reasoning_content`` /
  ``reasoning`` when the provider returns it.
* :class:`OpenAIResponsesClient` - OpenAI's ``/responses`` endpoint; records the
  reasoning summaries and passes encrypted reasoning items back.
* :class:`AnthropicClient` - the Claude Messages API through the official
  ``anthropic`` SDK (streaming). Records thinking blocks.
"""

from __future__ import annotations

import json
import logging
import random
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The model API failed (after retries) or returned something unusable."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None  # arguments were not valid JSON


@dataclass
class AssistantTurn:
    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    refusal: str | None = None
    model: str | None = None


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: list[dict[str, Any]]
    is_error: bool = False


def _text_of(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text")


def _images_of(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [b for b in blocks if b.get("type") == "image"]


def _object_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    schema = dict(schema or {})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


class LLMClient:
    """Interface used by the runner.

    ``complete`` blocks until the model answers. The runner calls it in a
    background thread with a deadline; when the deadline passes it calls
    :meth:`abandon`, and a late answer to that request is then discarded
    instead of being appended to the history (the request is sent again).
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._epoch = 0
        self._lock = threading.Lock()

    def abandon(self) -> None:
        """Forget the request in flight (it timed out)."""
        with self._lock:
            self._epoch += 1

    def _commit(self, epoch: int, append) -> None:
        """Append a finished turn unless its request was abandoned meanwhile."""
        with self._lock:
            if epoch != self._epoch:
                raise LLMError("answer arrived after the request was abandoned")
            append()

    def start(self, system: str, user: list[dict[str, Any]]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def add_user(self, blocks: list[dict[str, Any]]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def add_tool_results(self, results: list[ToolResult]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def complete(self, tools: list[ToolSpec]) -> AssistantTurn:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------
def _default_urlopen():
    """urlopen that trusts certifi's CA bundle when available (python.org builds
    on macOS ship without root certificates and fail with CERTIFICATE_VERIFY_FAILED)."""
    try:
        import certifi
    except ImportError:
        return urllib.request.urlopen
    context = ssl.create_default_context(cafile=certifi.where())
    return lambda req, timeout=None: urllib.request.urlopen(req, timeout=timeout, context=context)

class OpenAIChatClient(LLMClient):
    RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

    def __init__(self, cfg: LLMConfig, urlopen=None) -> None:
        super().__init__(cfg)
        self.messages: list[dict[str, Any]] = []
        self._urlopen = urlopen or _default_urlopen()
        # Newer OpenAI models reject max_tokens and want max_completion_tokens;
        # switched automatically on the first such error.
        self._max_tokens_key = "max_tokens"
        base = (cfg.base_url or "").rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"

    # -- content conversion ---------------------------------------------------
    def _content(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]] | str:
        parts: list[dict[str, Any]] = []
        for b in blocks:
            if b["type"] == "text":
                parts.append({"type": "text", "text": b["text"]})
            elif b["type"] == "image" and self.cfg.supports_images:
                parts.append({"type": "image_url", "image_url": {"url": f"data:{b['media_type']};base64,{b['data']}"}})
        if all(p["type"] == "text" for p in parts):
            return "\n\n".join(p["text"] for p in parts)
        return parts

    def start(self, system: str, user: list[dict[str, Any]]) -> None:
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": self._content(user)}]

    def add_user(self, blocks: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "user", "content": self._content(blocks)})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        images: list[tuple[str, dict[str, Any]]] = []
        for r in results:
            text = _text_of(r.content) or "(no text output)"
            imgs = _images_of(r.content)
            if imgs:
                text += f"\n[{len(imgs)} image(s) returned; shown in the next message]" if self.cfg.supports_images else ""
                images.extend((r.name, im) for im in imgs)
            if r.is_error:
                text = "ERROR: " + text
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": text})
        if images and self.cfg.supports_images:
            # Chat Completions tool messages are text-only; images follow as a user message.
            blocks: list[dict[str, Any]] = [{"type": "text", "text": "Images returned by the tool calls above:"}]
            for name, im in images:
                blocks.append({"type": "text", "text": f"({name})"})
                blocks.append(im)
            self.messages.append({"role": "user", "content": self._content(blocks)})

    # -- request --------------------------------------------------------------
    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with self._urlopen(req, timeout=self.cfg.timeout_seconds) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:2000]
                last = LLMError(f"HTTP {exc.code} from {self.url}: {detail}")
                if exc.code not in self.RETRY_STATUS:
                    raise last from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = LLMError(f"cannot reach {self.url}: {exc}")
            except ValueError as exc:
                raise LLMError(f"invalid JSON from {self.url}: {exc}") from None
            if attempt < self.cfg.max_retries:
                delay = min(2 ** attempt + random.random(), 30)
                logger.warning("%s; retrying in %.1fs", last, delay)
                time.sleep(delay)
        raise last  # type: ignore[misc]

    def complete(self, tools: list[ToolSpec]) -> AssistantTurn:
        epoch = self._epoch
        body: dict[str, Any] = {"model": self.cfg.model, "messages": list(self.messages)}
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": _object_schema(t.parameters)}}
                for t in tools
            ]
        if self.cfg.max_tokens is not None:
            body[self._max_tokens_key] = self.cfg.max_tokens
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        body.update(self.cfg.extra_body)
        try:
            resp = self._post(body)
        except LLMError as exc:
            if self._max_tokens_key != "max_tokens" or "max_completion_tokens" not in str(exc) or "max_tokens" not in body:
                raise
            logger.info("model wants max_completion_tokens instead of max_tokens; switching")
            self._max_tokens_key = "max_completion_tokens"
            body["max_completion_tokens"] = body.pop("max_tokens")
            resp = self._post(body)
        if resp.get("error"):
            raise LLMError(f"API error: {resp['error']}")
        choices = resp.get("choices") or []
        if not choices:
            raise LLMError(f"response has no choices: {json.dumps(resp)[:500]}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if isinstance(reasoning, (list, dict)):
            reasoning = json.dumps(reasoning, ensure_ascii=False)
        turn = AssistantTurn(
            text=content or "",
            reasoning=reasoning,
            stop_reason=choices[0].get("finish_reason"),
            usage=resp.get("usage") or {},
            refusal=msg.get("refusal"),
            model=resp.get("model"),
        )
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                if not isinstance(args, dict):
                    raise ValueError("arguments are not a JSON object")
                err = None
            except ValueError as exc:
                args, err = {}, f"{exc}: {str(raw)[:300]}"
            turn.tool_calls.append(ToolCall(id=tc.get("id") or f"call_{len(self.messages)}_{i}",
                                            name=fn.get("name", ""), arguments=args, parse_error=err))

        # Append the assistant message exactly as needed for the next request.
        echo: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if msg.get("tool_calls"):
            echo["tool_calls"] = msg["tool_calls"]
        if self.cfg.echo_reasoning and msg.get("reasoning_content"):
            echo["reasoning_content"] = msg["reasoning_content"]
        self._commit(epoch, lambda: self.messages.append(echo))
        return turn


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------
class OpenAIResponsesClient(OpenAIChatClient):
    """``POST {base_url}/responses``. The conversation is kept client-side
    (``store: false``); reasoning items come back with ``encrypted_content`` and
    are passed back unchanged, and their summaries are recorded as reasoning."""

    def __init__(self, cfg: LLMConfig, urlopen=None) -> None:
        super().__init__(cfg, urlopen)
        base = (cfg.base_url or "").rstrip("/")
        self.url = base if base.endswith("/responses") else base + "/responses"
        self.instructions = ""
        self.items: list[dict[str, Any]] = []

    def _input_content(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for b in blocks:
            if b["type"] == "text":
                parts.append({"type": "input_text", "text": b["text"]})
            elif b["type"] == "image" and self.cfg.supports_images:
                parts.append({"type": "input_image", "image_url": f"data:{b['media_type']};base64,{b['data']}"})
        return parts or [{"type": "input_text", "text": "(empty)"}]

    def start(self, system: str, user: list[dict[str, Any]]) -> None:
        self.instructions = system
        self.items = [{"role": "user", "content": self._input_content(user)}]

    def add_user(self, blocks: list[dict[str, Any]]) -> None:
        self.items.append({"role": "user", "content": self._input_content(blocks)})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        images: list[tuple[str, dict[str, Any]]] = []
        for r in results:
            text = _text_of(r.content) or "(no text output)"
            imgs = _images_of(r.content)
            if imgs and self.cfg.supports_images:
                text += f"\n[{len(imgs)} image(s) returned; shown in the next message]"
                images.extend((r.name, im) for im in imgs)
            self.items.append({"type": "function_call_output", "call_id": r.call_id,
                               "output": ("ERROR: " + text) if r.is_error else text})
        if images:
            blocks: list[dict[str, Any]] = [{"type": "text", "text": "Images returned by the tool calls above:"}]
            for name, im in images:
                blocks += [{"type": "text", "text": f"({name})"}, im]
            self.add_user(blocks)

    def complete(self, tools: list[ToolSpec]) -> AssistantTurn:
        epoch = self._epoch
        body: dict[str, Any] = {
            "model": self.cfg.model, "instructions": self.instructions, "input": list(self.items),
            "store": False, "include": ["reasoning.encrypted_content"], "reasoning": {"summary": "auto"},
        }
        if tools:
            body["tools"] = [{"type": "function", "name": t.name, "description": t.description,
                              "parameters": _object_schema(t.parameters)} for t in tools]
        if self.cfg.max_tokens is not None:
            body["max_output_tokens"] = self.cfg.max_tokens
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        body.update(self.cfg.extra_body)
        resp = self._post(body)
        if resp.get("error"):
            raise LLMError(f"API error: {resp['error']}")
        output = resp.get("output") or []
        turn = AssistantTurn(usage=resp.get("usage") or {}, model=resp.get("model"))
        texts, thoughts = [], []
        for item in output:
            kind = item.get("type")
            if kind == "reasoning":
                thoughts += [s.get("text", "") for s in item.get("summary") or [] if s.get("text")]
            elif kind == "message":
                for c in item.get("content") or []:
                    if c.get("type") == "output_text":
                        texts.append(c.get("text", ""))
                    elif c.get("type") == "refusal":
                        turn.refusal = c.get("refusal") or "refusal"
            elif kind == "function_call":
                raw = item.get("arguments") or "{}"
                try:
                    args = json.loads(raw)
                    if not isinstance(args, dict):
                        raise ValueError("arguments are not a JSON object")
                    err = None
                except ValueError as exc:
                    args, err = {}, f"{exc}: {str(raw)[:300]}"
                turn.tool_calls.append(ToolCall(id=item.get("call_id") or item.get("id", ""), name=item.get("name", ""),
                                                arguments=args, parse_error=err))
        turn.text, turn.reasoning = "\n".join(texts), "\n\n".join(thoughts)
        status = resp.get("status")
        turn.stop_reason = "tool_calls" if turn.tool_calls else (
            (resp.get("incomplete_details") or {}).get("reason") if status == "incomplete" else status)
        # reasoning (encrypted), messages and calls go back as they came
        self._commit(epoch, lambda: self.items.extend(output))
        return turn


# ---------------------------------------------------------------------------
# Anthropic (Claude Messages API, official SDK)
# ---------------------------------------------------------------------------
class AnthropicClient(LLMClient):
    def __init__(self, cfg: LLMConfig, client: Any = None) -> None:
        super().__init__(cfg)
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise LLMError("api_format 'anthropic' needs the SDK: pip install anthropic") from exc
            client = anthropic.Anthropic(
                api_key=cfg.api_key or None,
                base_url=cfg.base_url or None,
                timeout=cfg.timeout_seconds,
                max_retries=cfg.max_retries,
            )
        self.client = client
        self.system = ""
        self.messages: list[dict[str, Any]] = []

    @staticmethod
    def _blocks(blocks: list[dict[str, Any]], images: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for b in blocks:
            if b["type"] == "image":
                if images:
                    out.append({"type": "image", "source": {"type": "base64", "media_type": b["media_type"], "data": b["data"]}})
            else:
                out.append({"type": "text", "text": b["text"] or "(empty)"})
        return out or [{"type": "text", "text": "(empty)"}]

    def start(self, system: str, user: list[dict[str, Any]]) -> None:
        self.system = system
        self.messages = [{"role": "user", "content": self._blocks(user, self.cfg.supports_images)}]

    def add_user(self, blocks: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "user", "content": self._blocks(blocks, self.cfg.supports_images)})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        # All results of one assistant turn go back in a single user message.
        content = [
            {"type": "tool_result", "tool_use_id": r.call_id,
             "content": self._blocks(r.content, self.cfg.supports_images), "is_error": r.is_error}
            for r in results
        ]
        self.messages.append({"role": "user", "content": content})

    def complete(self, tools: list[ToolSpec]) -> AssistantTurn:
        epoch = self._epoch
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens or 16000,
            "system": self.system,
            "messages": list(self.messages),
        }
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": _object_schema(t.parameters)}
                               for t in tools]
        if self.cfg.thinking is not None:
            kwargs["thinking"] = self.cfg.thinking
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        if self.cfg.extra_body:
            kwargs["extra_body"] = self.cfg.extra_body
        try:
            with self.client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        except Exception as exc:  # SDK errors carry status + message; the loop records them
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        blocks = [b.model_dump(mode="json", exclude_none=True) for b in msg.content]
        turn = AssistantTurn(stop_reason=msg.stop_reason, model=getattr(msg, "model", None),
                             usage=msg.usage.model_dump(mode="json", exclude_none=True) if msg.usage else {})
        texts, thoughts = [], []
        for b in blocks:
            if b["type"] == "text":
                texts.append(b.get("text", ""))
            elif b["type"] == "thinking":
                if b.get("thinking"):
                    thoughts.append(b["thinking"])
            elif b["type"] == "redacted_thinking":
                thoughts.append("[redacted thinking]")
            elif b["type"] == "tool_use":
                turn.tool_calls.append(ToolCall(id=b["id"], name=b["name"], arguments=dict(b.get("input") or {})))
        turn.text, turn.reasoning = "\n".join(texts), "\n\n".join(thoughts)
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            turn.refusal = str(details.model_dump() if hasattr(details, "model_dump") else details or "refusal")
        # Echo the full content back (thinking blocks must stay unchanged).
        self._commit(epoch, lambda: self.messages.append({"role": "assistant", "content": blocks}))
        return turn


def make_client(cfg: LLMConfig) -> LLMClient:
    if cfg.api_format == "anthropic":
        return AnthropicClient(cfg)
    if cfg.api_format == "openai_responses":
        return OpenAIResponsesClient(cfg)
    return OpenAIChatClient(cfg)
