"""Ask an AI 3D-generation service to recreate a model from an image or a prompt.

Two hosted providers are implemented; both expose an asynchronous
"create task -> poll -> download GLB" workflow over plain HTTPS:

* **Meshy** (https://docs.meshy.ai) - ``MESHY_API_KEY``
  image-to-3D: ``POST /openapi/v1/image-to-3d`` with a base64 data URI;
  text-to-3D:  ``POST /openapi/v2/text-to-3d`` (``mode: preview``).
* **Tripo** (https://platform.tripo3d.ai) - ``TRIPO_API_KEY``
  image-to-3D: ``POST /v2/openapi/upload`` (multipart) then ``POST /task``
  with ``type: image_to_model``; text-to-3D: ``type: text_to_model``.

Both services are paid / credit based. Nothing here is called unless the
user explicitly runs ``generate-model`` or ``reproduce``.

Network access goes through :data:`urlopen` so tests can replace it.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen  # noqa: F401  (patched in tests)

logger = logging.getLogger(__name__)

USER_AGENT = "image-similarity-benchmark/0.1"
PROVIDERS = ("meshy", "tripo")
ENV_KEYS = {"meshy": "MESHY_API_KEY", "tripo": "TRIPO_API_KEY"}


class GenerationError(RuntimeError):
    """Raised for missing credentials, API errors, failed or timed-out tasks."""


@dataclass
class GenerationRequest:
    """What to generate. Exactly one of ``image`` / ``prompt`` drives the task."""

    image: Path | None = None
    prompt: str | None = None
    texture: bool = False
    model_version: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    poll_interval: float = 10.0
    timeout: float = 1800.0

    @property
    def mode(self) -> str:
        return "image" if self.image is not None else "text"

    def validate(self) -> None:
        if (self.image is None) == (self.prompt is None or self.prompt == ""):
            raise GenerationError("give exactly one of an input image or a text prompt")
        if self.image is not None and not Path(self.image).is_file():
            raise GenerationError(f"input image not found: {self.image}")
        if self.poll_interval <= 0 or self.timeout <= 0:
            raise GenerationError("poll_interval and timeout must be positive")


@dataclass
class GeneratedModel:
    provider: str
    task_id: str
    path: Path
    mode: str
    prompt: str | None = None
    image: str | None = None
    elapsed_seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "task_id": self.task_id,
            "path": str(self.path),
            "mode": self.mode,
            "prompt": self.prompt,
            "image": self.image,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _http(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 120.0,
) -> bytes:
    req = urllib.request.Request(url, data=body, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:400]
        except Exception:  # pragma: no cover
            pass
        raise GenerationError(f"HTTP {exc.code} from {url}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise GenerationError(f"network error for {url}: {exc.reason}") from exc


def _json(method: str, url: str, headers: dict[str, str], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None
    hdrs = dict(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    raw = _http(method, url, hdrs, body)
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise GenerationError(f"invalid JSON from {url}: {raw[:200]!r}") from exc
    if not isinstance(data, dict):
        raise GenerationError(f"unexpected response from {url}: {data!r}")
    return data


def _multipart(field_name: str, path: Path, content_type: str) -> tuple[bytes, str]:
    boundary = f"----imgsim{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + path.read_bytes() + tail, f"multipart/form-data; boundary={boundary}"


def _image_data_uri(path: Path) -> str:
    ext = path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext.lstrip("."))
    if mime is None:
        raise GenerationError(f"unsupported image type {ext!r}; use PNG, JPEG or WebP")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _poll(
    fetch: Callable[[], tuple[str, float | None, dict[str, Any]]],
    done: set[str],
    failed: set[str],
    interval: float,
    timeout: float,
    label: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll ``fetch`` until a terminal status. Returns the final payload."""
    start = time.monotonic()
    last_logged: float | None = None
    while True:
        status, progress, payload = fetch()
        if status in done:
            return payload
        if status in failed:
            raise GenerationError(f"{label}: task ended with status {status!r}: {payload}")
        if progress is not None and progress != last_logged:
            logger.info("%s: %s (%.0f%%)", label, status, progress)
            last_logged = progress
        if time.monotonic() - start > timeout:
            raise GenerationError(f"{label}: timed out after {timeout:.0f}s (last status {status!r})")
        sleep(interval)


def api_key_for(provider: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    env = ENV_KEYS.get(provider)
    key = os.environ.get(env, "").strip() if env else ""
    if not key:
        raise GenerationError(f"no API key for {provider}: pass --api-key or set {env}")
    return key


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class Provider:
    name = "base"

    def __init__(self, api_key: str, sleep: Callable[[float], None] = time.sleep) -> None:
        self.api_key = api_key
        self._sleep = sleep

    def generate(self, req: GenerationRequest, dest_dir: str | Path) -> GeneratedModel:
        req.validate()
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        task_id, glb_url, meta = self._run(req)
        target = dest_dir / f"{self.name}_{_safe(task_id)}.glb"
        logger.info("%s: downloading result to %s", self.name, target)
        payload = _http("GET", glb_url, self._download_headers(), timeout=600.0)
        if not payload:
            raise GenerationError(f"{self.name}: empty model download")
        target.write_bytes(payload)
        result = GeneratedModel(
            provider=self.name,
            task_id=task_id,
            path=target,
            mode=req.mode,
            prompt=req.prompt,
            image=str(req.image) if req.image else None,
            elapsed_seconds=time.monotonic() - start,
            meta={**meta, "model_version": req.model_version, "options": req.options, "texture": req.texture},
        )
        with open(target.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        return result

    def _run(self, req: GenerationRequest) -> tuple[str, str, dict[str, Any]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _download_headers(self) -> dict[str, str]:
        return {}


class MeshyProvider(Provider):
    name = "meshy"
    base = "https://api.meshy.ai/openapi"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _run(self, req: GenerationRequest) -> tuple[str, str, dict[str, Any]]:
        if req.mode == "image":
            endpoint = f"{self.base}/v1/image-to-3d"
            payload: dict[str, Any] = {"image_url": _image_data_uri(Path(req.image)), "should_texture": bool(req.texture)}
        else:
            endpoint = f"{self.base}/v2/text-to-3d"
            payload = {"mode": "preview", "prompt": req.prompt}
        if req.model_version:
            payload["ai_model"] = req.model_version
        payload.update(req.options)
        created = _json("POST", endpoint, self._headers(), payload)
        task_id = str(created.get("result") or "")
        if not task_id:
            raise GenerationError(f"meshy: no task id in response {created}")
        logger.info("meshy: task %s created (%s)", task_id, req.mode)

        def fetch() -> tuple[str, float | None, dict[str, Any]]:
            data = _json("GET", f"{endpoint}/{task_id}", self._headers())
            return str(data.get("status", "")), data.get("progress"), data

        final = _poll(fetch, {"SUCCEEDED"}, {"FAILED", "CANCELED", "CANCELLED"}, req.poll_interval, req.timeout, "meshy", self._sleep)
        urls = final.get("model_urls") or {}
        glb = urls.get("glb")
        if not glb:
            raise GenerationError(f"meshy: task succeeded but no glb url in {urls}")
        meta = {k: final.get(k) for k in ("thumbnail_url", "consumed_credits", "ai_model", "art_style") if final.get(k) is not None}
        return task_id, str(glb), meta


class TripoProvider(Provider):
    name = "tripo"
    base = "https://api.tripo3d.ai/v2/openapi"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _download_headers(self) -> dict[str, str]:
        return self._headers()

    @staticmethod
    def _data(resp: dict[str, Any], what: str) -> dict[str, Any]:
        if resp.get("code", 0) != 0:
            raise GenerationError(f"tripo: {what} failed: {resp}")
        data = resp.get("data")
        if not isinstance(data, dict):
            raise GenerationError(f"tripo: {what}: missing data in {resp}")
        return data

    def _run(self, req: GenerationRequest) -> tuple[str, str, dict[str, Any]]:
        if req.mode == "image":
            img = Path(req.image)
            ext = img.suffix.lower().lstrip(".")
            ftype = {"png": "png", "jpg": "jpg", "jpeg": "jpg", "webp": "webp"}.get(ext)
            if ftype is None:
                raise GenerationError(f"tripo: unsupported image type {img.suffix!r}")
            body, ctype = _multipart("file", img, f"image/{'jpeg' if ftype == 'jpg' else ftype}")
            raw = _http("POST", f"{self.base}/upload", {**self._headers(), "Content-Type": ctype}, body)
            upload = self._data(json.loads(raw.decode("utf-8")), "upload")
            token = upload.get("image_token")
            if not token:
                raise GenerationError(f"tripo: upload returned no image_token: {upload}")
            task: dict[str, Any] = {"type": "image_to_model", "file": {"type": ftype, "file_token": token}}
        else:
            task = {"type": "text_to_model", "prompt": req.prompt}
        if not req.texture:
            task["texture"] = False
            task["pbr"] = False
        if req.model_version:
            task["model_version"] = req.model_version
        task.update(req.options)
        logger.info("tripo: task parameters %s", {k: v for k, v in task.items() if k != "file"})
        created = self._data(_json("POST", f"{self.base}/task", self._headers(), task), "create task")
        task_id = str(created.get("task_id") or "")
        if not task_id:
            raise GenerationError(f"tripo: no task_id in {created}")
        logger.info("tripo: task %s created (%s)", task_id, req.mode)

        def fetch() -> tuple[str, float | None, dict[str, Any]]:
            data = self._data(_json("GET", f"{self.base}/task/{task_id}", self._headers()), "poll")
            return str(data.get("status", "")), data.get("progress"), data

        final = _poll(fetch, {"success"}, {"failed", "cancelled", "banned", "expired", "unknown"}, req.poll_interval, req.timeout, "tripo", self._sleep)
        output = final.get("output") or {}
        glb = output.get("pbr_model") or output.get("model") or output.get("base_model")
        if not glb:
            raise GenerationError(f"tripo: task succeeded but no model url in {output}")
        meta = {k: final.get(k) for k in ("type", "create_time", "consumed_credit") if final.get(k) is not None}
        inp = final.get("input") or {}
        meta["model_version_used"] = inp.get("model_version")
        return task_id, str(glb), meta


def tripo_balance(api_key: str) -> int | None:
    """Remaining Tripo credits, or ``None`` if the endpoint is unavailable."""
    try:
        data = _json("GET", f"{TripoProvider.base}/user/balance", {"Authorization": f"Bearer {api_key}"})
        return int((data.get("data") or {}).get("balance"))
    except (GenerationError, TypeError, ValueError):
        return None


def parse_options(items: list[str] | None) -> dict[str, Any]:
    """``["face_limit=5000", "quad=true"]`` -> ``{"face_limit": 5000, "quad": True}``."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise GenerationError(f"--param expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        v = value.strip()
        low = v.lower()
        if low in ("true", "false"):
            out[key] = low == "true"
        else:
            try:
                out[key] = int(v)
            except ValueError:
                try:
                    out[key] = float(v)
                except ValueError:
                    out[key] = v
    return out


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:80] or "task"


def get_provider(name: str, api_key: str | None = None) -> Provider:
    name = name.lower().strip()
    if name not in PROVIDERS:
        raise GenerationError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")
    key = api_key_for(name, api_key)
    return MeshyProvider(key) if name == "meshy" else TripoProvider(key)
