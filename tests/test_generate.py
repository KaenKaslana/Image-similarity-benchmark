"""Tests for the AI 3D-generation clients (src/generate.py) against a fake HTTP layer."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest
from PIL import Image

from src import generate
from src.generate import GenerationError, GenerationRequest, MeshyProvider, TripoProvider, get_provider


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _http_error(url: str, code: int, body: bytes = b'{"message":"bad"}') -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(body))


@pytest.fixture
def image(tmp_path: Path) -> Path:
    p = tmp_path / "hero.png"
    Image.new("RGB", (8, 8), (200, 200, 200)).save(p)
    return p


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(generate.time, "sleep", lambda s: None)


# ---------------------------------------------------------------------------
# Meshy
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_meshy(monkeypatch: pytest.MonkeyPatch):
    state = {"polls": 0, "statuses": ["PENDING", "IN_PROGRESS", "SUCCEEDED"], "requests": [], "glb": b"GLBDATA"}

    def fake_urlopen(req, timeout=None):
        url, method = req.full_url, req.get_method()
        headers = {k.lower(): v for k, v in req.header_items()}
        body = json.loads(req.data.decode()) if req.data else None
        state["requests"].append((method, url, headers.get("authorization"), body))
        if url == "https://assets.example/out.glb":  # signed asset URL, no auth header
            return _Resp(state["glb"])
        if headers.get("authorization") != "Bearer mk":
            raise _http_error(url, 401)
        if method == "POST" and url.endswith("/openapi/v1/image-to-3d"):
            assert body["image_url"].startswith("data:image/png;base64,")
            return _Resp(b'{"result": "task-img"}')
        if method == "POST" and url.endswith("/openapi/v2/text-to-3d"):
            assert body["mode"] == "preview" and body["prompt"]
            return _Resp(b'{"result": "task-txt"}')
        if method == "GET" and "/image-to-3d/task-img" in url or "/text-to-3d/task-txt" in url:
            status = state["statuses"][min(state["polls"], len(state["statuses"]) - 1)]
            state["polls"] += 1
            payload = {"status": status, "progress": 33 * state["polls"]}
            if status == "SUCCEEDED":
                payload["model_urls"] = {"glb": "https://assets.example/out.glb"}
                payload["consumed_credits"] = 5
            return _Resp(json.dumps(payload).encode())
        raise _http_error(url, 404)

    monkeypatch.setattr(generate, "urlopen", fake_urlopen)
    monkeypatch.delenv("MESHY_API_KEY", raising=False)
    monkeypatch.delenv("TRIPO_API_KEY", raising=False)
    return state


def test_meshy_image_to_3d(fake_meshy, image: Path, tmp_path: Path, no_sleep) -> None:
    provider = MeshyProvider("mk", sleep=lambda s: None)
    model = provider.generate(GenerationRequest(image=image, poll_interval=0.01, timeout=5), tmp_path / "gen")
    assert model.provider == "meshy" and model.task_id == "task-img" and model.mode == "image"
    assert model.path.read_bytes() == b"GLBDATA"
    assert model.meta["consumed_credits"] == 5
    assert json.loads(model.path.with_suffix(".json").read_text())["task_id"] == "task-img"
    assert fake_meshy["polls"] == 3


def test_meshy_text_to_3d(fake_meshy, tmp_path: Path) -> None:
    provider = MeshyProvider("mk", sleep=lambda s: None)
    model = provider.generate(GenerationRequest(prompt="a wooden chair", poll_interval=0.01, timeout=5), tmp_path)
    assert model.mode == "text" and model.prompt == "a wooden chair" and model.path.name == "meshy_task-txt.glb"


def test_meshy_failed_task(fake_meshy, image: Path, tmp_path: Path) -> None:
    fake_meshy["statuses"] = ["PENDING", "FAILED"]
    with pytest.raises(GenerationError, match="FAILED"):
        MeshyProvider("mk", sleep=lambda s: None).generate(GenerationRequest(image=image, poll_interval=0.01, timeout=5), tmp_path)


def test_meshy_timeout(fake_meshy, image: Path, tmp_path: Path) -> None:
    fake_meshy["statuses"] = ["IN_PROGRESS"]
    with pytest.raises(GenerationError, match="timed out"):
        MeshyProvider("mk", sleep=lambda s: None).generate(GenerationRequest(image=image, poll_interval=0.0001, timeout=0.05), tmp_path)


def test_bad_key_is_reported(fake_meshy, image: Path, tmp_path: Path) -> None:
    with pytest.raises(GenerationError, match="HTTP 401"):
        MeshyProvider("wrong", sleep=lambda s: None).generate(GenerationRequest(image=image, poll_interval=0.01, timeout=5), tmp_path)


def test_get_provider_needs_key(fake_meshy, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(GenerationError, match="MESHY_API_KEY"):
        get_provider("meshy")
    monkeypatch.setenv("TRIPO_API_KEY", "tk")
    assert isinstance(get_provider("tripo"), TripoProvider)
    assert isinstance(get_provider("meshy", "explicit"), MeshyProvider)
    with pytest.raises(GenerationError):
        get_provider("nope", "x")


def test_request_validation(image: Path) -> None:
    with pytest.raises(GenerationError):
        GenerationRequest().validate()
    with pytest.raises(GenerationError):
        GenerationRequest(image=image, prompt="both").validate()
    with pytest.raises(GenerationError):
        GenerationRequest(image=image.with_name("missing.png")).validate()
    GenerationRequest(prompt="ok").validate()


# ---------------------------------------------------------------------------
# Tripo
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_tripo(monkeypatch: pytest.MonkeyPatch):
    state = {"polls": 0, "statuses": ["queued", "running", "success"], "tasks": []}

    def fake_urlopen(req, timeout=None):
        url, method = req.full_url, req.get_method()
        headers = {k.lower(): v for k, v in req.header_items()}
        if headers.get("authorization") != "Bearer tk":
            raise _http_error(url, 401)
        if method == "POST" and url.endswith("/v2/openapi/upload"):
            assert headers["content-type"].startswith("multipart/form-data; boundary=")
            assert b'name="file"; filename="hero.png"' in req.data
            return _Resp(b'{"code": 0, "data": {"image_token": "img-token"}}')
        if method == "POST" and url.endswith("/v2/openapi/task"):
            body = json.loads(req.data.decode())
            state["tasks"].append(body)
            return _Resp(b'{"code": 0, "data": {"task_id": "t-1"}}')
        if method == "GET" and url.endswith("/v2/openapi/task/t-1"):
            status = state["statuses"][min(state["polls"], len(state["statuses"]) - 1)]
            state["polls"] += 1
            data = {"task_id": "t-1", "type": "image_to_model", "status": status, "progress": 50}
            if status == "success":
                data["output"] = {"model": "https://cdn.example/t1.glb", "pbr_model": "https://cdn.example/t1_pbr.glb"}
            return _Resp(json.dumps({"code": 0, "data": data}).encode())
        if url == "https://cdn.example/t1_pbr.glb":
            return _Resp(b"TRIPOGLB")
        raise _http_error(url, 404)

    monkeypatch.setattr(generate, "urlopen", fake_urlopen)
    return state


def test_tripo_image_to_model(fake_tripo, image: Path, tmp_path: Path) -> None:
    model = TripoProvider("tk", sleep=lambda s: None).generate(GenerationRequest(image=image, poll_interval=0.01, timeout=5), tmp_path)
    assert model.path.read_bytes() == b"TRIPOGLB" and model.task_id == "t-1"
    task = fake_tripo["tasks"][0]
    assert task["type"] == "image_to_model" and task["file"] == {"type": "png", "file_token": "img-token"}
    assert task["texture"] is False


def test_tripo_text_to_model_and_failure(fake_tripo, tmp_path: Path) -> None:
    provider = TripoProvider("tk", sleep=lambda s: None)
    model = provider.generate(GenerationRequest(prompt="a mug", texture=True, poll_interval=0.01, timeout=5), tmp_path)
    assert fake_tripo["tasks"][-1] == {"type": "text_to_model", "prompt": "a mug"}
    assert model.mode == "text"
    fake_tripo["statuses"] = ["queued", "failed"]
    fake_tripo["polls"] = 0
    with pytest.raises(GenerationError, match="failed"):
        provider.generate(GenerationRequest(prompt="a mug", poll_interval=0.01, timeout=5), tmp_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_generate_model(fake_meshy, image: Path, tmp_path: Path, capsys) -> None:
    from src.cli import main

    code = main(["generate-model", "--provider", "meshy", "--api-key", "mk", "--image", str(image),
                 "--output-dir", str(tmp_path / "gen"), "--poll-interval", "0.01", "--log-level", "WARNING"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "meshy" and Path(payload["path"]).is_file()


def test_cli_reproduce_with_fake_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """reproduce: renders a hero image, 'generates' (a rotated copy), auto-orients and scores."""
    trimesh = pytest.importorskip("trimesh")
    import numpy as np

    from src import cli
    from src.generate import GeneratedModel

    box = trimesh.creation.box(extents=[1.0, 2.0, 1.0])
    knob = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
    knob.apply_translation([0.7, 1.2, 0.7])
    mesh = trimesh.util.concatenate([box, knob])
    ref_path = tmp_path / "ref.glb"
    mesh.export(ref_path)
    rotated = mesh.copy()
    rotated.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    gen_path = tmp_path / "generated.glb"
    rotated.export(gen_path)

    seen: dict = {}

    class FakeProvider:
        name = "fake"

        def generate(self, req: GenerationRequest, dest_dir):
            seen["req"] = req
            assert req.image is not None and Path(req.image).is_file()
            with Image.open(req.image) as im:
                assert im.size == (256, 256) and im.mode == "RGB"
            return GeneratedModel(provider="fake", task_id="x", path=gen_path, mode=req.mode, image=str(req.image))

    monkeypatch.setattr(cli, "get_provider", lambda name, key=None: FakeProvider())

    out = tmp_path / "runs"
    code = cli.main(["reproduce", "--reference", str(ref_path), "--provider", "meshy", "--hero-size", "256",
                     "--output", str(out), "--size", "96", "--canvas-size", "96", "--device", "cpu",
                     "--log-level", "WARNING"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "overall_score" in printed
    run = next(out.glob("run_*"))
    assert run.name.endswith("_ref-vs-meshy-image")
    assert (run / "generation" / "hero_iso.png").is_file()
    meta = json.loads((run / "models.json").read_text())
    assert meta["generation"]["provider"] == "fake"
    assert meta["candidate"]["auto_orient"]["mean_iou"] > 0.97
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["overall_score"] > 95  # rotated copy is recovered by auto-orient

    # --candidate skips generation entirely and never touches the provider
    monkeypatch.setattr(cli, "get_provider", lambda *a, **k: pytest.fail("provider must not be used"))
    code = cli.main(["reproduce", "--reference", str(ref_path), "--candidate", str(gen_path), "--no-save",
                     "--size", "64", "--canvas-size", "64", "--device", "cpu", "--log-level", "WARNING"])
    assert code == 0


def test_describe_model(tmp_path: Path) -> None:
    from src.cli import describe_model

    plain = tmp_path / "thing.glb"
    plain.write_bytes(b"x")
    assert describe_model(plain) == "thing"
    assert describe_model(plain, {"name": "Victorian chair"}) == "Victorian chair"
    gen = {"provider": "tripo", "mode": "text", "meta": {"model_version_used": "v3.1-20260211"}}
    assert describe_model(plain, None, gen) == "tripo-text-v3.1"
    # sidecar json written by fetch-sketchfab / generate-model
    sf = tmp_path / "abc.glb"
    sf.write_bytes(b"x")
    sf.with_suffix(".json").write_text(json.dumps({"uid": "abc", "name": "Coffee Mug"}))
    assert describe_model(sf) == "Coffee Mug"
    g = tmp_path / "meshy_1.glb"
    g.write_bytes(b"x")
    g.with_suffix(".json").write_text(json.dumps({"provider": "meshy", "mode": "image", "meta": {}}))
    assert describe_model(g) == "meshy-image"
