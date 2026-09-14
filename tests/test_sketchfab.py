"""Tests for the Sketchfab client (src/sketchfab.py) with a fake HTTP layer."""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from pathlib import Path

import pytest

from src import sketchfab
from src.sketchfab import SketchfabError, download_model, is_sketchfab_reference, parse_model_uid

UID = "0123456789abcdef0123456789abcdef"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "denied", {}, io.BytesIO(b'{"detail":"nope"}'))


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch):
    """Patch ``urlopen`` with a small in-memory Sketchfab."""
    state = {
        "downloadable": True,
        "formats": {"glb": {"url": "https://cdn.example/model.glb", "size": 3, "expires": 60}},
        "payloads": {"https://cdn.example/model.glb": b"GLB"},
        "calls": [],
        "require_token": True,
    }

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        headers = {k.lower(): v for k, v in req.header_items()}
        state["calls"].append((url, headers.get("authorization")))
        if url == f"{sketchfab.API_ROOT}/models/{UID}":
            body = {
                "name": "Test Mug",
                "isDownloadable": state["downloadable"],
                "license": {"label": "CC Attribution", "slug": "by"},
                "user": {"displayName": "someone"},
                "viewerUrl": f"https://sketchfab.com/models/{UID}",
            }
            return _Resp(json.dumps(body).encode())
        if url == f"{sketchfab.API_ROOT}/models/{UID}/download":
            if state["require_token"] and not headers.get("authorization"):
                raise _http_error(url, 401)
            if headers.get("authorization") == "Token bad":
                raise _http_error(url, 403)
            return _Resp(json.dumps(state["formats"]).encode())
        if url in state["payloads"]:
            return _Resp(state["payloads"][url])
        raise _http_error(url, 404)

    monkeypatch.setattr(sketchfab, "urlopen", fake_urlopen)
    monkeypatch.delenv(sketchfab.ENV_API_TOKEN, raising=False)
    monkeypatch.delenv(sketchfab.ENV_ACCESS_TOKEN, raising=False)
    return state


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        UID,
        UID.upper(),
        f"sketchfab:{UID}",
        f"https://sketchfab.com/3d-models/coffee-mug-{UID}",
        f"https://sketchfab.com/3d-models/coffee-mug-{UID}?utm_source=x",
        f"https://sketchfab.com/models/{UID}",
        f"https://sketchfab.com/models/{UID}/embed",
        f"http://www.sketchfab.com/3d-models/{UID}",
    ],
)
def test_parse_model_uid_accepts_common_forms(text: str) -> None:
    assert parse_model_uid(text) == UID
    assert is_sketchfab_reference(text)


@pytest.mark.parametrize("text", ["mug.glb", "https://example.com/model.glb", "https://skfb.ly/abc", "123"])
def test_parse_model_uid_rejects_other_input(text: str) -> None:
    with pytest.raises(SketchfabError):
        parse_model_uid(text)
    assert not is_sketchfab_reference(text)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def test_download_requires_credentials(fake_api, tmp_path: Path) -> None:
    with pytest.raises(SketchfabError, match="SKETCHFAB_API_TOKEN"):
        download_model(UID, tmp_path)


def test_download_glb_and_cache(fake_api, tmp_path: Path) -> None:
    path, info = download_model(f"https://sketchfab.com/3d-models/mug-{UID}", tmp_path, token="secret")
    assert path == tmp_path / f"{UID}.glb"
    assert path.read_bytes() == b"GLB"
    assert info.name == "Test Mug" and info.license == "CC Attribution" and info.author == "someone"
    assert json.loads((tmp_path / f"{UID}.json").read_text())["license"] == "CC Attribution"
    assert any(auth == "Token secret" for _, auth in fake_api["calls"])

    calls_before = len(fake_api["calls"])
    path2, info2 = download_model(UID, tmp_path, token="secret")
    assert path2 == path and info2.name == "Test Mug"
    assert len(fake_api["calls"]) == calls_before  # served from cache, no HTTP


def test_download_uses_env_token(fake_api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sketchfab.ENV_API_TOKEN, "fromenv")
    download_model(UID, tmp_path)
    assert any(auth == "Token fromenv" for _, auth in fake_api["calls"])


def test_download_rejects_non_downloadable(fake_api, tmp_path: Path) -> None:
    fake_api["downloadable"] = False
    with pytest.raises(SketchfabError, match="not marked downloadable"):
        download_model(UID, tmp_path, token="secret")


def test_download_reports_forbidden(fake_api, tmp_path: Path) -> None:
    with pytest.raises(SketchfabError, match="refused"):
        download_model(UID, tmp_path, token="bad")


def test_download_gltf_zip_is_unpacked(fake_api, tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("scene.gltf", "{}")
        zf.writestr("scene.bin", b"\0")
    fake_api["formats"] = {"gltf": {"url": "https://cdn.example/model.zip", "size": 1, "expires": 60}}
    fake_api["payloads"] = {"https://cdn.example/model.zip": buf.getvalue()}
    path, _ = download_model(UID, tmp_path, token="secret")
    assert path.name == "scene.gltf" and path.is_file()
    assert (tmp_path / f"{UID}.zip").is_file()
