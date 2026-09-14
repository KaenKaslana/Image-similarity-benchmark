"""Fetch downloadable models from Sketchfab.

Sketchfab exposes two HTTP APIs:

* the **Data API** (``GET /v3/models/{uid}``) is public and returns metadata
  such as the model name, licence and whether it is downloadable;
* the **Download API** (``GET /v3/models/{uid}/download``) requires
  authentication and returns short-lived URLs for the glTF / GLB / USDZ
  exports. Only models whose author enabled downloads (usually Creative
  Commons licensed) can be fetched, and the licence must be respected.

Authentication: pass an API token (Sketchfab settings -> Password & API ->
"API token") via ``token=`` or the ``SKETCHFAB_API_TOKEN`` environment
variable. An OAuth access token is also accepted through
``SKETCHFAB_ACCESS_TOKEN`` (sent as ``Bearer``).

Network access goes through :func:`urllib.request.urlopen` so tests can
monkeypatch :data:`urlopen` without a real account.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen  # noqa: F401  (patched in tests)

logger = logging.getLogger(__name__)

API_ROOT = "https://api.sketchfab.com/v3"
ENV_API_TOKEN = "SKETCHFAB_API_TOKEN"
ENV_ACCESS_TOKEN = "SKETCHFAB_ACCESS_TOKEN"
USER_AGENT = "image-similarity-benchmark/0.1 (+https://github.com/)"
PREFERRED_FORMATS = ("glb", "gltf")

_UID_RE = re.compile(r"^[0-9a-f]{32}$")
_URL_PATTERNS = (
    re.compile(r"sketchfab\.com/3d-models/(?:[^/?#]*-)?([0-9a-f]{32})(?:[/?#]|$)"),
    re.compile(r"sketchfab\.com/models/([0-9a-f]{32})(?:[/?#]|$)"),
    re.compile(r"sketchfab\.com/show/([0-9a-f]{32})(?:[/?#]|$)"),
)


class SketchfabError(RuntimeError):
    """Raised for invalid references, missing credentials and API failures."""


@dataclass
class SketchfabModel:
    uid: str
    name: str
    license: str | None
    downloadable: bool
    author: str | None
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "name": self.name,
            "license": self.license,
            "downloadable": self.downloadable,
            "author": self.author,
            "url": self.url,
        }


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def is_sketchfab_reference(text: str) -> bool:
    """True for Sketchfab model URLs, ``sketchfab:<uid>`` and bare 32-hex uids."""
    try:
        parse_model_uid(text)
    except SketchfabError:
        return False
    return True


def parse_model_uid(text: str) -> str:
    """Extract the 32-character model uid from a URL, ``sketchfab:<uid>`` or a bare uid.

    Supported URL shapes::

        https://sketchfab.com/3d-models/<slug>-<uid>
        https://sketchfab.com/models/<uid>
        https://sketchfab.com/models/<uid>/embed

    ``skfb.ly`` short links are not resolved; open the link in a browser and
    use the full URL instead.
    """
    s = str(text).strip()
    if s.lower().startswith("sketchfab:"):
        s = s[len("sketchfab:") :]
    if _UID_RE.match(s.lower()):
        return s.lower()
    for pat in _URL_PATTERNS:
        m = pat.search(s.lower())
        if m:
            return m.group(1)
    if "skfb.ly" in s.lower():
        raise SketchfabError(f"short links are not supported ({s}); use the full sketchfab.com model URL")
    raise SketchfabError(f"not a Sketchfab model reference: {text!r}")


# ---------------------------------------------------------------------------
# Credentials and HTTP
# ---------------------------------------------------------------------------
def auth_header(token: str | None = None) -> dict[str, str]:
    """Build the ``Authorization`` header from an explicit token or the environment."""
    if token:
        return {"Authorization": f"Token {token.strip()}"}
    api = os.environ.get(ENV_API_TOKEN, "").strip()
    if api:
        return {"Authorization": f"Token {api}"}
    access = os.environ.get(ENV_ACCESS_TOKEN, "").strip()
    if access:
        return {"Authorization": f"Bearer {access}"}
    raise SketchfabError(
        "no Sketchfab credentials: pass --token or set the "
        f"{ENV_API_TOKEN} environment variable (Sketchfab -> Settings -> Password & API -> API token)"
    )


def _request(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # pragma: no cover
            pass
        raise SketchfabError(f"HTTP {exc.code} from {url}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SketchfabError(f"network error fetching {url}: {exc.reason}") from exc


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    raw = _request(url, headers)
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise SketchfabError(f"invalid JSON from {url}") from exc
    if not isinstance(data, dict):
        raise SketchfabError(f"unexpected response from {url}")
    return data


def fetch_model_info(uid: str) -> SketchfabModel:
    """Public metadata for a model (no credentials needed)."""
    data = _get_json(f"{API_ROOT}/models/{uid}")
    lic = data.get("license")
    if isinstance(lic, dict):
        lic = lic.get("label") or lic.get("slug")
    user = data.get("user")
    author = user.get("displayName") or user.get("username") if isinstance(user, dict) else None
    return SketchfabModel(
        uid=uid,
        name=str(data.get("name") or uid),
        license=lic if lic is None else str(lic),
        downloadable=bool(data.get("isDownloadable", False)),
        author=author,
        url=str(data.get("viewerUrl") or f"https://sketchfab.com/models/{uid}"),
    )


def request_download_urls(uid: str, token: str | None = None) -> dict[str, dict[str, Any]]:
    """Call the Download API. Returns ``{format: {"url": ..., "size": ..., "expires": ...}}``."""
    headers = auth_header(token)
    try:
        data = _get_json(f"{API_ROOT}/models/{uid}/download", headers)
    except SketchfabError as exc:
        msg = str(exc)
        if "HTTP 401" in msg or "HTTP 403" in msg:
            raise SketchfabError(
                f"download of {uid} refused ({msg}). Check the token and whether the author "
                "enabled downloads for this model (only 'Downloadable' models can be fetched)."
            ) from exc
        if "HTTP 404" in msg:
            raise SketchfabError(f"model {uid} not found on Sketchfab") from exc
        raise
    formats = {k: v for k, v in data.items() if isinstance(v, dict) and v.get("url")}
    if not formats:
        raise SketchfabError(f"no downloadable formats returned for {uid}: {data}")
    return formats


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _pick_mesh_in_dir(directory: Path) -> Path | None:
    for pattern in ("*.glb", "*.gltf", "*.obj", "*.stl", "*.ply"):
        hits = sorted(directory.rglob(pattern))
        if hits:
            return hits[0]
    return None


def _unpack_if_zip(path: Path) -> Path:
    if not zipfile.is_zipfile(path):
        return path
    target = path.with_suffix("")
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            member_path = (target / member).resolve()
            if not str(member_path).startswith(str(target.resolve())):
                raise SketchfabError(f"refusing to extract unsafe path {member!r} from {path.name}")
        zf.extractall(target)
    mesh = _pick_mesh_in_dir(target)
    if mesh is None:
        raise SketchfabError(f"archive {path.name} contains no supported mesh file")
    return mesh


def download_model(
    reference: str,
    dest_dir: str | Path,
    token: str | None = None,
    formats: tuple[str, ...] = PREFERRED_FORMATS,
    force: bool = False,
) -> tuple[Path, SketchfabModel]:
    """Download a Sketchfab model and return ``(local mesh path, metadata)``.

    Files are cached in ``dest_dir/<uid>.<ext>``; an existing non-empty file
    is reused unless ``force`` is set. ``dest_dir/<uid>.json`` records the
    metadata (name, licence, author) so attribution is not lost.
    """
    uid = parse_model_uid(reference)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    info_path = dest_dir / f"{uid}.json"
    if not force:
        for ext in (".glb", ".zip", ".gltf"):
            cached = dest_dir / f"{uid}{ext}"
            if cached.is_file() and cached.stat().st_size > 0:
                logger.info("Using cached Sketchfab model %s", cached)
                info = _load_cached_info(info_path, uid)
                return _unpack_if_zip(cached), info

    info = fetch_model_info(uid)
    logger.info("Sketchfab model '%s' by %s (licence: %s)", info.name, info.author, info.license)
    if not info.downloadable:
        raise SketchfabError(
            f"model '{info.name}' ({uid}) is not marked downloadable by its author; "
            "pick a model with the 'Download' button enabled"
        )
    urls = request_download_urls(uid, token)
    fmt = next((f for f in formats if f in urls), None)
    if fmt is None:
        raise SketchfabError(f"none of the formats {formats} available; got {sorted(urls)}")
    url = str(urls[fmt]["url"])
    ext = ".glb" if fmt == "glb" else ".zip"
    target = dest_dir / f"{uid}{ext}"
    logger.info("Downloading %s (%s, %s bytes)", info.name, fmt, urls[fmt].get("size", "?"))
    payload = _request(url, timeout=600.0)
    if not payload:
        raise SketchfabError(f"empty download for {uid}")
    target.write_bytes(payload)
    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump(info.to_dict(), fh, indent=2)
    return _unpack_if_zip(target), info


def _load_cached_info(info_path: Path, uid: str) -> SketchfabModel:
    if info_path.is_file():
        try:
            d = json.loads(info_path.read_text(encoding="utf-8"))
            return SketchfabModel(
                uid=uid,
                name=str(d.get("name", uid)),
                license=d.get("license"),
                downloadable=bool(d.get("downloadable", True)),
                author=d.get("author"),
                url=str(d.get("url", f"https://sketchfab.com/models/{uid}")),
            )
        except (ValueError, OSError):
            pass
    return SketchfabModel(uid, uid, None, True, None, f"https://sketchfab.com/models/{uid}")
