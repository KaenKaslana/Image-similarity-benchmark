"""Smoke test for the headless Blender MCP container.

Connects to the Streamable HTTP endpoint, builds a small model through
execute_blender_code, exports it as glb into the shared models/ volume, and
grabs a screenshot. Afterwards the glb can be scored with the benchmark:

    python -m src.cli compare-models --reference models/ref.glb --candidate models/mcp_smoke.glb --auto-orient

Requires the MCP client library on the host:  pip install "mcp>=1.9,<2"

    python scripts/blender_mcp_smoke.py --url http://localhost:8000/mcp --out models/mcp_smoke.glb
"""
import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # pragma: no cover
    sys.exit("pip install 'mcp>=1.9,<2' to run this smoke test")

PROMPT = "smoke test: build a simple mug and export it as glb"

BUILD_CODE = """
import bpy
# Start from an empty scene: the default startup file contains a cube, a light and a camera.
bpy.ops.wm.read_factory_settings(use_empty=True)
# A mug: cylinder body, hollowed, plus a torus handle.
bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1.0, depth=2.0, location=(0, 0, 1.0))
body = bpy.context.active_object
body.name = "mug_body"
bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.85, depth=2.0, location=(0, 0, 1.2))
inner = bpy.context.active_object
mod = body.modifiers.new("hollow", "BOOLEAN")
mod.operation = "DIFFERENCE"
mod.object = inner
bpy.context.view_layer.objects.active = body
bpy.ops.object.modifier_apply(modifier="hollow")
bpy.data.objects.remove(inner, do_unlink=True)
bpy.ops.mesh.primitive_torus_add(major_radius=0.6, minor_radius=0.12, location=(1.3, 0, 1.0), rotation=(1.5708, 0, 0))
handle = bpy.context.active_object
handle.name = "mug_handle"
print("objects:", [o.name for o in bpy.context.scene.objects])
"""

EXPORT_CODE = """
import bpy, os
path = {path!r}
os.makedirs(os.path.dirname(path), exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True)
print("exported", path, os.path.getsize(path), "bytes")
"""


def text_of(result):
    parts = []
    for c in result.content:
        if getattr(c, "type", "") == "text":
            parts.append(c.text)
    return "\n".join(parts)


async def run(url: str, container_path: str, screenshot: Path | None):
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"connected to {url}: {len(names)} tools")
            print("  " + ", ".join(names))

            info = await session.call_tool("get_scene_info", {"user_prompt": PROMPT})
            print("get_scene_info:", text_of(info)[:300])

            r = await session.call_tool("execute_blender_code", {"code": BUILD_CODE, "user_prompt": PROMPT})
            print("build:", text_of(r).strip())
            if r.isError:
                raise SystemExit("execute_blender_code failed")

            r = await session.call_tool("execute_blender_code", {"code": EXPORT_CODE.format(path=container_path), "user_prompt": PROMPT})
            print("export:", text_of(r).strip())
            if r.isError:
                raise SystemExit("export failed")

            r = await session.call_tool("get_viewport_screenshot", {"max_size": 512, "user_prompt": PROMPT})
            imgs = [c for c in r.content if getattr(c, "type", "") == "image"]
            if r.isError or not imgs:
                print("screenshot failed:", text_of(r))
            else:
                data = base64.b64decode(imgs[0].data)
                print(f"screenshot: {len(data)} bytes ({imgs[0].mimeType})")
                if screenshot:
                    screenshot.parent.mkdir(parents=True, exist_ok=True)
                    screenshot.write_bytes(data)
                    print(f"saved {screenshot}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000/mcp")
    ap.add_argument("--out", default="models/mcp_smoke.glb",
                    help="glb path relative to the repo root (models/ is mounted at /app/models in the container)")
    ap.add_argument("--screenshot", default="outputs/mcp_smoke.png", help="where to save the screenshot on the host ('' to skip)")
    args = ap.parse_args()
    container_path = "/app/" + args.out.lstrip("./")
    asyncio.run(run(args.url, container_path, Path(args.screenshot) if args.screenshot else None))
    print(f"glb on host: {args.out}")


if __name__ == "__main__":
    main()
