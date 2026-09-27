"""Run the "MCP for Blender" addon socket server inside a headless Blender.

Usage (inside the container):  blender -b --python headless_server.py

Why this exists: the stock addon refuses to start under `blender -b`. Its socket
thread queues commands and a bpy.app.timers callback executes them on the main
thread, but timers only fire from the window event loop, which background mode
does not have. This launcher loads the addon unmodified, opens the same socket
server with the background check bypassed, and pumps the addon's own command
queue from a plain main-thread loop. Every command therefore runs on the main
thread exactly as it would in the GUI.

Differences from the GUI addon, all handled here:
  * get_viewport_screenshot: there is no viewport, so the scene is rendered
    from an auto-framed camera instead (Workbench, falling back to Cycles CPU).
  * Integrations that rely on bpy.app.timers after the reply (Hyper3D's
    deferred glb import) are not pumped; use Poly Haven / Sketchfab / direct
    execute_blender_code instead.

Environment:
  BLENDER_MCP_ADDON        path to addon.py (default /opt/blender-mcp/addon.py)
  BLENDER_BIND_HOST        socket bind address (default 127.0.0.1; 0.0.0.0 to
                           expose the raw, unauthenticated socket to other hosts)
  BLENDER_PORT             socket port (default 9876)
  BLENDERMCP_INTEGRATIONS  comma-separated integrations to enable in the scene:
                           polyhaven,sketchfab,polypizza,hyper3d,hunyuan3d,tripo
                           (default: polyhaven)
  BLENDERMCP_*_API_KEY     read by the addon itself (see upstream README).
"""
import ctypes
import importlib.util
import os
import signal
import socket
import sys
import threading
import time
import traceback

import bpy
from mathutils import Vector

ADDON_PATH = os.environ.get("BLENDER_MCP_ADDON", "/opt/blender-mcp/addon.py")
BIND_HOST = os.environ.get("BLENDER_BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("BLENDER_PORT", "9876"))
INTEGRATIONS = [
    s.strip().lower()
    for s in os.environ.get("BLENDERMCP_INTEGRATIONS", "polyhaven").split(",")
    if s.strip()
]
POLL_INTERVAL = 0.02
MODULE_NAME = "blender_mcp_addon"


def log(msg):
    print(f"[headless] {msg}", flush=True)


def load_addon():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, ADDON_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    module.register()
    version = ".".join(str(p) for p in module.bl_info.get("version", ()))
    log(f"addon {module.bl_info.get('name')} {version} registered from {ADDON_PATH}")
    return module


def enable_integrations(scene):
    for name in INTEGRATIONS:
        prop = f"blendermcp_use_{name}"
        if hasattr(scene, prop):
            setattr(scene, prop, True)
            log(f"integration enabled: {name}")
        else:
            log(f"unknown integration ignored: {name}")


# ---------------------------------------------------------------------------
# Screenshot replacement: render the scene from an auto-framed camera.
# ---------------------------------------------------------------------------

_FORMAT_MAP = {"jpg": "JPEG", "jpeg": "JPEG"}


def _egl_available():
    """Workbench/Eevee need an EGL context; if libEGL is missing Blender aborts
    the whole process (no Python exception), so probe before selecting them."""
    try:
        ctypes.CDLL("libEGL.so.1")
        return True
    except OSError:
        return False


SNAPSHOT_ENGINES = (["BLENDER_WORKBENCH"] if _egl_available() else []) + ["CYCLES"]


def _scene_bounds(scene):
    points = []
    for obj in scene.objects:
        if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"} and obj.visible_get():
            points.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)
    if not points:
        return Vector((0.0, 0.0, 0.0)), 1.0
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-3)
    return center, radius


def render_snapshot(max_size=800, filepath=None, fmt="png"):
    """Render an isometric-style overview of the scene to *filepath*."""
    if not filepath:
        return {"error": "No filepath provided"}

    scene = bpy.context.scene
    render = scene.render
    saved = {
        "engine": render.engine,
        "res_x": render.resolution_x,
        "res_y": render.resolution_y,
        "res_pct": render.resolution_percentage,
        "filepath": render.filepath,
        "file_format": render.image_settings.file_format,
        "color_mode": render.image_settings.color_mode,
        "camera": scene.camera,
    }

    center, radius = _scene_bounds(scene)
    direction = Vector((1.0, -1.0, 0.8)).normalized()  # front-right, slightly above
    cam_data = bpy.data.cameras.new("mcp_snapshot_cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = radius * 2.2
    cam_data.clip_start = 0.01
    cam_data.clip_end = radius * 20 + 100
    cam_obj = bpy.data.objects.new("mcp_snapshot_cam", cam_data)
    cam_obj.location = center + direction * (radius * 4)
    cam_obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    scene.collection.objects.link(cam_obj)

    method = None
    try:
        scene.camera = cam_obj
        size = max(64, int(max_size))
        render.resolution_x = size
        render.resolution_y = size
        render.resolution_percentage = 100
        render.filepath = filepath
        render.image_settings.file_format = _FORMAT_MAP.get(fmt.lower(), fmt.upper())
        render.image_settings.color_mode = "RGB"

        errors = []
        for engine in SNAPSHOT_ENGINES:
            try:
                render.engine = engine
                if engine == "CYCLES":
                    scene.cycles.device = "CPU"
                    scene.cycles.samples = 16
                    scene.cycles.use_denoising = False
                bpy.ops.render.render(write_still=True)
                if os.path.exists(filepath):
                    method = f"render_{engine.lower()}"
                    break
                errors.append(f"{engine}: no file written")
            except Exception as exc:  # engine unavailable in this build / no GL
                errors.append(f"{engine}: {exc}")
        if method is None:
            return {"error": "render failed: " + "; ".join(errors)}
        return {"success": True, "width": size, "height": size, "filepath": filepath, "method": method}
    finally:
        scene.camera = saved["camera"]
        render.engine = saved["engine"]
        render.resolution_x = saved["res_x"]
        render.resolution_y = saved["res_y"]
        render.resolution_percentage = saved["res_pct"]
        render.filepath = saved["filepath"]
        render.image_settings.file_format = saved["file_format"]
        render.image_settings.color_mode = saved["color_mode"]
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def make_server_class(addon):
    class HeadlessServer(addon.BlenderMCPServer):
        def start(self):
            """Same as the addon's start() minus the background check and the
            timer registration; the main loop below drains the queue instead."""
            if self.running:
                return
            self.running = True
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
            log(f"socket server listening on {self.host}:{self.port}")

        def get_viewport_screenshot(self, max_size=800, filepath=None, format="png"):
            try:
                return render_snapshot(max_size, filepath, format)
            except Exception as exc:
                traceback.print_exc()
                return {"error": f"headless render failed: {exc}"}

    return HeadlessServer


def main():
    if not bpy.app.background:
        log("not in background mode; the normal addon UI handles this case")
    addon = load_addon()
    scene = bpy.context.scene
    scene.blendermcp_port = PORT
    enable_integrations(scene)

    server = make_server_class(addon)(host=BIND_HOST, port=PORT)
    bpy.types.blendermcp_server = server
    server.start()
    scene.blendermcp_server_running = True

    stop = threading.Event()

    def _stop(signum, _frame):
        log(f"signal {signum}, shutting down")
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    log(f"Blender {bpy.app.version_string} ready; snapshot engines {SNAPSHOT_ENGINES}; pumping command queue every {POLL_INTERVAL}s")
    while not stop.is_set():
        try:
            server._drain_command_queue()
        except Exception:
            traceback.print_exc()
        time.sleep(POLL_INTERVAL)

    server.stop()
    log("stopped")


if __name__ == "__main__":
    main()
