# Headless Blender + "MCP for Blender" served over Streamable HTTP, for remote agents.
#
#   docker compose build blender
#   docker compose up -d blender                # MCP endpoint: http://<host>:8000/mcp
#   docker compose logs -f blender
#
# What is inside:
#   * Blender from Debian trixie (4.3.x). The package is built for both amd64 and
#     arm64, so the image runs natively on x86 servers and on Apple-silicon Macs.
#     Blender's official tarballs are x64-only.
#   * mcp-for-blender (the MCP server half of https://github.com/ahujasid/blender-mcp)
#     in its own venv, plus the addon.py bundled with that release.
#   * headless_server.py: runs the addon's socket server inside `blender -b`.
#     The stock addon refuses background mode because it relies on bpy.app.timers,
#     which never fire without a window; the launcher pumps the command queue itself.
#   * mcp_http.py: starts the MCP server with the streamable-http transport
#     instead of stdio so agents on other machines can connect.
FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Debian's blender links against the system python3, so the addon's imports
# (requests, numpy) are satisfied by apt packages. libegl1 + mesa's software
# rasteriser give Blender a GL context in background mode, which the Workbench
# snapshot renderer needs (without libEGL Blender aborts instead of raising).
RUN apt-get update && apt-get install -y --no-install-recommends \
        blender \
        python3 python3-venv python3-requests python3-numpy \
        libegl1 libgl1-mesa-dri libgles2 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# MCP server half. Pin the release so the addon and server versions match.
ARG MCP_FOR_BLENDER_VERSION=2.1.0
RUN python3 -m venv /opt/mcp \
    && /opt/mcp/bin/pip install "mcp-for-blender==${MCP_FOR_BLENDER_VERSION}" \
    && cp /opt/mcp/lib/python3*/site-packages/blender_mcp/bundled/addon.py /opt/blender-mcp/addon.py 2>/dev/null \
       || (mkdir -p /opt/blender-mcp && cp /opt/mcp/lib/python3*/site-packages/blender_mcp/bundled/addon.py /opt/blender-mcp/addon.py)

COPY headless_server.py mcp_http.py entrypoint.sh /opt/blender-mcp/
RUN chmod +x /opt/blender-mcp/entrypoint.sh

WORKDIR /app
# models/ and outputs/ are bind-mounted from the host: export glb files into
# /app/models and score them with the benchmark container.
VOLUME ["/app/models", "/app/outputs"]

ENV BLENDER_HOST=127.0.0.1 \
    BLENDER_PORT=9876 \
    BLENDER_BIND_HOST=127.0.0.1 \
    MCP_HTTP_HOST=0.0.0.0 \
    MCP_HTTP_PORT=8000 \
    BLENDER_MCP_DISABLE_TELEMETRY=1

EXPOSE 8000
ENTRYPOINT ["/opt/blender-mcp/entrypoint.sh"]
