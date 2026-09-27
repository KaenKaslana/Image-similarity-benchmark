#!/bin/bash
# Start headless Blender (addon socket server) and the MCP HTTP server in one
# container. They share localhost and /tmp, which get_viewport_screenshot needs.
# Any arguments run instead (e.g. `docker compose run blender bash`).
set -euo pipefail

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

cd /app
blender -b --python-exit-code 1 --python /opt/blender-mcp/headless_server.py &
BLENDER_PID=$!

cleanup() {
    kill -TERM "$BLENDER_PID" "${MCP_PID:-}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "entrypoint: waiting for Blender on ${BLENDER_HOST}:${BLENDER_PORT}"
for _ in $(seq 1 90); do
    if python3 - "$BLENDER_HOST" "$BLENDER_PORT" <<'PY' 2>/dev/null; then break; fi
import socket, sys
socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1).close()
PY
    if ! kill -0 "$BLENDER_PID" 2>/dev/null; then
        echo "entrypoint: Blender exited before its socket server came up" >&2
        exit 1
    fi
    sleep 1
done

/opt/mcp/bin/python /opt/blender-mcp/mcp_http.py &
MCP_PID=$!

# Exit as soon as either process dies so the orchestrator can restart us.
wait -n "$BLENDER_PID" "$MCP_PID"
