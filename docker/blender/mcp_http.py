"""Serve "MCP for Blender" over Streamable HTTP instead of stdio.

The upstream entry point (`mcp-for-blender`) only speaks stdio, which is fine
when the MCP client launches the server itself. Remote agents need a network
endpoint, so this wrapper reuses the same FastMCP instance and runs it with the
streamable-http transport at http://<host>:<port>/mcp.

Environment:
  MCP_HTTP_HOST      bind address (default 0.0.0.0)
  MCP_HTTP_PORT      port (default 8000)
  MCP_ALLOWED_HOSTS  comma-separated Host header patterns, e.g.
                     "myserver.example.com:*,10.0.0.5:*". When set, FastMCP's
                     DNS-rebinding protection is enabled and only those hosts are
                     accepted. Unset (default) accepts any Host header.
  BLENDER_HOST/BLENDER_PORT  where the Blender addon socket lives (upstream vars).
"""
import logging
import os

from mcp.server.transport_security import TransportSecuritySettings
import blender_mcp
from blender_mcp import server as blender_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mcp_http")

host = os.environ.get("MCP_HTTP_HOST", "0.0.0.0")
port = int(os.environ.get("MCP_HTTP_PORT", "8000"))

mcp = blender_server.mcp
mcp.settings.host = host
mcp.settings.port = port
mcp.settings.streamable_http_path = "/mcp"

# FastMCP auto-enables DNS-rebinding protection (localhost-only Host headers)
# because the upstream server is constructed with the default 127.0.0.1 host.
# That would reject every remote request with 421, so replace it.
allowed = [h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
if allowed:
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed,
        allowed_origins=[f"http://{h}" for h in allowed] + [f"https://{h}" for h in allowed],
    )
else:
    mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

log.info(
    "MCP for Blender %s listening on http://%s:%d/mcp (Blender socket %s:%s)",
    getattr(blender_mcp, "__version__", "?"),
    host, port,
    os.environ.get("BLENDER_HOST", "localhost"), os.environ.get("BLENDER_PORT", "9876"),
)
mcp.run(transport="streamable-http")
