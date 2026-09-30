"""Config-driven headless agent benchmark: run an LLM agent on modeling tasks
(through an MCP server, this project's own tools, or any external agent CLI),
record its reasoning and tool calls, and score the model it produces.

    python -m src.agentbench check benchmarks/example/mcp_blender.json
    python -m src.agentbench run   benchmarks/example/mcp_blender.json
"""
