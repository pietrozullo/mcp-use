from typing import ClassVar

from pydantic import BaseModel

from mcp_use.logging import logger

from .base_tool import MCPServerTool


class listServersInput(BaseModel):
    """Empty input for listing available servers"""

    pass


class ListServersTool(MCPServerTool):
    """Tool for listing available MCP servers."""

    name: ClassVar[str] = "list_mcp_servers"
    description: ClassVar[str] = (
        "Lists all available MCP (Model Context Protocol) servers that can be "
        "connected to, along with the tools available on each server. "
        "Use this tool to discover servers and see what functionalities they offer."
    )
    args_schema: ClassVar[type[BaseModel]] = listServersInput

    def _run(self, **kwargs) -> str:
        """List all available MCP servers along with their available tools."""
        servers = self.server_manager.client.get_server_names()
        if not servers:
            return "No MCP servers are currently defined."

        result = "Available MCP servers:\n"
        for i, server_name in enumerate(servers):
            active_marker = " (ACTIVE)" if server_name == self.server_manager.active_server else ""
            result += f"{i + 1}. {server_name}{active_marker}\n"

            tools: list = []
            try:
                # Check cache first
                if server_name in self.server_manager._server_tools:
                    tools = self.server_manager._server_tools[server_name]
                    tool_count = len(tools)
                    result += f"   {tool_count} tools available for this server\n"
            except Exception as e:
                logger.error(f"Unexpected error listing tools for server '{server_name}': {e}")

        return result

    async def _arun(self, **kwargs) -> str:
        """Async implementation of _run - calls the synchronous version."""
        return self._run(**kwargs)
