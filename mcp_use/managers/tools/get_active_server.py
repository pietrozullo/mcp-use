from typing import ClassVar

from pydantic import BaseModel

from .base_tool import MCPServerTool


class CurrentServerInput(BaseModel):
    """Empty input for checking current server"""

    pass


class GetActiveServerTool(MCPServerTool):
    """Tool for getting the currently active MCP server."""

    name: ClassVar[str] = "get_active_mcp_server"
    description: ClassVar[str] = "Get the currently active MCP (Model Context Protocol) server"
    args_schema: ClassVar[type[BaseModel]] = CurrentServerInput

    def _run(self, **kwargs) -> str:
        """Get the currently active MCP server."""
        if not self.server_manager.active_server:
            return (
                "No MCP server is currently active. "
                "Use connect_to_mcp_server to connect to a server."
            )
        return f"Currently active MCP server: {self.server_manager.active_server}"

    async def _arun(self, **kwargs) -> str:
        """Async implementation of _run."""
        return self._run(**kwargs)
