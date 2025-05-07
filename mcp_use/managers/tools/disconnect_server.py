from typing import ClassVar

from pydantic import BaseModel

from mcp_use.logging import logger

from .base_tool import MCPServerTool


class DisconnectServerInput(BaseModel):
    """Empty input for disconnecting from the current server"""

    pass


class DisconnectServerTool(MCPServerTool):
    """Tool for disconnecting from the currently active MCP server."""

    name: ClassVar[str] = "disconnect_from_mcp_server"
    description: ClassVar[str] = (
        "Disconnect from the currently active MCP (Model Context Protocol) server"
    )
    args_schema: ClassVar[type[BaseModel]] = DisconnectServerInput

    def _run(self, **kwargs) -> str:
        """Disconnect from the currently active MCP server."""
        if not self.server_manager.active_server:
            return "No MCP server is currently active, so there's nothing to disconnect from."

        server_name = self.server_manager.active_server
        try:
            # Clear the active server
            self.server_manager.active_server = None

            # Note: We're not actually closing the session here, just 'deactivating'
            # This way we keep the session cache without requiring reconnection if needed again

            return f"Successfully disconnected from MCP server '{server_name}'."
        except Exception as e:
            logger.error(f"Error disconnecting from server '{server_name}': {e}")
            return f"Failed to disconnect from server '{server_name}': {str(e)}"

    async def _arun(self, **kwargs) -> str:
        """Async implementation of _run."""
        return self._run(**kwargs)
