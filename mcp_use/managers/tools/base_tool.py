from typing import ClassVar

from langchain_core.tools import BaseTool


class MCPServerTool(BaseTool):
    """Base tool for MCP server operations."""

    name: ClassVar[str] = "mcp_server_tool"
    description: ClassVar[str] = "Base tool for MCP server operations."

    def __init__(self, server_manager):
        """Initialize with server manager."""
        super().__init__()
        self._server_manager = server_manager

    @property
    def server_manager(self):
        return self._server_manager
