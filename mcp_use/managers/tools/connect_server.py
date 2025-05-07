from typing import ClassVar

from pydantic import BaseModel, Field

from mcp_use.logging import logger

from .base_tool import MCPServerTool


class ServerActionInput(BaseModel):
    """Base input for server-related actions"""

    server_name: str = Field(description="The name of the MCP server")


class ConnectServerTool(MCPServerTool):
    """Tool for connecting to a specific MCP server."""

    name: ClassVar[str] = "connect_to_mcp_server"
    description: ClassVar[str] = (
        "Connect to a specific MCP (Model Context Protocol) server to use its "
        "tools. Use this tool to connect to a specific server and use its tools."
    )
    args_schema: ClassVar[type[BaseModel]] = ServerActionInput

    async def _arun(self, server_name: str) -> str:
        """Connect to a specific MCP server."""
        # Check if server exists
        servers = self.server_manager.client.get_server_names()
        if server_name not in servers:
            available = ", ".join(servers) if servers else "none"
            return f"Server '{server_name}' not found. Available servers: {available}"

        # If we're already connected to this server, just return
        if self.server_manager.active_server == server_name:
            return f"Already connected to MCP server '{server_name}'"

        try:
            # Create or get session for this server
            try:
                session = self.server_manager.client.get_session(server_name)
                logger.debug(f"Using existing session for server '{server_name}'")
            except ValueError:
                logger.debug(f"Creating new session for server '{server_name}'")
                session = await self.server_manager.client.create_session(server_name)

            # Set as active server
            self.server_manager.active_server = server_name

            # Initialize server tools if not already initialized
            if server_name not in self.server_manager._server_tools:
                connector = session.connector
                self.server_manager._server_tools[
                    server_name
                ] = await self.server_manager.adapter._create_tools_from_connectors([connector])
                self.server_manager.initialized_servers[server_name] = True

            server_tools = self.server_manager._server_tools.get(server_name, [])
            num_tools = len(server_tools)

            return f"Connected to MCP server '{server_name}'. {num_tools} tools are now available."

        except Exception as e:
            logger.error(f"Error connecting to server '{server_name}': {e}")
            return f"Failed to connect to server '{server_name}': {str(e)}"

    def _run(self, server_name: str) -> str:
        """Synchronous version that raises a NotImplementedError - use _arun instead."""
        raise NotImplementedError("ConnectServerTool requires async execution. Use _arun instead.")
