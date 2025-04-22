from langchain_core.tools import BaseTool

from mcp_use.client import MCPClient
from mcp_use.logging import logger

from ..adapters.base import BaseAdapter
from .tools import (
    ConnectServerTool,
    DisconnectServerTool,
    GetActiveServerTool,
    ListServersTool,
    SearchToolsTool,
    UseToolFromServerTool,
)


class ServerManager:
    """Manages MCP servers and provides tools for server selection and management.

    This class allows an agent to discover and select which MCP server to use,
    dynamically activating the tools for the selected server.
    """

    def __init__(self, client: MCPClient, adapter: BaseAdapter) -> None:
        """Initialize the server manager.

        Args:
            client: The MCPClient instance managing server connections
            adapter: The LangChainAdapter for converting MCP tools to LangChain tools
        """
        self.client = client
        self.adapter = adapter
        self.active_server: str | None = None
        self.initialized_servers: dict[str, bool] = {}
        self._server_tools: dict[str, list[BaseTool]] = {}

    async def initialize(self) -> None:
        """Initialize the server manager and prepare server management tools."""
        # Make sure we have server configurations
        if not self.client.get_server_names():
            logger.warning("No MCP servers defined in client configuration")

    async def _prefetch_server_tools(self) -> None:
        """Pre-fetch tools for all servers to populate the tool search index."""
        servers = self.client.get_server_names()
        for server_name in servers:
            try:
                # Only create session if needed, don't set active
                session = None
                try:
                    session = self.client.get_session(server_name)
                    logger.debug(
                        f"Using existing session for server '{server_name}' to prefetch tools."
                    )
                except ValueError:
                    try:
                        session = await self.client.create_session(server_name)
                        logger.debug(
                            f"Temporarily created session for '{server_name}' to prefetch tools"
                        )
                    except Exception:
                        logger.warning(
                            f"Could not create session for '{server_name}' during prefetch"
                        )
                        continue

                # Fetch tools if session is available
                if session:
                    connector = session.connector
                    tools = await self.adapter._create_tools_from_connectors([connector])

                    # Check if this server's tools have changed
                    if (
                        server_name not in self._server_tools
                        or self._server_tools[server_name] != tools
                    ):
                        self._server_tools[server_name] = tools  # Cache tools
                        self.initialized_servers[server_name] = True  # Mark as initialized
                        logger.debug(f"Prefetched {len(tools)} tools for server '{server_name}'.")
                    else:
                        logger.debug(
                            f"Tools for server '{server_name}' unchanged, using cached version."
                        )
            except Exception as e:
                logger.error(f"Error prefetching tools for server '{server_name}': {e}")

    @property
    def tools(self) -> list[BaseTool]:
        """Get all server management tools.

        Returns:
            list of LangChain tools for server management
        """
        return [
            ListServersTool(self),
            ConnectServerTool(self),
            GetActiveServerTool(self),
            DisconnectServerTool(self),
            SearchToolsTool(self),
            UseToolFromServerTool(self),
        ]
