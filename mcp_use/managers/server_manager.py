import asyncio
import time

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from mcp_use.client import MCPClient
from mcp_use.logging import logger

from ..adapters.base import BaseAdapter
from .tool_search import ToolSearch, ToolSearchInput


class ServerActionInput(BaseModel):
    """Base input for server-related actions"""

    server_name: str = Field(description="The name of the MCP server")


class DisconnectServerInput(BaseModel):
    """Empty input for disconnecting from the current server"""

    pass


class listServersInput(BaseModel):
    """Empty input for listing available servers"""

    pass


class CurrentServerInput(BaseModel):
    """Empty input for checking current server"""

    pass


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
        self.tool_search = ToolSearch()

        # Track when tools were last indexed to avoid redundant indexing
        self._last_index_time: float = 0
        self._indexed_tool_count: int = 0
        self._indexed_server_names: set[str] = set()
        self._index_in_progress: bool = False

        # Task to track background indexing
        self._indexing_task = None

    async def initialize(self) -> None:
        """Initialize the server manager and prepare server management tools."""
        # Make sure we have server configurations
        if not self.client.get_server_names():
            logger.warning("No MCP servers defined in client configuration")

        # Pre-fetch tools for all servers to populate the tool search index
        await self._prefetch_server_tools()

    async def _prefetch_server_tools(self) -> None:
        """Pre-fetch tools for all servers to populate the tool search index."""
        servers = self.client.get_server_names()

        tool_changes = False
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
                        tool_changes = True
                        logger.debug(f"Prefetched {len(tools)} tools for server '{server_name}'.")
                    else:
                        logger.debug(
                            f"Tools for server '{server_name}' unchanged, using cached version."
                        )
            except Exception as e:
                logger.error(f"Error prefetching tools for server '{server_name}': {e}")

        # Only update the index if tools have changed
        if tool_changes:
            # Start indexing in the background
            self._start_background_indexing()
        else:
            logger.info("No tool changes detected, search index remains current")

    def _start_background_indexing(self) -> None:
        """Start the index update process in the background."""
        if self._indexing_task is not None and not self._indexing_task.done():
            logger.debug("Indexing task already running")
            return

        self._indexing_task = asyncio.create_task(self._update_search_index())
        logger.debug("Started background indexing task")

    async def wait_for_index_update(self) -> None:
        """Wait for any ongoing index update to complete."""
        if self._indexing_task is not None and not self._indexing_task.done():
            logger.debug("Waiting for index update to complete...")
            await self._indexing_task
            logger.debug("Index update completed")

    async def _update_search_index(self) -> None:
        """Update the tool search index, avoiding redundant indexing."""
        # If indexing is already in progress, don't start another index operation
        if self._index_in_progress:
            logger.debug("Tool indexing already in progress, skipping redundant indexing")
            return

        try:
            # Set flag to prevent concurrent indexing
            self._index_in_progress = True

            # Calculate a signature of the current tool set
            current_tool_count = sum(len(tools) for tools in self._server_tools.values())
            current_server_names = set(self._server_tools.keys())

            # Check if index needs to be updated
            index_needs_update = (
                current_tool_count != self._indexed_tool_count
                or current_server_names != self._indexed_server_names
                or (time.time() - self._last_index_time) > 3600  # Force reindex after 1 hour
            )

            if index_needs_update:
                logger.info(
                    f"Updating tool search index with tools from {len(self._server_tools)} servers"
                )
                await self.tool_search.index_tools(self._server_tools)

                # Update tracking info
                self._last_index_time = time.time()
                self._indexed_tool_count = current_tool_count
                self._indexed_server_names = current_server_names

                # Get stats for logging
                stats = self.tool_search.get_stats()
                logger.info(
                    f"Tool search index updated with {stats['indexed_tools']} tools "
                    f"in {stats['index_build_time']:.2f}s"
                )
            else:
                logger.debug("Tool search index is up to date, skipping reindexing")

        finally:
            # Clear flag when done
            self._index_in_progress = False

    async def get_server_management_tools(self) -> list[BaseTool]:
        """Get tools for managing server connections.

        Returns:
            list of LangChain tools for server management
        """
        # Create structured tools for server management with direct parameter passing
        list_servers_tool = StructuredTool.from_function(
            coroutine=self.list_servers,
            name="list_mcp_servers",
            description="lists all available MCP (Model Context Protocol) servers that can be "
            "connected to, along with the tools available on each server. "
            "Use this tool to discover servers and see what functionalities they offer.",
            args_schema=listServersInput,
        )

        connect_server_tool = StructuredTool.from_function(
            coroutine=self.connect_to_server,
            name="connect_to_mcp_server",
            description="Connect to a specific MCP (Model Context Protocol) server to use its "
            "tools. Use this tool to connect to a specific server and use its tools.",
            args_schema=ServerActionInput,
        )

        get_active_server_tool = StructuredTool.from_function(
            coroutine=self.get_active_server,
            name="get_active_mcp_server",
            description="Get the currently active MCP (Model Context Protocol) server",
            args_schema=CurrentServerInput,
        )

        disconnect_server_tool = StructuredTool.from_function(
            func=None,
            coroutine=self.disconnect_from_server,
            name="disconnect_from_mcp_server",
            description="Disconnect from the currently active MCP (Model Context Protocol) server",
            args_schema=DisconnectServerInput,
        )

        search_tools_tool = StructuredTool.from_function(
            coroutine=self.search_tools,
            name="search_mcp_tools",
            description=(
                "Search for relevant tools across all MCP servers using semantic search. "
                "Provide a description of the tool you think you might need to be able to perform "
                "the task you are assigned. Do not be too specific, the search will give you many"
                " options. It is important you search for the tool, not for the goal."
                "If your first search doesn't yield relevant results, try using different keywords "
                "or more general terms."
            ),
            args_schema=ToolSearchInput,
        )

        return [
            list_servers_tool,
            connect_server_tool,
            get_active_server_tool,
            disconnect_server_tool,
            search_tools_tool,
        ]

    async def list_servers(self) -> str:
        """list all available MCP servers along with their available tools.

        Returns:
            String listing all available servers and their tools.
        """
        servers = self.client.get_server_names()
        if not servers:
            return "No MCP servers are currently defined."

        result = "Available MCP servers:\n"
        for i, server_name in enumerate(servers):
            active_marker = " (ACTIVE)" if server_name == self.active_server else ""
            result += f"{i + 1}. {server_name}{active_marker}\n"

            tools: list[BaseTool] = []
            try:
                # Check cache first
                if server_name in self._server_tools:
                    tools = self._server_tools[server_name]
                else:
                    # Attempt to get/create session without setting active
                    session = None
                    try:
                        session = self.client.get_session(server_name)
                        logger.debug(
                            f"Using existing session for server '{server_name}' to list tools."
                        )
                    except ValueError:
                        try:
                            # Only create session if needed, don't set active
                            session = await self.client.create_session(server_name)
                            logger.debug(f"Temporarily created session for server '{server_name}'")
                        except Exception:
                            logger.warning(f"Could not create session for server '{server_name}'")

                    # Fetch tools if session is available
                    if session:
                        try:
                            connector = session.connector
                            fetched_tools = await self.adapter._create_tools_from_connectors(
                                [connector]
                            )
                            self._server_tools[server_name] = fetched_tools  # Cache tools
                            self.initialized_servers[server_name] = True  # Mark as initialized
                            tools = fetched_tools
                            logger.debug(f"Fetched {len(tools)} tools for server '{server_name}'.")

                            # Check if tools have changed
                            if (
                                server_name not in self._server_tools
                                or self._server_tools[server_name] != fetched_tools
                            ):
                                self._server_tools[server_name] = fetched_tools  # Cache tools
                                self.initialized_servers[server_name] = True  # Mark as initialized
                                tools = fetched_tools
                                logger.debug(
                                    f"Fetched {len(tools)} tools for server '{server_name}'."
                                )

                                # Update the tool search index with the new tools
                                await self._update_search_index()
                            else:
                                tools = fetched_tools
                                logger.debug(f"Tools for server '{server_name}' unchanged.")

                        except Exception as e:
                            logger.warning(f"Could not fetch tools for server '{server_name}': {e}")

            except Exception as e:
                logger.error(f"Unexpected error listing tools for server '{server_name}': {e}")

        return result

    async def connect_to_server(self, server_name: str) -> str:
        """Connect to a specific MCP server.

        Args:
            server_name: The name of the server to connect to

        Returns:
            Status message about the connection
        """
        # Check if server exists
        servers = self.client.get_server_names()
        if server_name not in servers:
            available = ", ".join(servers) if servers else "none"
            return f"Server '{server_name}' not found. Available servers: {available}"

        # If we're already connected to this server, just return
        if self.active_server == server_name:
            return f"Already connected to MCP server '{server_name}'"

        try:
            # Create or get session for this server
            try:
                session = self.client.get_session(server_name)
                logger.debug(f"Using existing session for server '{server_name}'")
            except ValueError:
                logger.debug(f"Creating new session for server '{server_name}'")
                session = await self.client.create_session(server_name)

            # Set as active server
            self.active_server = server_name

            # Initialize server tools if not already initialized
            tool_changes = False
            if server_name not in self._server_tools:
                connector = session.connector
                self._server_tools[server_name] = await self.adapter._create_tools_from_connectors(
                    [connector]
                )
                self.initialized_servers[server_name] = True
                tool_changes = True

                # Only update the search index if we have new tools
                if tool_changes:
                    self._start_background_indexing()

            server_tools = self._server_tools.get(server_name, [])
            num_tools = len(server_tools)

            tool_descriptions = "\nAvailable tools for this server:\n"
            for i, tool in enumerate(server_tools):
                tool_descriptions += f"{i + 1}. {tool.name}: {tool.description}\n"

            return (
                f"Connected to MCP server '{server_name}'. "
                f"{num_tools} tools are now available."
                f"{tool_descriptions}"
            )

        except Exception as e:
            logger.error(f"Error connecting to server '{server_name}': {e}")
            return f"Failed to connect to server '{server_name}': {str(e)}"

    async def get_active_server(self) -> str:
        """Get the currently active MCP server.

        Returns:
            Name of the active server or message if none is active
        """
        if not self.active_server:
            return (
                "No MCP server is currently active. "
                "Use connect_to_mcp_server to connect to a server."
            )
        return f"Currently active MCP server: {self.active_server}"

    async def disconnect_from_server(self) -> str:
        """Disconnect from the currently active MCP server.

        Returns:
            Status message about the disconnection
        """

        if not self.active_server:
            return "No MCP server is currently active, so there's nothing to disconnect from."

        server_name = self.active_server
        try:
            # Clear the active server
            self.active_server = None

            # Note: We're not actually closing the session here, just 'deactivating'
            # This way we keep the session cache without requiring reconnection if needed again
            # TODO: consider closing the sessions

            return f"Successfully disconnected from MCP server '{server_name}'."
        except Exception as e:
            logger.error(f"Error disconnecting from server '{server_name}': {e}")
            return f"Failed to disconnect from server '{server_name}': {str(e)}"

    async def get_active_server_tools(self) -> list[BaseTool]:
        """Get the tools for the currently active server.

        Returns:
            list of LangChain tools for the active server or empty list if no active server
        """
        if not self.active_server:
            return []

        return self._server_tools.get(self.active_server, [])

    async def get_all_tools(self) -> list[BaseTool]:
        """Get all tools - both server management tools and tools for the active server.

        Returns:
            Combined list of server management tools and active server tools
        """
        management_tools = await self.get_server_management_tools()
        active_server_tools = await self.get_active_server_tools()
        return management_tools + active_server_tools

    async def search_tools(self, query: str, top_k: int = 100) -> str:
        """Search for tools across all MCP servers using semantic search.

        Args:
            query: The search query to find relevant tools
            top_k: Number of top results to return
            use_fast_search: Whether to use fast keyword search instead of semantic search

        Returns:
            String with formatted search results
        """
        # If we have no tools indexed, try to prefetch them
        if not self._server_tools:
            logger.info("No tools found in index, attempting to prefetch server tools")
            await self._prefetch_server_tools()
        # Wait for any ongoing indexing to complete since we need the index now
        await self.wait_for_index_update()

        search_method = "hybrid"

        # Log search attempt with tool index stats
        tool_count = sum(len(tools) for tools in self._server_tools.values())
        logger.info(
            f"Searching {tool_count} tools with query: '{query}' using {search_method} search"
        )

        results = self.tool_search.search(query, top_k=top_k)
        if not results:
            logger.warning(f"No tool search results found for query: '{query}'")
            return (
                "No relevant tools found. The search provided no results"
                " You can also try searching again with different keywords."
                " Try using more general terms or focusing on the capability you need rather than"
                "specific implementation details."
            )

        # Format the results
        response = f"Top {len(results)} tools matching '{query}'\n\n"

        for i, (tool, server_name, score) in enumerate(results):
            score_percent = f"{score * 100:.1f}%"
            active_marker = " (ACTIVE)" if server_name == self.active_server else ""
            response += (
                f"{i + 1}. Tool: {tool.name} (Relevance: {score_percent})\n"
                f"   Server: {server_name}{active_marker}\n"
                f"   Description: {tool.description}\n\n"
            )

        response += "\nIf these tools don't match what you're looking for, try searching again with"
        " different keywords. Consider trying broader terms (e.g., 'image' instead of 'resize image"
        "') or focusing on the core functionality needed."

        return response
