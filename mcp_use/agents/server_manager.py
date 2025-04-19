from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from mcp_use.client import MCPClient
from mcp_use.logging import logger

from ..adapters.langchain_adapter import LangChainAdapter


class ServerActionInput(BaseModel):
    """Base input for server-related actions"""

    server_name: str = Field(description="The name of the MCP server")


class DisconnectServerInput(BaseModel):
    """Empty input for disconnecting from the current server"""

    pass


class ListServersInput(BaseModel):
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

    def __init__(self, client: MCPClient, adapter: LangChainAdapter) -> None:
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

    async def get_server_management_tools(self) -> list[BaseTool]:
        """Get tools for managing server connections.

        Returns:
            List of LangChain tools for server management
        """
        # Create structured tools for server management with direct parameter passing
        list_servers_tool = StructuredTool.from_function(
            coroutine=self.list_servers,
            name="list_mcp_servers",
            description="Lists all available MCP (Model Context Protocol) servers that can be "
            "connected to, along with the tools available on each server. "
            "Use this tool to discover servers and see what functionalities they offer.",
            args_schema=ListServersInput,
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

        return [
            list_servers_tool,
            connect_server_tool,
            get_active_server_tool,
            disconnect_server_tool,
        ]

    async def list_servers(self) -> str:
        """List all available MCP servers along with their available tools.

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
                        except Exception as e:
                            logger.warning(f"Could not fetch tools for server '{server_name}': {e}")
                            # Keep tools as empty list if fetching failed

                # Append tool names to the result string
                if tools:
                    tool_names = ", ".join([tool.name for tool in tools])
                    result += f"   Tools: {tool_names}\n"
                else:
                    result += "   Tools: (Could not retrieve or none available)\n"

            except Exception as e:
                logger.error(f"Unexpected error listing tools for server '{server_name}': {e}")
                result += "   Tools: (Error retrieving tools)\n"

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
            if server_name not in self._server_tools:
                connector = session.connector
                self._server_tools[server_name] = await self.adapter._create_tools_from_connectors(
                    [connector]
                )
                self.initialized_servers[server_name] = True

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
            List of LangChain tools for the active server or empty list if no active server
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
