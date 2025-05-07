"""
Base connector for MCP implementations.

This module provides the base connector interface that all MCP connectors
must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, GetPromptResult, Prompt, ReadResourceResult, Resource, Tool

from ..logging import logger
from ..task_managers import ConnectionManager


class BaseConnector(ABC):
    """Base class for MCP connectors.

    This class defines the interface that all MCP connectors must implement.
    """

    def __init__(self):
        """Initialize base connector with common attributes."""
        self.client: ClientSession | None = None
        self._connection_manager: ConnectionManager | None = None
        self._tools: list[Tool] | None = None
        self._resources: list[Resource] | None = None
        self._prompts: list[Prompt] | None = None
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish a connection to the MCP implementation."""
        pass

    async def disconnect(self) -> None:
        """Close the connection to the MCP implementation."""
        if not self._connected:
            logger.debug("Not connected to MCP implementation")
            return

        logger.debug("Disconnecting from MCP implementation")
        await self._cleanup_resources()
        self._connected = False
        logger.debug("Disconnected from MCP implementation")

    async def _cleanup_resources(self) -> None:
        """Clean up all resources associated with this connector."""
        errors = []

        # First close the client session
        if self.client:
            try:
                logger.debug("Closing client session")
                await self.client.__aexit__(None, None, None)
            except Exception as e:
                error_msg = f"Error closing client session: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
            finally:
                self.client = None

        # Then stop the connection manager
        if self._connection_manager:
            try:
                logger.debug("Stopping connection manager")
                await self._connection_manager.stop()
            except Exception as e:
                error_msg = f"Error stopping connection manager: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
            finally:
                self._connection_manager = None

        # Reset tools
        self._tools = None
        self._resources = None
        self._prompts = None

        if errors:
            logger.warning(f"Encountered {len(errors)} errors during resource cleanup")

    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP session and return session information."""
        if not self.client:
            raise RuntimeError("MCP client is not connected")

        logger.debug("Initializing MCP session")

        # Initialize the session
        result = await self.client.initialize()

        server_capabilities = result.capabilities

        if server_capabilities.tools:
            # Get available tools
            tools_result = await self.list_tools()
            self._tools = tools_result or []
        else:
            self._tools = []

        if server_capabilities.resources:
            # Get available resources
            resources_result = await self.list_resources()
            self._resources = resources_result or []
        else:
            self._resources = []

        if server_capabilities.prompts:
            # Get available prompts
            prompts_result = await self.list_prompts()
            self._prompts = prompts_result or []
        else:
            self._prompts = []

        logger.debug(
            f"MCP session initialized with {len(self._tools)} tools, "
            f"{len(self._resources)} resources, "
            f"and {len(self._prompts)} prompts"
        )

        return result

    @property
    def tools(self) -> list[Tool]:
        """Get the list of available tools."""
        if self._tools is None:
            raise RuntimeError("MCP client is not initialized")
        return self._tools

    @property
    def resources(self) -> list[Resource]:
        """Get the list of available resources."""
        if self._resources is None:
            raise RuntimeError("MCP client is not initialized")
        return self._resources

    @property
    def prompts(self) -> list[Prompt]:
        """Get the list of available prompts."""
        if self._prompts is None:
            raise RuntimeError("MCP client is not initialized")
        return self._prompts

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Call an MCP tool with the given arguments."""
        if not self.client:
            raise RuntimeError("MCP client is not connected")

        logger.debug(f"Calling tool '{name}' with arguments: {arguments}")
        result = await self.client.call_tool(name, arguments)
        logger.debug(f"Tool '{name}' called with result: {result}")
        return result

    async def list_tools(self) -> list[Tool]:
        """List all available tools from the MCP implementation."""
        if not self.client:
            raise RuntimeError("MCP client is not connected")

        logger.debug("Listing tools")
        try:
            result = await self.client.list_tools()
            return result.tools
        except McpError as e:
            logger.error(f"Error listing tools: {e}")
            return []

    async def list_resources(self) -> list[Resource]:
        """List all available resources from the MCP implementation."""
        if not self.client:
            raise RuntimeError("MCP client is not connected")

        logger.debug("Listing resources")
        try:
            result = await self.client.list_resources()
            return result.resources
        except McpError as e:
            logger.error(f"Error listing resources: {e}")
            return []

    async def read_resource(self, uri: str) -> ReadResourceResult:
        """Read a resource by URI."""
        if not self.client:
            raise RuntimeError("MCP client is not connected")

        logger.debug(f"Reading resource: {uri}")
        result = await self.client.read_resource(uri)
        return result

    async def list_prompts(self) -> list[Prompt]:
        """List all available prompts from the MCP implementation."""
        if not self.client:
            raise RuntimeError("MCP client is not connected")

        logger.debug("Listing prompts")
        try:
            result = await self.client.list_prompts()
            return result.prompts
        except McpError as e:
            logger.error(f"Error listing prompts: {e}")
            return []

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        """Get a prompt by name."""
        if not self.client:
            raise RuntimeError("MCP client is not connected")

        logger.debug(f"Getting prompt: {name}")
        result = await self.client.get_prompt(name, arguments)
        return result

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a raw request to the MCP implementation."""
        if not self.client:
            raise RuntimeError("MCP client is not connected")

        logger.debug(f"Sending request: {method} with params: {params}")
        return await self.client.request({"method": method, "params": params or {}})
