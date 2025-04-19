"""
Base adapter interface for MCP tools.

This module provides the abstract base class that all MCP tool adapters should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from ..client import MCPClient
from ..connectors.base import BaseConnector
from ..logging import logger

# Generic type for the tools created by the adapter
T = TypeVar("T")


class BaseAdapter(ABC):
    """Abstract base class for converting MCP tools to other framework formats.

    This class defines the common interface that all adapter implementations
    should follow to ensure consistency across different frameworks.
    """

    def __init__(self, disallowed_tools: list[str] | None = None) -> None:
        """Initialize a new adapter.

        Args:
            disallowed_tools: list of tool names that should not be available.
        """
        self.disallowed_tools = disallowed_tools or []
        self._connector_tool_map: dict[BaseConnector, list[T]] = {}

    @classmethod
    async def create_tools(
        cls, client: "MCPClient", disallowed_tools: list[str] | None = None
    ) -> list[T]:
        """Create tools from an MCPClient instance.

        This is the recommended way to create tools from an MCPClient, as it handles
        session creation and connector extraction automatically.

        Args:
            client: The MCPClient to extract tools from.
            disallowed_tools: Optional list of tool names to exclude.

        Returns:
            A list of tools in the target framework's format.

        Example:
            ```python
            from mcp_use.client import MCPClient
            from mcp_use.adapters import YourAdapter

            client = MCPClient.from_config_file("config.json")
            tools = await YourAdapter.create_tools(client)
            ```
        """
        # Create the adapter
        adapter = cls(disallowed_tools=disallowed_tools)

        # Ensure we have active sessions
        if not client.active_sessions:
            logger.info("No active sessions found, creating new ones...")
            await client.create_all_sessions()

        # Get all active sessions
        sessions = client.get_all_active_sessions()

        # Extract connectors from sessions
        connectors = [session.connector for session in sessions.values()]

        # Create tools from connectors
        return await adapter._create_tools_from_connectors(connectors)

    async def load_tools_for_connector(self, connector: BaseConnector) -> list[T]:
        """Dynamically load tools for a specific connector.

        Args:
            connector: The connector to load tools for.

        Returns:
            The list of tools that were loaded in the target framework's format.
        """
        # Check if we already have tools for this connector
        if connector in self._connector_tool_map:
            logger.debug(
                f"Returning {len(self._connector_tool_map[connector])} existing tools for connector"
            )
            return self._connector_tool_map[connector]

        # Create tools for this connector
        connector_tools = []

        # Make sure the connector is initialized and has tools
        success = await self._ensure_connector_initialized(connector)
        if not success:
            return []

        # Now create tools for each MCP tool
        for tool in connector.tools:
            # Convert the tool and add it to the list if conversion was successful
            converted_tool = self._convert_tool(tool, connector)
            if converted_tool:
                connector_tools.append(converted_tool)

        # Store the tools for this connector
        self._connector_tool_map[connector] = connector_tools

        # Log available tools for debugging
        logger.debug(
            f"Loaded {len(connector_tools)} new tools for connector: "
            f"{[getattr(tool, 'name', str(tool)) for tool in connector_tools]}"
        )

        return connector_tools

    @abstractmethod
    def _convert_tool(self, mcp_tool: dict[str, Any], connector: BaseConnector) -> T:
        """Convert an MCP tool to the target framework's tool format.

        Args:
            mcp_tool: The MCP tool to convert.
            connector: The connector that provides this tool.

        Returns:
            A tool in the target framework's format.
        """
        pass

    async def _create_tools_from_connectors(self, connectors: list[BaseConnector]) -> list[T]:
        """Create tools from MCP tools in all provided connectors.

        Args:
            connectors: list of MCP connectors to create tools from.

        Returns:
            A list of tools in the target framework's format.
        """
        tools = []
        for connector in connectors:
            # Create tools for this connector
            connector_tools = await self.load_tools_for_connector(connector)
            tools.extend(connector_tools)

        # Log available tools for debugging
        logger.debug(f"Available tools: {len(tools)}")
        return tools

    def _check_connector_initialized(self, connector: BaseConnector) -> bool:
        """Check if a connector is initialized and has tools.

        Args:
            connector: The connector to check.

        Returns:
            True if the connector is initialized and has tools, False otherwise.
        """
        return hasattr(connector, "tools") and connector.tools

    async def _ensure_connector_initialized(self, connector: BaseConnector) -> bool:
        """Ensure a connector is initialized.

        Args:
            connector: The connector to initialize.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if not self._check_connector_initialized(connector):
            logger.debug("Connector doesn't have tools, initializing it")
            try:
                await connector.initialize()
                return True
            except Exception as e:
                logger.error(f"Error initializing connector: {e}")
                return False
        return True
