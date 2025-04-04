"""
Unit tests for the MCPSession class.
"""

import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from mcp_use.session import MCPSession


class TestMCPSessionInitialization(unittest.TestCase):
    """Tests for MCPSession initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        connector = MagicMock()
        session = MCPSession(connector)

        self.assertEqual(session.connector, connector)
        self.assertIsNone(session.session_info)
        self.assertEqual(session.tools, [])
        self.assertTrue(session.auto_connect)

    def test_init_with_auto_connect_false(self):
        """Test initialization with auto_connect set to False."""
        connector = MagicMock()
        session = MCPSession(connector, auto_connect=False)

        self.assertEqual(session.connector, connector)
        self.assertIsNone(session.session_info)
        self.assertEqual(session.tools, [])
        self.assertFalse(session.auto_connect)


class TestMCPSessionConnection(IsolatedAsyncioTestCase):
    """Tests for MCPSession connection methods."""

    def setUp(self):
        """Set up a session with a mock connector for each test."""
        self.connector = MagicMock()
        self.connector.connect = AsyncMock()
        self.connector.disconnect = AsyncMock()

        # By default, the connector is not connected
        type(self.connector).client = PropertyMock(return_value=None)

        self.session = MCPSession(self.connector)

    async def test_connect(self):
        """Test connecting to the MCP implementation."""
        await self.session.connect()
        self.connector.connect.assert_called_once()

    async def test_disconnect(self):
        """Test disconnecting from the MCP implementation."""
        await self.session.disconnect()
        self.connector.disconnect.assert_called_once()

    async def test_async_context_manager(self):
        """Test using the session as an async context manager."""
        async with self.session as session:
            self.assertEqual(session, self.session)
            self.connector.connect.assert_called_once()

        self.connector.disconnect.assert_called_once()

    async def test_is_connected_property(self):
        """Test the is_connected property."""
        # Test when not connected
        self.assertFalse(self.session.is_connected)

        # Test when connected
        type(self.connector).client = PropertyMock(return_value=MagicMock())
        self.assertTrue(self.session.is_connected)


class TestMCPSessionOperations(IsolatedAsyncioTestCase):
    """Tests for MCPSession operations."""

    def setUp(self):
        """Set up a session with a mock connector for each test."""
        self.connector = MagicMock()
        self.connector.connect = AsyncMock()
        self.connector.disconnect = AsyncMock()
        self.connector.initialize = AsyncMock(return_value={"session_id": "test_session"})
        self.connector.tools = [{"name": "test_tool"}]
        self.connector.call_tool = AsyncMock(return_value={"result": "success"})

        # By default, the connector is not connected
        type(self.connector).client = PropertyMock(return_value=None)

        self.session = MCPSession(self.connector)

    async def test_initialize(self):
        """Test initializing the session."""
        # Test initialization when not connected
        result = await self.session.initialize()

        # Verify connect was called since auto_connect is True
        self.connector.connect.assert_called_once()
        self.connector.initialize.assert_called_once()

        # Verify session_info was set
        self.assertEqual(self.session.session_info, {"session_id": "test_session"})
        self.assertEqual(result, {"session_id": "test_session"})

        # Verify tools were discovered
        self.assertEqual(self.session.tools, [{"name": "test_tool"}])

    async def test_initialize_already_connected(self):
        """Test initializing the session when already connected."""
        # Set up the connector to indicate it's already connected
        type(self.connector).client = PropertyMock(return_value=MagicMock())

        # Test initialization when already connected
        await self.session.initialize()

        # Verify connect was not called since already connected
        self.connector.connect.assert_not_called()
        self.connector.initialize.assert_called_once()

    async def test_discover_tools(self):
        """Test discovering available tools."""
        tools = await self.session.discover_tools()

        # Verify tools were set correctly
        self.assertEqual(tools, [{"name": "test_tool"}])
        self.assertEqual(self.session.tools, [{"name": "test_tool"}])

    async def test_call_tool_connected(self):
        """Test calling a tool when already connected."""
        # Set up the connector to indicate it's already connected
        type(self.connector).client = PropertyMock(return_value=MagicMock())

        # Call the tool
        result = await self.session.call_tool("test_tool", {"param": "value"})

        # Verify the connector's call_tool method was called with the right arguments
        self.connector.call_tool.assert_called_once_with("test_tool", {"param": "value"})

        # Verify the result is correct
        self.assertEqual(result, {"result": "success"})

        # Verify connect was not called since already connected
        self.connector.connect.assert_not_called()

    async def test_call_tool_not_connected(self):
        """Test calling a tool when not connected."""
        # Call the tool
        result = await self.session.call_tool("test_tool", {"param": "value"})

        # Verify connect was called since auto_connect is True
        self.connector.connect.assert_called_once()

        # Verify the connector's call_tool method was called with the right arguments
        self.connector.call_tool.assert_called_once_with("test_tool", {"param": "value"})

        # Verify the result is correct
        self.assertEqual(result, {"result": "success"})

    async def test_call_tool_with_auto_connect_false(self):
        """Test calling a tool with auto_connect set to False."""
        # Create a session with auto_connect=False
        session = MCPSession(self.connector, auto_connect=False)

        # Set up the connector to indicate it's already connected
        type(self.connector).client = PropertyMock(return_value=MagicMock())

        # Call the tool
        await session.call_tool("test_tool", {"param": "value"})

        # Verify connect was not called since auto_connect is False
        self.connector.connect.assert_not_called()
