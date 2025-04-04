"""
Unit tests for the HttpConnector class.
"""

import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from mcp.types import Tool

from mcp_use.connectors.http import HttpConnector
from mcp_use.task_managers import HttpConnectionManager


class TestHttpConnectorInitialization(unittest.TestCase):
    """Tests for HttpConnector initialization."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        connector = HttpConnector(base_url="http://localhost:8000")

        self.assertEqual(connector.base_url, "http://localhost:8000")
        self.assertIsNone(connector.auth_token)
        self.assertEqual(connector.headers, {})
        self.assertIsNone(connector.session)
        self.assertIsNone(connector._connection_manager)
        self.assertIsNone(connector._tools)
        self.assertFalse(connector._connected)

    def test_init_with_auth_token(self):
        """Test initialization with auth token."""
        connector = HttpConnector(base_url="http://localhost:8000", auth_token="test_token")

        self.assertEqual(connector.base_url, "http://localhost:8000")
        self.assertEqual(connector.auth_token, "test_token")
        self.assertEqual(connector.headers, {"Authorization": "Bearer test_token"})
        self.assertIsNone(connector.session)
        self.assertIsNone(connector._connection_manager)
        self.assertIsNone(connector._tools)
        self.assertFalse(connector._connected)

    def test_init_with_headers(self):
        """Test initialization with custom headers."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        connector = HttpConnector(base_url="http://localhost:8000", headers=headers)

        self.assertEqual(connector.base_url, "http://localhost:8000")
        self.assertIsNone(connector.auth_token)
        self.assertEqual(connector.headers, headers)
        self.assertIsNone(connector.session)
        self.assertIsNone(connector._connection_manager)
        self.assertIsNone(connector._tools)
        self.assertFalse(connector._connected)

    def test_init_with_auth_token_and_headers(self):
        """Test initialization with both auth token and headers."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        connector = HttpConnector(
            base_url="http://localhost:8000", auth_token="test_token", headers=headers
        )

        expected_headers = headers.copy()
        expected_headers["Authorization"] = "Bearer test_token"

        self.assertEqual(connector.base_url, "http://localhost:8000")
        self.assertEqual(connector.auth_token, "test_token")
        self.assertEqual(connector.headers, expected_headers)
        self.assertIsNone(connector.session)
        self.assertIsNone(connector._connection_manager)
        self.assertIsNone(connector._tools)
        self.assertFalse(connector._connected)

    def test_base_url_trailing_slash_removal(self):
        """Test that trailing slashes are removed from base_url."""
        connector = HttpConnector(base_url="http://localhost:8000/")
        self.assertEqual(connector.base_url, "http://localhost:8000")


class TestHttpConnectorConnection(IsolatedAsyncioTestCase):
    """Tests for HttpConnector connection methods."""

    def setUp(self):
        """Set up a connector for each test."""
        self.connector = HttpConnector(base_url="http://localhost:8000")

        # Mock the connection manager
        self.mock_cm = MagicMock(spec=HttpConnectionManager)
        self.mock_cm.start = AsyncMock()
        self.mock_cm.stop = AsyncMock()

        # Mock the session
        self.mock_session = MagicMock(spec=aiohttp.ClientSession)

    @patch("mcp_use.connectors.http.HttpConnectionManager")
    async def test_connect(self, mock_cm_class):
        """Test connecting to the MCP implementation."""
        # Setup mocks
        mock_cm_instance = self.mock_cm
        mock_cm_class.return_value = mock_cm_instance
        mock_cm_instance.start.return_value = self.mock_session

        # Test connect
        await self.connector.connect()

        # Verify connection manager was created and started
        mock_cm_class.assert_called_once_with("http://localhost:8000", {})
        mock_cm_instance.start.assert_called_once()

        # Verify state changes
        self.assertEqual(self.connector.session, self.mock_session)
        self.assertEqual(self.connector._connection_manager, mock_cm_instance)
        self.assertTrue(self.connector._connected)

    @patch("mcp_use.connectors.http.HttpConnectionManager")
    async def test_connect_already_connected(self, mock_cm_class):
        """Test connecting when already connected."""
        # Set up the connector as already connected
        self.connector._connected = True

        # Test connect
        await self.connector.connect()

        # Verify connection manager was not created or started
        mock_cm_class.assert_not_called()

    @patch("mcp_use.connectors.http.HttpConnectionManager")
    async def test_connect_failure(self, mock_cm_class):
        """Test handling connection failures."""
        # Setup mocks
        mock_cm_instance = self.mock_cm
        mock_cm_class.return_value = mock_cm_instance
        mock_cm_instance.start.side_effect = Exception("Connection failed")

        # Test connect failure
        with self.assertRaises(Exception) as context:
            await self.connector.connect()

        self.assertEqual(str(context.exception), "Connection failed")

        # Verify cleanup was called
        mock_cm_instance.stop.assert_called_once()

        # Verify state remains unchanged
        self.assertIsNone(self.connector.session)
        self.assertIsNone(self.connector._connection_manager)
        self.assertFalse(self.connector._connected)

    async def test_disconnect(self):
        """Test disconnecting from the MCP implementation."""
        # Set up the connector as connected
        self.connector._connected = True
        self.connector._connection_manager = self.mock_cm

        # Test disconnect
        await self.connector.disconnect()

        # Verify connection manager was stopped
        self.mock_cm.stop.assert_called_once()

        # Verify state changes
        self.assertFalse(self.connector._connected)
        self.assertIsNone(self.connector._connection_manager)

    async def test_disconnect_not_connected(self):
        """Test disconnecting when not connected."""
        # Ensure the connector is not connected
        self.connector._connected = False

        # Test disconnect
        await self.connector.disconnect()

        # Verify no action was taken
        self.assertIsNone(self.connector._connection_manager)
        self.assertFalse(self.connector._connected)


# Mock the Tool class to avoid validation errors
@patch("mcp_use.connectors.http.Tool")
class TestHttpConnectorOperations(IsolatedAsyncioTestCase):
    """Tests for HttpConnector operations."""

    def setUp(self):
        """Set up a connector for each test."""
        self.connector = HttpConnector(base_url="http://localhost:8000")

        # Set up the connector as connected and initialized
        self.connector._connected = True
        self.connector.session = MagicMock(spec=aiohttp.ClientSession)

        # Create mock response
        self.mock_response = MagicMock()
        self.mock_response.raise_for_status = MagicMock()
        self.mock_response.json = AsyncMock()
        self.mock_response.__aenter__ = AsyncMock(return_value=self.mock_response)
        self.mock_response.__aexit__ = AsyncMock()

    async def test_request_get(self, _):
        """Test sending a GET request."""
        # Set up mock response
        self.mock_response.json.return_value = {"result": "success"}
        self.connector.session.get.return_value = self.mock_response

        # Test _request GET method
        result = await self.connector._request("GET", "endpoint", {"param": "value"})

        # Verify request
        self.connector.session.get.assert_called_once_with(
            "http://localhost:8000/endpoint", params={"param": "value"}
        )

        # Verify result
        self.assertEqual(result, {"result": "success"})

    async def test_request_post(self, _):
        """Test sending a POST request."""
        # Set up mock response
        self.mock_response.json.return_value = {"result": "success"}
        self.connector.session.request.return_value = self.mock_response

        # Test _request POST method
        result = await self.connector._request("POST", "endpoint", {"param": "value"})

        # Verify request
        self.connector.session.request.assert_called_once_with(
            "POST", "http://localhost:8000/endpoint", json={"param": "value"}
        )

        # Verify result
        self.assertEqual(result, {"result": "success"})

    async def test_request_not_connected(self, _):
        """Test sending a request when not connected."""
        # Set up connector as not connected
        self.connector.session = None

        # Test _request raises RuntimeError
        with self.assertRaises(RuntimeError) as context:
            await self.connector._request("GET", "endpoint")

        self.assertEqual(str(context.exception), "HTTP session is not connected")

    async def test_initialize(self, mock_tool):
        """Test initializing the MCP session."""
        # Set up mock Tool class
        mock_tool_instance = MagicMock()
        mock_tool.return_value = mock_tool_instance

        # Set up mock responses
        initialize_response = {"session_id": "test_session"}
        tools_response = {
            "tools": [{"name": "test_tool", "description": "A test tool", "inputSchema": {}}]
        }

        # Set up connector to return mock responses
        self.connector._request = AsyncMock()
        self.connector._request.side_effect = [initialize_response, tools_response]

        # Test initialize
        result = await self.connector.initialize()

        # Verify requests
        self.connector._request.assert_any_call("POST", "initialize")
        self.connector._request.assert_any_call("GET", "tools")

        # Verify result
        self.assertEqual(result, initialize_response)
        self.assertEqual(len(self.connector._tools), 1)
        self.assertEqual(self.connector._tools[0], mock_tool_instance)

        # Verify Tool was created with the right parameters
        mock_tool.assert_called_with(name="test_tool", description="A test tool", inputSchema={})

    async def test_list_tools(self, _):
        """Test listing available tools."""
        # Set up mock response
        tools_response = {"tools": [{"name": "test_tool", "description": "A test tool"}]}
        self.connector._request = AsyncMock(return_value=tools_response)

        # Test list_tools
        result = await self.connector.list_tools()

        # Verify request
        self.connector._request.assert_called_once_with("GET", "tools")

        # Verify result
        self.assertEqual(result, [{"name": "test_tool", "description": "A test tool"}])

    async def test_tools_property_initialized(self, _):
        """Test the tools property when initialized."""
        # Set up mock tools
        mock_tool_instance = MagicMock(name="test_tool")
        self.connector._tools = [mock_tool_instance]

        # Test tools property
        tools = self.connector.tools

        # Verify result
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0], mock_tool_instance)

    async def test_tools_property_not_initialized(self, _):
        """Test the tools property when not initialized."""
        # Set up connector with no tools
        self.connector._tools = None

        # Test tools property raises RuntimeError
        with self.assertRaises(RuntimeError) as context:
            _ = self.connector.tools

        self.assertEqual(str(context.exception), "MCP client is not initialized")

    async def test_call_tool(self, _):
        """Test calling a tool."""
        # Set up mock response
        tool_response = {"result": "success"}
        self.connector._request = AsyncMock(return_value=tool_response)

        # Test call_tool
        result = await self.connector.call_tool("test_tool", {"param": "value"})

        # Verify request
        self.connector._request.assert_called_once_with(
            "POST", "tools/test_tool", {"param": "value"}
        )

        # Verify result
        self.assertEqual(result, tool_response)
