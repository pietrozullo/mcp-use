"""
Unit tests for the HttpConnector class.
"""

import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from mcp.types import Prompt, Resource, Tool

from mcp_use.connectors.http import HttpConnector
from mcp_use.task_managers import SseConnectionManager


@patch("mcp_use.connectors.base.logger")
class TestHttpConnectorInitialization(unittest.TestCase):
    """Tests for HttpConnector initialization."""

    def test_init_minimal(self, _):
        """Test initialization with minimal parameters."""
        connector = HttpConnector(base_url="http://localhost:8000")

        self.assertEqual(connector.base_url, "http://localhost:8000")
        self.assertIsNone(connector.auth_token)
        self.assertEqual(connector.headers, {})
        self.assertIsNone(connector.client)
        self.assertIsNone(connector._connection_manager)
        self.assertIsNone(connector._tools)
        self.assertFalse(connector._connected)

    def test_init_with_auth_token(self, _):
        """Test initialization with auth token."""
        connector = HttpConnector(base_url="http://localhost:8000", auth_token="test_token")

        self.assertEqual(connector.base_url, "http://localhost:8000")
        self.assertEqual(connector.auth_token, "test_token")
        self.assertEqual(connector.headers, {"Authorization": "Bearer test_token"})
        self.assertIsNone(connector.client)
        self.assertIsNone(connector._connection_manager)
        self.assertIsNone(connector._tools)
        self.assertFalse(connector._connected)

    def test_init_with_headers(self, _):
        """Test initialization with custom headers."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        connector = HttpConnector(base_url="http://localhost:8000", headers=headers)

        self.assertEqual(connector.base_url, "http://localhost:8000")
        self.assertIsNone(connector.auth_token)
        self.assertEqual(connector.headers, headers)
        self.assertIsNone(connector.client)
        self.assertIsNone(connector._connection_manager)
        self.assertIsNone(connector._tools)
        self.assertFalse(connector._connected)

    def test_init_with_auth_token_and_headers(self, _):
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
        self.assertIsNone(connector.client)
        self.assertIsNone(connector._connection_manager)
        self.assertIsNone(connector._tools)
        self.assertFalse(connector._connected)

    def test_base_url_trailing_slash_removal(self, _):
        """Test that trailing slashes are removed from base_url."""
        connector = HttpConnector(base_url="http://localhost:8000/")
        self.assertEqual(connector.base_url, "http://localhost:8000")


@patch("mcp_use.connectors.base.logger")
class TestHttpConnectorConnection(IsolatedAsyncioTestCase):
    """Tests for HttpConnector connection methods."""

    def setUp(self):
        """Set up a connector for each test."""
        self.connector = HttpConnector(base_url="http://localhost:8000")

        # Mock the connection manager
        self.mock_cm = MagicMock(spec=SseConnectionManager)
        self.mock_cm.start = AsyncMock()
        self.mock_cm.stop = AsyncMock()

        # Mock the client session
        self.mock_client_session = MagicMock()
        self.mock_client_session.__aenter__ = AsyncMock()

    @patch("mcp_use.connectors.http.SseConnectionManager")
    @patch("mcp_use.connectors.http.ClientSession")
    async def test_connect(self, mock_client_session_class, mock_cm_class, _):
        """Test connecting to the MCP implementation."""
        # Setup mocks
        mock_cm_instance = self.mock_cm
        mock_cm_class.return_value = mock_cm_instance
        mock_cm_instance.start.return_value = ("read_stream", "write_stream")

        mock_client_session_instance = self.mock_client_session
        mock_client_session_class.return_value = mock_client_session_instance

        # Test connect
        await self.connector.connect()

        # Verify connection manager was created and started
        mock_cm_class.assert_called_once_with("http://localhost:8000", {}, 5, 300)
        mock_cm_instance.start.assert_called_once()

        # Verify client session was created
        mock_client_session_class.assert_called_once_with(
            "read_stream", "write_stream", sampling_callback=None
        )
        mock_client_session_instance.__aenter__.assert_called_once()

        # Verify state changes
        self.assertEqual(self.connector.client, mock_client_session_instance)
        self.assertEqual(self.connector._connection_manager, mock_cm_instance)
        self.assertTrue(self.connector._connected)

    @patch("mcp_use.connectors.http.SseConnectionManager")
    async def test_connect_already_connected(self, mock_cm_class, _):
        """Test connecting when already connected."""
        # Set up the connector as already connected
        self.connector._connected = True

        # Test connect
        await self.connector.connect()

        # Verify connection manager was not created or started
        mock_cm_class.assert_not_called()

    @patch("mcp_use.connectors.http.SseConnectionManager")
    async def test_connect_failure(self, mock_cm_class, _):
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
        self.assertIsNone(self.connector.client)
        self.assertIsNone(self.connector._connection_manager)
        self.assertFalse(self.connector._connected)

    async def test_disconnect(self, mock_logger):
        """Test disconnecting from the MCP implementation."""
        # Set up the connector as connected
        self.connector._connected = True
        self.connector._connection_manager = self.mock_cm
        self.connector._cleanup_resources = AsyncMock()

        # Test disconnect
        await self.connector.disconnect()

        # Verify cleanup was called
        self.connector._cleanup_resources.assert_called_once()

        # Verify state changes
        self.assertFalse(self.connector._connected)

    async def test_disconnect_not_connected(self, _):
        """Test disconnecting when not connected."""
        # Ensure the connector is not connected
        self.connector._connected = False

        # Test disconnect
        await self.connector.disconnect()

        # Verify no action was taken
        self.assertIsNone(self.connector._connection_manager)
        self.assertFalse(self.connector._connected)


@patch("mcp_use.connectors.base.logger")
class TestHttpConnectorOperations(IsolatedAsyncioTestCase):
    """Tests for HttpConnector operations."""

    def setUp(self):
        """Set up a connector for each test."""
        self.connector = HttpConnector(base_url="http://localhost:8000")
        # Most operations assume the connector is connected and the client exists.
        self.connector._connected = True
        # Mock the internal client that HttpConnector methods will call.
        # Client methods are async, so AsyncMock is appropriate.
        self.connector.client = AsyncMock()
        # _tools is populated by initialize(). Tests that rely on _tools
        # should either call a mocked initialize() or set _tools directly.
        self.connector._tools = None  # Ensure clean state for _tools

    async def test_call_tool(self, _):
        """Test calling a tool."""
        self.connector.client.call_tool.return_value = {"result": "success"}

        result = await self.connector.call_tool("test_tool", {"param": "value"})

        self.connector.client.call_tool.assert_called_once_with("test_tool", {"param": "value"})
        self.assertEqual(result, {"result": "success"})

    async def test_call_tool_no_client(self, _):
        """Test calling a tool when not connected."""
        self.connector.client = None

        with self.assertRaises(RuntimeError) as context:
            await self.connector.call_tool("test_tool", {})

        self.assertEqual(str(context.exception), "MCP client is not connected")

    async def test_initialize(self, _):
        """Test initializing the MCP session with all capabilities enabled."""
        # Setup mock for client.initialize() to return capabilities
        mock_init_result = MagicMock()
        mock_init_result.session_id = "test_session"
        mock_init_result.capabilities = MagicMock(tools=True, resources=True, prompts=True)
        self.connector.client.initialize.return_value = mock_init_result

        # Setup mocks for list_tools, list_resources, and list_prompts
        self.connector.client.list_tools.return_value = MagicMock(tools=[MagicMock(spec=Tool)])
        self.connector.client.list_resources.return_value = MagicMock(
            resources=[MagicMock(spec=Resource)]
        )
        self.connector.client.list_prompts.return_value = MagicMock(
            prompts=[MagicMock(spec=Prompt)]
        )

        # Initialize
        result_session_info = await self.connector.initialize()

        # Verify calls
        self.connector.client.initialize.assert_called_once()
        self.connector.client.list_tools.assert_called_once()
        self.connector.client.list_resources.assert_called_once()
        self.connector.client.list_prompts.assert_called_once()

        # Verify connector state
        self.assertEqual(result_session_info, mock_init_result)
        self.assertEqual(len(self.connector._tools), 1)
        self.assertEqual(len(self.connector._resources), 1)
        self.assertEqual(len(self.connector._prompts), 1)

    async def test_initialize_no_client(self, _):
        """Test initializing without a client."""
        self.connector.client = None

        with self.assertRaises(RuntimeError) as context:
            await self.connector.initialize()

        self.assertEqual(str(context.exception), "MCP client is not connected")

    async def test_tools_property_initialized(self, _):
        """Test the tools property when initialized."""
        mock_tools = [MagicMock(spec=Tool)]
        self.connector._tools = mock_tools

        tools = self.connector.tools

        self.assertEqual(tools, mock_tools)

    async def test_tools_property_not_initialized(self, _):
        """Test the tools property when not initialized."""
        self.connector._tools = None

        with self.assertRaises(RuntimeError) as context:
            _ = self.connector.tools

        self.assertEqual(str(context.exception), "MCP client is not initialized")

    async def test_list_resources(self, _):
        """Test listing resources."""
        expected_resources_list = [{"uri": "test/resource"}]
        # Mock the client's list_resources method to return an object
        # that has a .resources attribute, as expected by the connector.
        mock_client_response = MagicMock()
        mock_client_response.resources = expected_resources_list
        self.connector.client.list_resources.return_value = mock_client_response

        # Call the connector's list_resources method
        result = await self.connector.list_resources()

        # Verify the client's method was called correctly by the connector
        self.connector.client.list_resources.assert_called_once_with()
        # The connector's list_resources method should return the list of resources directly.
        self.assertEqual(result, expected_resources_list)

    async def test_list_resources_no_client(self, _):
        """Test listing resources when not connected."""
        self.connector.client = None

        with self.assertRaises(RuntimeError) as context:
            await self.connector.list_resources()

        self.assertEqual(str(context.exception), "MCP client is not connected")

    async def test_read_resource(self, _):
        """Test reading a resource."""
        # Define the detailed structure that the connector's read_resource method
        # will parse from the object returned by client.read_resource().
        # This assumes a common MCP pattern where data is nested.
        mock_content_part = MagicMock()
        mock_content_part.content = b"test content"  # Changed from .text
        mock_content_part.mimeType = "text/plain"  # Changed from .mime_type, note camelCase

        mock_result_obj = MagicMock()
        mock_result_obj.contents = [mock_content_part]

        # This is the object returned by self.connector.client.read_resource()
        mock_client_return = MagicMock()
        mock_client_return.result = mock_result_obj  # Actual data is nested under 'result'

        self.connector.client.read_resource.return_value = mock_client_return

        # Act: Call the connector's method
        # The connector's read_resource method returns a ReadResourceResult object
        read_resource_result = await self.connector.read_resource("test/resource")

        # Assert: Verify client interaction and the processed result
        self.connector.client.read_resource.assert_called_once_with("test/resource")
        # Now, assert the attributes of the returned ReadResourceResult object
        # based on the mocked client_return object's structure.
        self.assertIsNotNone(read_resource_result.result)
        self.assertIsNotNone(read_resource_result.result.contents)
        self.assertEqual(len(read_resource_result.result.contents), 1)
        self.assertEqual(read_resource_result.result.contents[0].content, b"test content")
        self.assertEqual(read_resource_result.result.contents[0].mimeType, "text/plain")

    async def test_read_resource_no_client(self, _):
        """Test reading a resource when not connected."""
        self.connector.client = None

        with self.assertRaises(RuntimeError) as context:
            await self.connector.read_resource("test/resource")

        self.assertEqual(str(context.exception), "MCP client is not connected")

    async def test_request(self, _):
        """Test sending a request."""
        self.connector.client.request.return_value = {"result": "success"}

        result = await self.connector.request("test_method", {"param": "value"})

        self.connector.client.request.assert_called_once_with(
            {"method": "test_method", "params": {"param": "value"}}
        )
        self.assertEqual(result, {"result": "success"})

    async def test_request_no_params(self, _):
        """Test sending a request without params."""
        self.connector.client.request.return_value = {"result": "success"}

        result = await self.connector.request("test_method")

        self.connector.client.request.assert_called_once_with(
            {"method": "test_method", "params": {}}
        )
        self.assertEqual(result, {"result": "success"})

    async def test_request_no_client(self, _):
        """Test sending a request when not connected."""
        self.connector.client = None

        with self.assertRaises(RuntimeError) as context:
            await self.connector.request("test_method")

        self.assertEqual(str(context.exception), "MCP client is not connected")
