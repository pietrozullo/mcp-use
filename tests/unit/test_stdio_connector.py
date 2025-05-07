"""
Unit tests for the StdioConnector class.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from mcp.types import CallToolResult, Tool

from mcp_use.connectors.stdio import StdioConnector
from mcp_use.task_managers.stdio import StdioConnectionManager


@pytest.fixture(autouse=True)
def mock_logger():
    """Mock the logger to prevent errors during tests."""
    with patch("mcp_use.connectors.base.logger") as mock_logger:
        yield mock_logger


class TestStdioConnectorInitialization:
    """Tests for StdioConnector initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        connector = StdioConnector()

        assert connector.command == "npx"
        assert connector.args == []
        assert connector.env is None
        assert connector.errlog == sys.stderr
        assert connector.client is None
        assert connector._connection_manager is None
        assert connector._tools is None
        assert connector._connected is False

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        command = "custom-command"
        args = ["--arg1", "--arg2"]
        env = {"ENV_VAR": "value"}
        errlog = Mock()

        connector = StdioConnector(command, args, env, errlog)

        assert connector.command == command
        assert connector.args == args
        assert connector.env == env
        assert connector.errlog == errlog
        assert connector.client is None
        assert connector._connection_manager is None
        assert connector._tools is None
        assert connector._connected is False


class TestStdioConnectorConnection:
    """Tests for StdioConnector connection methods."""

    @pytest.mark.asyncio
    @patch("mcp_use.connectors.stdio.StdioConnectionManager")
    @patch("mcp_use.connectors.stdio.ClientSession")
    @patch("mcp_use.connectors.stdio.logger")
    async def test_connect(self, mock_stdio_logger, mock_client_session, mock_connection_manager):
        """Test connecting to the MCP implementation."""
        # Setup mocks
        mock_manager_instance = Mock(spec=StdioConnectionManager)
        mock_manager_instance.start = AsyncMock(return_value=("read_stream", "write_stream"))
        mock_connection_manager.return_value = mock_manager_instance

        mock_client_instance = Mock()
        mock_client_instance.__aenter__ = AsyncMock()
        mock_client_session.return_value = mock_client_instance

        # Create connector and connect
        connector = StdioConnector(command="test-command", args=["--test"])
        await connector.connect()

        # Verify connection manager creation
        mock_connection_manager.assert_called_once()
        mock_manager_instance.start.assert_called_once()

        # Verify client session creation
        mock_client_session.assert_called_once_with(
            "read_stream", "write_stream", sampling_callback=None
        )
        mock_client_instance.__aenter__.assert_called_once()

        # Verify state
        assert connector._connected is True
        assert connector.client == mock_client_instance
        assert connector._connection_manager == mock_manager_instance

    @pytest.mark.asyncio
    @patch("mcp_use.connectors.stdio.logger")
    async def test_connect_already_connected(self, mock_stdio_logger):
        """Test connecting when already connected."""
        connector = StdioConnector()
        connector._connected = True

        await connector.connect()

        # Verify no connection established since already connected
        assert connector._connection_manager is None
        assert connector.client is None

    @pytest.mark.asyncio
    @patch("mcp_use.connectors.stdio.StdioConnectionManager")
    @patch("mcp_use.connectors.stdio.ClientSession")
    @patch("mcp_use.connectors.stdio.logger")
    @patch("mcp_use.connectors.base.logger")
    async def test_connect_error(
        self,
        mock_base_logger,
        mock_stdio_logger,
        mock_client_session,
        mock_connection_manager,
    ):
        """Test connection error handling."""
        # Setup mocks to raise an exception
        mock_manager_instance = Mock(spec=StdioConnectionManager)
        mock_manager_instance.start = AsyncMock(side_effect=Exception("Connection error"))
        mock_connection_manager.return_value = mock_manager_instance

        mock_manager_instance.stop = AsyncMock()

        # Create connector and attempt to connect
        connector = StdioConnector()

        # Expect the exception to be re-raised
        with pytest.raises(Exception, match="Connection error"):
            await connector.connect()

        # Verify resources were cleaned up
        assert connector._connected is False
        assert connector.client is None

        # Mock should be called to clean up resources
        mock_manager_instance.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self):
        """Test disconnecting when not connected."""
        connector = StdioConnector()
        connector._connected = False

        await connector.disconnect()

        # Should do nothing since not connected
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnecting from MCP implementation."""
        connector = StdioConnector()
        connector._connected = True

        # Mock the _cleanup_resources method to replace the actual method
        connector._cleanup_resources = AsyncMock()

        # Disconnect
        await connector.disconnect()

        # Verify _cleanup_resources was called
        connector._cleanup_resources.assert_called_once()

        # Verify state
        assert connector._connected is False


class TestStdioConnectorOperations:
    """Tests for StdioConnector operations."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initializing the MCP session."""
        connector = StdioConnector()

        # Setup mocks
        mock_client = MagicMock()
        # Mock client.initialize() to return capabilities
        mock_init_result = MagicMock()
        mock_init_result.status = (
            "success"  # Or whatever structure the Stdio connector expects to return
        )
        mock_init_result.capabilities = MagicMock(tools=True, resources=True, prompts=True)
        mock_client.initialize = AsyncMock(return_value=mock_init_result)

        # Mocks for list_tools, list_resources, list_prompts (already well-structured)
        mock_tools_response = MagicMock(tools=[MagicMock(spec=Tool)])
        mock_client.list_tools = AsyncMock(return_value=mock_tools_response)

        mock_list_resources_response = MagicMock()
        mock_list_resources_response.resources = []
        mock_client.list_resources = AsyncMock(return_value=mock_list_resources_response)

        # Mock list_prompts (called by base initialize)
        mock_list_prompts_response = MagicMock()
        mock_list_prompts_response.prompts = []  # Assumes a .prompts attribute
        mock_client.list_prompts = AsyncMock(return_value=mock_list_prompts_response)

        connector.client = mock_client

        # Initialize
        result_session_info = await connector.initialize()

        # Verify calls
        mock_client.initialize.assert_called_once()
        mock_client.list_tools.assert_called_once()
        mock_client.list_resources.assert_called_once()
        mock_client.list_prompts.assert_called_once()

        # Verify connector state and return value
        assert result_session_info == mock_init_result
        assert connector._tools is not None
        assert len(connector._tools) == 1
        assert connector._resources is not None
        assert len(connector._resources) == 0  # Based on current mock for list_resources
        assert connector._prompts is not None
        assert len(connector._prompts) == 0  # Based on current mock for list_prompts

    @pytest.mark.asyncio
    async def test_initialize_no_client(self):
        """Test initializing without a client."""
        connector = StdioConnector()
        connector.client = None

        # Expect RuntimeError
        with pytest.raises(RuntimeError, match="MCP client is not connected"):
            await connector.initialize()

    def test_tools_property(self):
        """Test the tools property."""
        connector = StdioConnector()
        mock_tools = [Mock(spec=Tool)]
        connector._tools = mock_tools

        # Get tools
        tools = connector.tools

        assert tools == mock_tools

    def test_tools_property_not_initialized(self):
        """Test the tools property when not initialized."""
        connector = StdioConnector()
        connector._tools = None

        # Expect RuntimeError
        with pytest.raises(RuntimeError, match="MCP client is not initialized"):
            _ = connector.tools

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """Test calling an MCP tool."""
        connector = StdioConnector()
        mock_client = Mock()
        mock_result = Mock(spec=CallToolResult)
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        connector.client = mock_client

        # Call tool
        tool_name = "test_tool"
        arguments = {"param": "value"}
        result = await connector.call_tool(tool_name, arguments)

        # Verify
        mock_client.call_tool.assert_called_once_with(tool_name, arguments)
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_call_tool_no_client(self):
        """Test calling a tool without a client."""
        connector = StdioConnector()
        connector.client = None

        # Expect RuntimeError
        with pytest.raises(RuntimeError, match="MCP client is not connected"):
            await connector.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test listing resources."""
        connector = StdioConnector()
        mock_client = Mock()
        mock_result = MagicMock()
        mock_result.resources = [MagicMock()]
        mock_client.list_resources = AsyncMock(return_value=mock_result)
        connector.client = mock_client

        # List resources
        result = await connector.list_resources()

        # Verify
        mock_client.list_resources.assert_called_once()
        assert result == mock_result.resources

    @pytest.mark.asyncio
    async def test_list_resources_no_client(self):
        """Test listing resources without a client."""
        connector = StdioConnector()
        connector.client = None

        # Expect RuntimeError
        with pytest.raises(RuntimeError, match="MCP client is not connected"):
            await connector.list_resources()

    @pytest.mark.asyncio
    async def test_read_resource(self):
        """Test reading a resource."""
        # Mocked return for connector.client.read_resource().
        # Needs the structure StdioConnector.read_resource expects.
        mock_client_return_value = (
            MagicMock()
        )  # spec=ReadResourceResult optional with MagicMock if defining manually

        # Define the nested structure
        content_item_mock = MagicMock()
        content_item_mock.content = b"test content"
        # Note camelCase.
        # Adjust if StdioConnector expects a different attribute name for mimetype.
        content_item_mock.mimeType = "text/plain"

        result_attribute_mock = MagicMock()  # This is for the .result attribute
        result_attribute_mock.contents = [content_item_mock]  # .contents is a list of these items

        mock_client_return_value.result = result_attribute_mock

        # Setup the connector and mock client
        connector = StdioConnector()
        # Mock the Stdio client and its methods.
        mock_stdio_client = MagicMock()
        mock_stdio_client.read_resource = AsyncMock(return_value=mock_client_return_value)
        # If other client methods are called by StdioConnector.read_resource,
        # ensure they are AsyncMocks.

        connector.client = mock_stdio_client
        # If connector relies on state from connect/initialize (e.g., _connected = True),
        # set it manually for this unit test.

        # Act: Call the connector's method
        # connector.read_resource returns a ReadResourceResult object.
        read_resource_result = await connector.read_resource("test/resource")

        # Assert: Verify the outcome and client interaction
        # Assert attributes of ReadResourceResult against mock_client_return_value.
        assert read_resource_result.result is not None
        assert read_resource_result.result.contents is not None
        assert len(read_resource_result.result.contents) == 1
        assert read_resource_result.result.contents[0].content == b"test content"
        assert read_resource_result.result.contents[0].mimeType == "text/plain"
        mock_stdio_client.read_resource.assert_called_once_with("test/resource")

    @pytest.mark.asyncio
    async def test_read_resource_no_client(self):
        """Test reading a resource without a client."""
        connector = StdioConnector()
        connector.client = None

        # Expect RuntimeError
        with pytest.raises(RuntimeError, match="MCP client is not connected"):
            await connector.read_resource("test_uri")

    @pytest.mark.asyncio
    async def test_request(self):
        """Test sending a raw request."""
        connector = StdioConnector()
        mock_client = Mock()
        mock_result = {"result": "success"}
        mock_client.request = AsyncMock(return_value=mock_result)
        connector.client = mock_client

        # Send request
        method = "test_method"
        params = {"param": "value"}
        result = await connector.request(method, params)

        # Verify
        mock_client.request.assert_called_once_with({"method": method, "params": params})
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_request_no_params(self):
        """Test sending a raw request without params."""
        connector = StdioConnector()
        mock_client = Mock()
        mock_result = {"result": "success"}
        mock_client.request = AsyncMock(return_value=mock_result)
        connector.client = mock_client

        # Send request without params
        method = "test_method"
        result = await connector.request(method)

        # Verify
        mock_client.request.assert_called_once_with({"method": method, "params": {}})
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_request_no_client(self):
        """Test sending a raw request without a client."""
        connector = StdioConnector()
        connector.client = None

        # Expect RuntimeError
        with pytest.raises(RuntimeError, match="MCP client is not connected"):
            await connector.request("test_method")
