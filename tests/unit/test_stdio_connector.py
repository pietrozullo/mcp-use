"""
Unit tests for the StdioConnector class.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from mcp.types import CallToolResult, ListResourcesResult, ReadResourceResult, Tool

from mcp_use.connectors.stdio import StdioConnector
from mcp_use.task_managers.stdio import StdioConnectionManager


@pytest.fixture(autouse=True)
def mock_logger():
    """Mock the logger to prevent errors during tests."""
    with patch("mcp_use.connectors.stdio.logger") as mock_logger:
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
    async def test_connect(self, mock_client_session, mock_connection_manager):
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
    async def test_connect_already_connected(self):
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
    async def test_connect_error(self, mock_client_session, mock_connection_manager):
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

        # Mock the _cleanup_resources method directly on the instance
        connector._cleanup_resources = AsyncMock()

        # Disconnect
        await connector.disconnect()

        # Verify _cleanup_resources was called
        connector._cleanup_resources.assert_called_once()

        # Verify state
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_cleanup_resources_with_errors(self):
        """Test resource cleanup with errors."""
        connector = StdioConnector()

        # Setup mocks for cleanup with errors
        mock_client = Mock()
        mock_client.__aexit__ = AsyncMock(side_effect=Exception("Client exit error"))
        connector.client = mock_client

        mock_connection_manager = Mock()
        mock_connection_manager.stop = AsyncMock(
            side_effect=Exception("Connection manager stop error")
        )
        connector._connection_manager = mock_connection_manager

        connector._tools = [Mock(spec=Tool)]

        # Store references to check later
        client_aexit = mock_client.__aexit__
        connection_manager_stop = mock_connection_manager.stop

        # Cleanup resources
        await connector._cleanup_resources()

        # Verify attempts were made to clean up despite errors
        client_aexit.assert_called_once_with(None, None, None)
        connection_manager_stop.assert_called_once()

        # Resources should be set to None
        assert connector.client is None
        assert connector._connection_manager is None
        assert connector._tools is None


class TestStdioConnectorOperations:
    """Tests for StdioConnector operations."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initializing the MCP session."""
        connector = StdioConnector()

        # Setup mocks
        mock_client = Mock()
        mock_client.initialize = AsyncMock(return_value={"status": "success"})
        mock_client.list_tools = AsyncMock(return_value=Mock(tools=[Mock(spec=Tool)]))
        connector.client = mock_client

        # Initialize
        result = await connector.initialize()

        # Verify
        mock_client.initialize.assert_called_once()
        mock_client.list_tools.assert_called_once()

        assert result == {"status": "success"}
        assert connector._tools is not None
        assert len(connector._tools) == 1

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
        """Test calling a tool."""
        connector = StdioConnector()

        # Setup mock
        mock_client = Mock()
        mock_result = Mock(spec=CallToolResult)
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        connector.client = mock_client

        # Call tool
        result = await connector.call_tool("test-tool", {"arg": "value"})

        # Verify
        mock_client.call_tool.assert_called_once_with("test-tool", {"arg": "value"})
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_call_tool_no_client(self):
        """Test calling a tool without a client."""
        connector = StdioConnector()
        connector.client = None

        # Expect RuntimeError
        with pytest.raises(RuntimeError, match="MCP client is not connected"):
            await connector.call_tool("test-tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test listing resources."""
        connector = StdioConnector()

        # Setup mock
        mock_client = Mock()
        mock_resources = [{"uri": "test://resource"}]
        mock_client.list_resources = AsyncMock(return_value=mock_resources)
        connector.client = mock_client

        # List resources
        resources = await connector.list_resources()

        # Verify
        mock_client.list_resources.assert_called_once()
        assert resources == mock_resources

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
        connector = StdioConnector()

        # Setup mock
        mock_client = Mock()
        mock_resource = Mock(
            spec=ReadResourceResult, content=b"test content", mimeType="text/plain"
        )
        mock_client.read_resource = AsyncMock(return_value=mock_resource)
        connector.client = mock_client

        # Read resource
        content, mime_type = await connector.read_resource("test://resource")

        # Verify
        mock_client.read_resource.assert_called_once_with("test://resource")
        assert content == b"test content"
        assert mime_type == "text/plain"

    @pytest.mark.asyncio
    async def test_read_resource_no_client(self):
        """Test reading a resource without a client."""
        connector = StdioConnector()
        connector.client = None

        # Expect RuntimeError
        with pytest.raises(RuntimeError, match="MCP client is not connected"):
            await connector.read_resource("test://resource")

    @pytest.mark.asyncio
    async def test_request(self):
        """Test sending a raw request."""
        connector = StdioConnector()

        # Setup mock
        mock_client = Mock()
        mock_client.request = AsyncMock(return_value={"result": "success"})
        connector.client = mock_client

        # Send request
        result = await connector.request("test.method", {"param": "value"})

        # Verify
        mock_client.request.assert_called_once_with(
            {"method": "test.method", "params": {"param": "value"}}
        )
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_request_no_params(self):
        """Test sending a raw request without params."""
        connector = StdioConnector()

        # Setup mock
        mock_client = Mock()
        mock_client.request = AsyncMock(return_value={"result": "success"})
        connector.client = mock_client

        # Send request without params
        result = await connector.request("test.method")

        # Verify
        mock_client.request.assert_called_once_with({"method": "test.method", "params": {}})
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_request_no_client(self):
        """Test sending a request without a client."""
        connector = StdioConnector()
        connector.client = None

        # Expect RuntimeError
        with pytest.raises(RuntimeError, match="MCP client is not connected"):
            await connector.request("test.method")
