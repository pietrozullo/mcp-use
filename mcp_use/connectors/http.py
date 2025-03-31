"""
HTTP connector for MCP implementations.

This module provides a connector for communicating with MCP implementations
through HTTP APIs.
"""

from typing import Any

import aiohttp
from mcp.types import Tool

from ..logging import logger
from ..task_managers import ConnectionManager, HttpConnectionManager
from .base import BaseConnector


class HttpConnector(BaseConnector):
    """Connector for MCP implementations using HTTP transport.

    This connector uses HTTP requests to communicate with remote MCP implementations,
    using a connection manager to handle the proper lifecycle management.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        """Initialize a new HTTP connector.

        Args:
            base_url: The base URL of the MCP HTTP API.
            auth_token: Optional authentication token.
            headers: Optional additional headers.
        """
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.headers = headers or {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

        self.session: aiohttp.ClientSession | None = None
        self._connection_manager: ConnectionManager | None = None
        self._tools: list[Tool] | None = None
        self._connected = False

    async def connect(self) -> None:
        """Establish a connection to the MCP implementation."""
        if self._connected:
            logger.debug("Already connected to MCP implementation")
            return

        logger.info(f"Connecting to MCP implementation via HTTP: {self.base_url}")
        try:
            # Create and start the connection manager
            self._connection_manager = HttpConnectionManager(self.base_url, self.headers)
            self.session = await self._connection_manager.start()

            # Mark as connected
            self._connected = True
            logger.info(f"Successfully connected to MCP implementation via HTTP: {self.base_url}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP implementation via HTTP: {e}")

            # Clean up any resources if connection failed
            await self._cleanup_resources()

            # Re-raise the original exception
            raise

    async def disconnect(self) -> None:
        """Close the connection to the MCP implementation."""
        if not self._connected:
            logger.debug("Not connected to MCP implementation")
            return

        logger.info("Disconnecting from MCP implementation")
        await self._cleanup_resources()
        self._connected = False
        logger.info("Disconnected from MCP implementation")

    async def _cleanup_resources(self) -> None:
        """Clean up all resources associated with this connector."""
        errors = []

        # Stop the connection manager
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
                self.session = None

        # Reset tools
        self._tools = None

        if errors:
            logger.warning(f"Encountered {len(errors)} errors during resource cleanup")

    async def _request(self, method: str, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """Send an HTTP request to the MCP API.

        Args:
            method: The HTTP method (GET, POST, etc.).
            endpoint: The API endpoint path.
            data: Optional request data.

        Returns:
            The parsed JSON response.
        """
        if not self.session:
            raise RuntimeError("HTTP session is not connected")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.debug(f"Sending {method} request to {url}")

        if method.upper() == "GET" and data:
            # For GET requests, convert data to query parameters
            async with self.session.get(url, params=data) as response:
                response.raise_for_status()
                return await response.json()
        else:
            # For other methods, send data as JSON body
            async with self.session.request(method, url, json=data) as response:
                response.raise_for_status()
                return await response.json()

    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP session and return session information."""
        logger.info("Initializing MCP session")

        # Initialize the session
        result = await self._request("POST", "initialize")

        # Get available tools
        tools_result = await self.list_tools()
        self._tools = [Tool(**tool) for tool in tools_result]

        logger.info(f"MCP session initialized with {len(self._tools)} tools")
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools from the MCP implementation."""
        logger.debug("Listing tools")
        result = await self._request("GET", "tools")
        return result.get("tools", [])

    @property
    def tools(self) -> list[Tool]:
        """Get the list of available tools."""
        if not self._tools:
            raise RuntimeError("MCP client is not initialized")
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool with the given arguments."""
        logger.debug(f"Calling tool '{name}' with arguments: {arguments}")
        return await self._request("POST", f"tools/{name}", arguments)

    async def list_resources(self) -> list[dict[str, Any]]:
        """List all available resources from the MCP implementation."""
        logger.debug("Listing resources")
        result = await self._request("GET", "resources")
        return result

    async def read_resource(self, uri: str) -> tuple[bytes, str]:
        """Read a resource by URI."""
        logger.debug(f"Reading resource: {uri}")

        # For resources, we may need to handle binary data
        if not self.session:
            raise RuntimeError("HTTP session is not connected")

        url = f"{self.base_url}/resources/read"

        async with self.session.get(url, params={"uri": uri}) as response:
            response.raise_for_status()

            # Check if this is a JSON response or binary data
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                data = await response.json()
                content = data.get("content", b"")
                mime_type = data.get("mimeType", "")

                # If content is base64 encoded, decode it
                if isinstance(content, str):
                    import base64

                    content = base64.b64decode(content)

                return content, mime_type
            else:
                # Assume binary response
                content = await response.read()
                return content, content_type

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a raw request to the MCP implementation."""
        logger.debug(f"Sending request: {method} with params: {params}")
        # For custom methods, we'll use the RPC-style endpoint
        return await self._request("POST", "rpc", {"method": method, "params": params or {}})
