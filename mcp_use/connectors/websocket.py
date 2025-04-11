"""
WebSocket connector for MCP implementations.

This module provides a connector for communicating with MCP implementations
through WebSocket connections.
"""

import asyncio
import json
import uuid
from typing import Any

from mcp.types import Tool
from websockets.client import WebSocketClientProtocol

from ..logging import logger
from ..task_managers import ConnectionManager, WebSocketConnectionManager
from .base import BaseConnector


class WebSocketConnector(BaseConnector):
    """Connector for MCP implementations using WebSocket transport.

    This connector uses WebSockets to communicate with remote MCP implementations,
    using a connection manager to handle the proper lifecycle management.
    """

    def __init__(
        self,
        url: str,
        auth_token: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        """Initialize a new WebSocket connector.

        Args:
            url: The WebSocket URL to connect to.
            auth_token: Optional authentication token.
            headers: Optional additional headers.
        """
        self.url = url
        self.auth_token = auth_token
        self.headers = headers or {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

        self.ws: WebSocketClientProtocol | None = None
        self._connection_manager: ConnectionManager | None = None
        self._receiver_task: asyncio.Task | None = None
        self.pending_requests: dict[str, asyncio.Future] = {}
        self._tools: list[Tool] | None = None
        self._connected = False

    async def connect(self) -> None:
        """Establish a connection to the MCP implementation."""
        if self._connected:
            logger.debug("Already connected to MCP implementation")
            return

        logger.debug(f"Connecting to MCP implementation via WebSocket: {self.url}")
        try:
            # Create and start the connection manager
            self._connection_manager = WebSocketConnectionManager(self.url, self.headers)
            self.ws = await self._connection_manager.start()

            # Start the message receiver task
            self._receiver_task = asyncio.create_task(
                self._receive_messages(), name="websocket_receiver_task"
            )

            # Mark as connected
            self._connected = True
            logger.debug(f"Successfully connected to MCP implementation via WebSocket: {self.url}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP implementation via WebSocket: {e}")

            # Clean up any resources if connection failed
            await self._cleanup_resources()

            # Re-raise the original exception
            raise

    async def _receive_messages(self) -> None:
        """Continuously receive and process messages from the WebSocket."""
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")

        try:
            async for message in self.ws:
                # Parse the message
                data = json.loads(message)

                # Check if this is a response to a pending request
                request_id = data.get("id")
                if request_id and request_id in self.pending_requests:
                    future = self.pending_requests.pop(request_id)
                    if "result" in data:
                        future.set_result(data["result"])
                    elif "error" in data:
                        future.set_exception(Exception(data["error"]))

                    logger.debug(f"Received response for request {request_id}")
                else:
                    logger.debug(f"Received message: {data}")
        except Exception as e:
            logger.error(f"Error in WebSocket message receiver: {e}")
            # If the websocket connection was closed or errored,
            # reject all pending requests
            for future in self.pending_requests.values():
                if not future.done():
                    future.set_exception(e)

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

        # First cancel the receiver task
        if self._receiver_task and not self._receiver_task.done():
            try:
                logger.debug("Cancelling WebSocket receiver task")
                self._receiver_task.cancel()
                try:
                    await self._receiver_task
                except asyncio.CancelledError:
                    logger.debug("WebSocket receiver task cancelled successfully")
                except Exception as e:
                    logger.warning(f"Error during WebSocket receiver task cancellation: {e}")
            except Exception as e:
                error_msg = f"Error cancelling WebSocket receiver task: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
            finally:
                self._receiver_task = None

        # Reject any pending requests
        if self.pending_requests:
            logger.debug(f"Rejecting {len(self.pending_requests)} pending requests")
            for future in self.pending_requests.values():
                if not future.done():
                    future.set_exception(ConnectionError("WebSocket disconnected"))
            self.pending_requests.clear()

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
                self.ws = None

        # Reset tools
        self._tools = None

        if errors:
            logger.warning(f"Encountered {len(errors)} errors during resource cleanup")

    async def _send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a request and wait for a response."""
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")

        # Create a request ID
        request_id = str(uuid.uuid4())

        # Create a future to receive the response
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        # Send the request
        await self.ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))

        logger.debug(f"Sent request {request_id} method: {method}")

        # Wait for the response
        try:
            return await future
        except Exception as e:
            # Remove the request from pending requests
            self.pending_requests.pop(request_id, None)
            logger.error(f"Error waiting for response to request {request_id}: {e}")
            raise

    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP session and return session information."""
        logger.debug("Initializing MCP session")
        result = await self._send_request("initialize")

        # Get available tools
        tools_result = await self.list_tools()
        self._tools = [Tool(**tool) for tool in tools_result]

        logger.debug(f"MCP session initialized with {len(self._tools)} tools")
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools from the MCP implementation."""
        logger.debug("Listing tools")
        result = await self._send_request("tools/list")
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
        return await self._send_request("tools/call", {"name": name, "arguments": arguments})

    async def list_resources(self) -> list[dict[str, Any]]:
        """List all available resources from the MCP implementation."""
        logger.debug("Listing resources")
        result = await self._send_request("resources/list")
        return result

    async def read_resource(self, uri: str) -> tuple[bytes, str]:
        """Read a resource by URI."""
        logger.debug(f"Reading resource: {uri}")
        result = await self._send_request("resources/read", {"uri": uri})
        return result.get("content", b""), result.get("mimeType", "")

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a raw request to the MCP implementation."""
        logger.debug(f"Sending request: {method} with params: {params}")
        return await self._send_request(method, params)
