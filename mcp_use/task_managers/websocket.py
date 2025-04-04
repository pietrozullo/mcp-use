"""
WebSocket connection management for MCP implementations.

This module provides a connection manager for WebSocket-based MCP connections.
"""

import websockets
from websockets.client import ClientConnection

from ..logging import logger
from .base import ConnectionManager


class WebSocketConnectionManager(ConnectionManager[ClientConnection]):
    """Connection manager for WebSocket-based MCP connections.

    This class handles the lifecycle of WebSocket connections, ensuring proper
    connection establishment and cleanup.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ):
        """Initialize a new WebSocket connection manager.

        Args:
            url: The WebSocket URL to connect to
            headers: Optional headers to include in the WebSocket connection
        """
        super().__init__()
        self.url = url
        self.headers = headers or {}

    async def _establish_connection(self) -> ClientConnection:
        """Establish a WebSocket connection.

        Returns:
            The established WebSocket connection

        Raises:
            Exception: If connection cannot be established
        """
        logger.debug(f"Connecting to WebSocket: {self.url}")
        try:
            ws = await websockets.connect(self.url, extra_headers=self.headers)
            return ws
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise

    async def _close_connection(self, connection: ClientConnection) -> None:
        """Close the WebSocket connection.

        Args:
            connection: The WebSocket connection to close
        """
        try:
            logger.debug("Closing WebSocket connection")
            await connection.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket connection: {e}")
