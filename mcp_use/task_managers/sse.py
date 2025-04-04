"""
SSE connection management for MCP implementations.

This module provides a connection manager for SSE-based MCP connections
that ensures proper task isolation and resource cleanup.
"""

from typing import Any

from mcp.client.sse import sse_client

from ..logging import logger
from .base import ConnectionManager


class SseConnectionManager(ConnectionManager[tuple[Any, Any]]):
    """Connection manager for SSE-based MCP connections.

    This class handles the proper task isolation for sse_client context managers
    to prevent the "cancel scope in different task" error. It runs the sse_client
    in a dedicated task and manages its lifecycle.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
    ):
        """Initialize a new SSE connection manager.

        Args:
            url: The SSE endpoint URL
            headers: Optional HTTP headers
            timeout: Timeout for HTTP operations in seconds
            sse_read_timeout: Timeout for SSE read operations in seconds
        """
        super().__init__()
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self._sse_ctx = None

    async def _establish_connection(self) -> tuple[Any, Any]:
        """Establish an SSE connection.

        Returns:
            A tuple of (read_stream, write_stream)

        Raises:
            Exception: If connection cannot be established.
        """
        # Create the context manager
        self._sse_ctx = sse_client(
            url=self.url,
            headers=self.headers,
            timeout=self.timeout,
            sse_read_timeout=self.sse_read_timeout,
        )

        # Enter the context manager
        read_stream, write_stream = await self._sse_ctx.__aenter__()

        # Return the streams
        return (read_stream, write_stream)

    async def _close_connection(self, connection: tuple[Any, Any]) -> None:
        """Close the SSE connection.

        Args:
            connection: The connection to close (ignored, we use the context manager)
        """
        if self._sse_ctx:
            # Exit the context manager
            try:
                await self._sse_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing SSE context: {e}")
            finally:
                self._sse_ctx = None
