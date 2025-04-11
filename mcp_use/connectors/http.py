"""
HTTP connector for MCP implementations.

This module provides a connector for communicating with MCP implementations
through HTTP APIs with SSE for transport.
"""

from mcp import ClientSession

from ..logging import logger
from ..task_managers import SseConnectionManager
from .base import BaseConnector


class HttpConnector(BaseConnector):
    """Connector for MCP implementations using HTTP transport with SSE.

    This connector uses HTTP/SSE to communicate with remote MCP implementations,
    using a connection manager to handle the proper lifecycle management.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
    ):
        """Initialize a new HTTP connector.

        Args:
            base_url: The base URL of the MCP HTTP API.
            auth_token: Optional authentication token.
            headers: Optional additional headers.
            timeout: Timeout for HTTP operations in seconds.
            sse_read_timeout: Timeout for SSE read operations in seconds.
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.headers = headers or {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout

    async def connect(self) -> None:
        """Establish a connection to the MCP implementation."""
        if self._connected:
            logger.debug("Already connected to MCP implementation")
            return

        logger.debug(f"Connecting to MCP implementation via HTTP/SSE: {self.base_url}")
        try:
            # Create the SSE connection URL
            sse_url = f"{self.base_url}"

            # Create and start the connection manager
            self._connection_manager = SseConnectionManager(
                sse_url, self.headers, self.timeout, self.sse_read_timeout
            )
            read_stream, write_stream = await self._connection_manager.start()

            # Create the client session
            self.client = ClientSession(read_stream, write_stream, sampling_callback=None)
            await self.client.__aenter__()

            # Mark as connected
            self._connected = True
            logger.debug(
                f"Successfully connected to MCP implementation via HTTP/SSE: {self.base_url}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to MCP implementation via HTTP/SSE: {e}")

            # Clean up any resources if connection failed
            await self._cleanup_resources()

            # Re-raise the original exception
            raise
