"""
HTTP connection management for MCP implementations.

This module provides a connection manager for HTTP-based MCP connections.
"""

import aiohttp

from ..logging import logger
from .base import ConnectionManager


class HttpConnectionManager(ConnectionManager[aiohttp.ClientSession]):
    """Connection manager for HTTP-based MCP connections.

    This class handles the lifecycle of HTTP client sessions, ensuring proper
    connection establishment and cleanup.
    """

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
    ):
        """Initialize a new HTTP connection manager.

        Args:
            base_url: The base URL for HTTP requests
            headers: Optional headers to include in all HTTP requests
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}

    async def _establish_connection(self) -> aiohttp.ClientSession:
        """Establish an HTTP client session.

        Returns:
            The established HTTP client session

        Raises:
            Exception: If session cannot be established
        """
        logger.debug(f"Creating HTTP client session for: {self.base_url}")
        try:
            session = aiohttp.ClientSession(headers=self.headers)
            return session
        except Exception as e:
            logger.error(f"Failed to create HTTP client session: {e}")
            raise

    async def _close_connection(self, connection: aiohttp.ClientSession) -> None:
        """Close the HTTP client session.

        Args:
            connection: The HTTP client session to close
        """
        try:
            logger.debug("Closing HTTP client session")
            await connection.close()
        except Exception as e:
            logger.warning(f"Error closing HTTP client session: {e}")
