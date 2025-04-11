"""
StdIO connector for MCP implementations.

This module provides a connector for communicating with MCP implementations
through the standard input/output streams.
"""

import sys

from mcp import ClientSession, StdioServerParameters

from ..logging import logger
from ..task_managers import StdioConnectionManager
from .base import BaseConnector


class StdioConnector(BaseConnector):
    """Connector for MCP implementations using stdio transport.

    This connector uses the stdio transport to communicate with MCP implementations
    that are executed as child processes. It uses a connection manager to handle
    the proper lifecycle management of the stdio client.
    """

    def __init__(
        self,
        command: str = "npx",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        errlog=sys.stderr,
    ):
        """Initialize a new stdio connector.

        Args:
            command: The command to execute.
            args: Optional command line arguments.
            env: Optional environment variables.
            errlog: Stream to write error output to.
        """
        super().__init__()
        self.command = command
        self.args = args or []  # Ensure args is never None
        self.env = env
        self.errlog = errlog

    async def connect(self) -> None:
        """Establish a connection to the MCP implementation."""
        if self._connected:
            logger.debug("Already connected to MCP implementation")
            return

        logger.debug(f"Connecting to MCP implementation: {self.command}")
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=self.command, args=self.args, env=self.env
            )

            # Create and start the connection manager
            self._connection_manager = StdioConnectionManager(server_params, self.errlog)
            read_stream, write_stream = await self._connection_manager.start()

            # Create the client session
            self.client = ClientSession(read_stream, write_stream, sampling_callback=None)
            await self.client.__aenter__()

            # Mark as connected
            self._connected = True
            logger.debug(f"Successfully connected to MCP implementation: {self.command}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP implementation: {e}")

            # Clean up any resources if connection failed
            await self._cleanup_resources()

            # Re-raise the original exception
            raise
