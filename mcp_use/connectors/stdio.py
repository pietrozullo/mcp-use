"""
StdIO connector for MCP implementations.

This module provides a connector for communicating with MCP implementations
through the standard input/output streams.
"""

import os
import shutil
import subprocess
import sys
import platform

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
        resolve_command_path: bool = True,
    ):
        """Initialize a new stdio connector.

        Args:
            command: The command to execute.
            args: Optional command line arguments.
            env: Optional environment variables.
            errlog: Stream to write error output to.
            resolve_command_path: Whether to resolve the command path at initialization time.
                                 This can help catch command not found errors early.
        """
        super().__init__()
        self.command = command
        self.args = args or []  # Ensure args is never None
        self.env = env
        self.errlog = errlog

        # Optionally resolve the command path at initialization time
        if resolve_command_path:
            try:
                resolved_command = self._find_command_path(command)
                if resolved_command != command:
                    logger.debug(f"Resolved command '{command}' to '{resolved_command}'")
                    self.command = resolved_command
            except Exception as e:
                # Just log a warning, don't fail initialization
                logger.warning(f"Failed to resolve command path for '{command}': {e}")

    def _find_command_path(self, command: str) -> str:
        """Find the full path to a command, handling platform-specific issues.

        Args:
            command: The command to find

        Returns:
            The full path to the command or the original command if not found

        This is particularly important for Windows where commands like 'npx' might not be
        directly accessible without specifying the full path or using the appropriate extension.
        """
        # If it's already a full path, return it
        if os.path.isfile(command) and os.access(command, os.X_OK):
            return command

        # On Windows, we need to handle this differently
        if platform.system() == "Windows":
            # Try to find the command using 'where' (Windows equivalent of 'which')
            try:
                # Check if it's in PATH
                result = subprocess.run(
                    ["where", command],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0 and result.stdout.strip():
                    # Return the first match
                    return result.stdout.strip().split('\n')[0]

                # Special handling for npm/npx commands which might be in node_modules
                if command in ["npm", "npx"]:
                    # Common locations for npm/npx on Windows
                    possible_paths = [
                        os.path.join(os.environ.get("APPDATA", ""), "npm"),
                        os.path.join(os.environ.get("ProgramFiles", ""), "nodejs"),
                        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "nodejs"),
                        # Add Node.js global install directory
                        os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Roaming", "npm"),
                        # Add potential npm global prefix locations
                        os.path.join(os.environ.get("USERPROFILE", ""), ".npm-global"),
                        os.path.join(os.environ.get("USERPROFILE", ""), "npm-global"),
                    ]

                    # Try to get npm prefix path
                    try:
                        npm_prefix_result = subprocess.run(
                            ["cmd", "/c", "npm config get prefix"],
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        if npm_prefix_result.returncode == 0 and npm_prefix_result.stdout.strip():
                            npm_prefix = npm_prefix_result.stdout.strip()
                            possible_paths.append(npm_prefix)
                            possible_paths.append(os.path.join(npm_prefix, "node_modules", ".bin"))
                    except Exception as e:
                        logger.debug(f"Failed to get npm prefix: {e}")

                    # Check for various extensions on Windows
                    extensions = [".cmd", ".exe", ".bat", ".ps1"]
                    for base_path in possible_paths:
                        # First check without extension
                        cmd_path = os.path.join(base_path, command)
                        if os.path.isfile(cmd_path):
                            return cmd_path

                        # Then check with various extensions
                        for ext in extensions:
                            cmd_path = os.path.join(base_path, f"{command}{ext}")
                            if os.path.isfile(cmd_path):
                                return cmd_path

                    # Also check with various extensions using 'where' command
                    for ext in extensions:
                        try:
                            result = subprocess.run(
                                ["where", f"{command}{ext}"],
                                capture_output=True,
                                text=True,
                                check=False
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                return result.stdout.strip().split('\n')[0]
                        except Exception as e:
                            logger.debug(f"Error checking for {command}{ext}: {e}")
            except Exception as e:
                logger.warning(f"Error finding command path for '{command}': {e}")
        else:
            # On Unix-like systems, use 'which'
            try:
                path = shutil.which(command)
                if path:
                    return path

                # Special handling for macOS
                if platform.system() == "Darwin":
                    # Check common macOS locations for npm/npx
                    if command in ["npm", "npx"]:
                        # Common locations for npm/npx on macOS
                        possible_paths = [
                            "/usr/local/bin",
                            "/opt/homebrew/bin",  # For Apple Silicon (M1/M2) Macs
                            os.path.expanduser("~/node_modules/.bin"),
                            os.path.expanduser("~/.npm-global/bin"),
                            "/opt/local/bin",  # MacPorts
                        ]

                        # Try to get npm prefix path
                        try:
                            npm_prefix_result = subprocess.run(
                                ["npm", "config", "get", "prefix"],
                                capture_output=True,
                                text=True,
                                check=False
                            )
                            if npm_prefix_result.returncode == 0 and npm_prefix_result.stdout.strip():
                                npm_prefix = npm_prefix_result.stdout.strip()
                                possible_paths.append(os.path.join(npm_prefix, "bin"))
                        except Exception as e:
                            logger.debug(f"Failed to get npm prefix on macOS: {e}")

                        # Check each possible path
                        for base_path in possible_paths:
                            cmd_path = os.path.join(base_path, command)
                            if os.path.isfile(cmd_path) and os.access(cmd_path, os.X_OK):
                                return cmd_path
            except Exception as e:
                logger.warning(f"Error finding command path for '{command}': {e}")

        # If we couldn't find it, return the original command
        # This will likely fail, but it's the best we can do
        return command

    async def connect(self) -> None:
        """Establish a connection to the MCP implementation."""
        if self._connected:
            logger.debug("Already connected to MCP implementation")
            return

        logger.debug(f"Connecting to MCP implementation: {self.command}")
        try:
            # Find the full path to the command
            command_path = self._find_command_path(self.command)
            if command_path != self.command:
                logger.debug(f"Using resolved command path: {command_path}")

            # Create server parameters
            server_params = StdioServerParameters(
                command=command_path, args=self.args, env=self.env
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
