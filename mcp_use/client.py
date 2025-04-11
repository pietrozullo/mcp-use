"""
Client for managing MCP servers and sessions.

This module provides a high-level client that manages MCP servers, connectors,
and sessions from configuration.
"""

import json
from typing import Any

from .config import create_connector_from_config, load_config_file
from .logging import logger
from .session import MCPSession


class MCPClient:
    """Client for managing MCP servers and sessions.

    This class provides a unified interface for working with MCP servers,
    handling configuration, connector creation, and session management.
    """

    def __init__(
        self,
        config: str | dict[str, Any] | None = None,
    ) -> None:
        """Initialize a new MCP client.

        Args:
            config: Either a dict containing configuration or a path to a JSON config file.
                   If None, an empty configuration is used.
        """
        self.config: dict[str, Any] = {}
        self.sessions: dict[str, MCPSession] = {}
        self.active_sessions: list[str] = []

        # Load configuration if provided
        if config is not None:
            if isinstance(config, str):
                self.config = load_config_file(config)
            else:
                self.config = config

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MCPClient":
        """Create a MCPClient from a dictionary.

        Args:
            config: The configuration dictionary.
        """
        return cls(config=config)

    @classmethod
    def from_config_file(cls, filepath: str) -> "MCPClient":
        """Create a MCPClient from a configuration file.

        Args:
            filepath: The path to the configuration file.
        """
        return cls(config=load_config_file(filepath))

    def add_server(
        self,
        name: str,
        server_config: dict[str, Any],
    ) -> None:
        """Add a server configuration.

        Args:
            name: The name to identify this server.
            server_config: The server configuration.
        """
        if "mcpServers" not in self.config:
            self.config["mcpServers"] = {}

        self.config["mcpServers"][name] = server_config

    def remove_server(self, name: str) -> None:
        """Remove a server configuration.

        Args:
            name: The name of the server to remove.
        """
        if "mcpServers" in self.config and name in self.config["mcpServers"]:
            del self.config["mcpServers"][name]

            # If we removed an active session, remove it from active_sessions
            if name in self.active_sessions:
                self.active_sessions.remove(name)

    def get_server_names(self) -> list[str]:
        """Get the list of configured server names.

        Returns:
            List of server names.
        """
        return list(self.config.get("mcpServers", {}).keys())

    def save_config(self, filepath: str) -> None:
        """Save the current configuration to a file.

        Args:
            filepath: The path to save the configuration to.
        """
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)

    async def create_session(self, server_name: str, auto_initialize: bool = True) -> MCPSession:
        """Create a session for the specified server.

        Args:
            server_name: The name of the server to create a session for.

        Returns:
            The created MCPSession.

        Raises:
            ValueError: If no servers are configured or the specified server doesn't exist.
        """
        # Get server config
        servers = self.config.get("mcpServers", {})
        if not servers:
            raise ValueError("No MCP servers defined in config")

        if server_name not in servers:
            raise ValueError(f"Server '{server_name}' not found in config")

        server_config = servers[server_name]
        connector = create_connector_from_config(server_config)

        # Create the session
        session = MCPSession(connector)
        if auto_initialize:
            await session.initialize()
        self.sessions[server_name] = session

        # Add to active sessions
        if server_name not in self.active_sessions:
            self.active_sessions.append(server_name)

        return session

    async def create_all_sessions(
        self,
        auto_initialize: bool = True,
    ) -> dict[str, MCPSession]:
        """Create a session for the specified server.

        Args:
            auto_initialize: Whether to automatically initialize the session.

        Returns:
            The created MCPSession. If server_name is None, returns the first created session.

        Raises:
            ValueError: If no servers are configured or the specified server doesn't exist.
        """
        # Get server config
        servers = self.config.get("mcpServers", {})
        if not servers:
            raise ValueError("No MCP servers defined in config")

        # Create sessions for all servers
        for name in servers:
            session = await self.create_session(name, auto_initialize)
            if auto_initialize:
                await session.initialize()

        return self.sessions

    def get_session(self, server_name: str) -> MCPSession:
        """Get an existing session.

        Args:
            server_name: The name of the server to get the session for.
                        If None, uses the first active session.

        Returns:
            The MCPSession for the specified server.

        Raises:
            ValueError: If no active sessions exist or the specified session doesn't exist.
        """
        if server_name not in self.sessions:
            raise ValueError(f"No session exists for server '{server_name}'")

        return self.sessions[server_name]

    def get_all_active_sessions(self) -> dict[str, MCPSession]:
        """Get all active sessions.

        Returns:
            Dictionary mapping server names to their MCPSession instances.
        """
        return {name: self.sessions[name] for name in self.active_sessions if name in self.sessions}

    async def close_session(self, server_name: str) -> None:
        """Close a session.

        Args:
            server_name: The name of the server to close the session for.
                        If None, uses the first active session.

        Raises:
            ValueError: If no active sessions exist or the specified session doesn't exist.
        """
        # Check if the session exists
        if server_name not in self.sessions:
            logger.warning(f"No session exists for server '{server_name}', nothing to close")
            return

        # Get the session
        session = self.sessions[server_name]

        try:
            # Disconnect from the session
            logger.debug(f"Closing session for server '{server_name}'")
            await session.disconnect()
        except Exception as e:
            logger.error(f"Error closing session for server '{server_name}': {e}")
        finally:
            # Remove the session regardless of whether disconnect succeeded
            del self.sessions[server_name]

            # Remove from active_sessions
            if server_name in self.active_sessions:
                self.active_sessions.remove(server_name)

    async def close_all_sessions(self) -> None:
        """Close all active sessions.

        This method ensures all sessions are closed even if some fail.
        """
        # Get a list of all session names first to avoid modification during iteration
        server_names = list(self.sessions.keys())
        errors = []

        for server_name in server_names:
            try:
                logger.debug(f"Closing session for server '{server_name}'")
                await self.close_session(server_name)
            except Exception as e:
                error_msg = f"Failed to close session for server '{server_name}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Log summary if there were errors
        if errors:
            logger.error(f"Encountered {len(errors)} errors while closing sessions")
        else:
            logger.debug("All sessions closed successfully")
