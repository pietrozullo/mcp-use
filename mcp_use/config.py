"""
Configuration loader for MCP session.

This module provides functionality to load MCP configuration from JSON files.
"""

import json
from typing import Any

from .connectors import BaseConnector, HttpConnector, StdioConnector, WebSocketConnector


def load_config_file(filepath: str) -> dict[str, Any]:
    """Load a configuration file.

    Args:
        filepath: Path to the configuration file

    Returns:
        The parsed configuration
    """
    with open(filepath) as f:
        return json.load(f)


def create_connector_from_config(server_config: dict[str, Any]) -> BaseConnector:
    """Create a connector based on server configuration.

    Args:
        server_config: The server configuration section

    Returns:
        A configured connector instance
    """
    # Stdio connector (command-based)
    if "command" in server_config and "args" in server_config:
        return StdioConnector(
            command=server_config["command"],
            args=server_config["args"],
            env=server_config.get("env", None),
        )

    # HTTP connector
    elif "url" in server_config:
        return HttpConnector(
            base_url=server_config["url"],
            headers=server_config.get("headers", None),
            auth_token=server_config.get("auth_token", None),
        )

    # WebSocket connector
    elif "ws_url" in server_config:
        return WebSocketConnector(
            url=server_config["ws_url"],
            headers=server_config.get("headers", None),
            auth_token=server_config.get("auth_token", None),
        )

    raise ValueError("Cannot determine connector type from config")
