"""
Unit tests for the config module.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from mcp_use.config import create_connector_from_config, load_config_file
from mcp_use.connectors import HttpConnector, StdioConnector, WebSocketConnector


class TestConfigLoading(unittest.TestCase):
    """Tests for configuration loading functions."""

    def test_load_config_file(self):
        """Test loading a configuration file."""
        test_config = {"mcpServers": {"test": {"url": "http://test.com"}}}

        # Create a temporary file with test config
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp:
            json.dump(test_config, temp)
            temp_path = temp.name

        try:
            # Test loading from file
            loaded_config = load_config_file(temp_path)
            self.assertEqual(loaded_config, test_config)
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    def test_load_config_file_nonexistent(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_config_file("/tmp/nonexistent_file.json")


class TestConnectorCreation(unittest.TestCase):
    """Tests for connector creation from configuration."""

    def test_create_http_connector(self):
        """Test creating an HTTP connector from config."""
        server_config = {
            "url": "http://test.com",
            "headers": {"Content-Type": "application/json"},
            "auth_token": "test_token",
        }

        connector = create_connector_from_config(server_config)

        self.assertIsInstance(connector, HttpConnector)
        self.assertEqual(connector.base_url, "http://test.com")
        self.assertEqual(
            connector.headers,
            {"Content-Type": "application/json", "Authorization": "Bearer test_token"},
        )
        self.assertEqual(connector.auth_token, "test_token")

    def test_create_http_connector_minimal(self):
        """Test creating an HTTP connector with minimal config."""
        server_config = {"url": "http://test.com"}

        connector = create_connector_from_config(server_config)

        self.assertIsInstance(connector, HttpConnector)
        self.assertEqual(connector.base_url, "http://test.com")
        self.assertEqual(connector.headers, {})
        self.assertIsNone(connector.auth_token)

    def test_create_websocket_connector(self):
        """Test creating a WebSocket connector from config."""
        server_config = {
            "ws_url": "ws://test.com",
            "headers": {"Content-Type": "application/json"},
            "auth_token": "test_token",
        }

        connector = create_connector_from_config(server_config)

        self.assertIsInstance(connector, WebSocketConnector)
        self.assertEqual(connector.url, "ws://test.com")
        self.assertEqual(
            connector.headers,
            {"Content-Type": "application/json", "Authorization": "Bearer test_token"},
        )
        self.assertEqual(connector.auth_token, "test_token")

    def test_create_websocket_connector_minimal(self):
        """Test creating a WebSocket connector with minimal config."""
        server_config = {"ws_url": "ws://test.com"}

        connector = create_connector_from_config(server_config)

        self.assertIsInstance(connector, WebSocketConnector)
        self.assertEqual(connector.url, "ws://test.com")
        self.assertEqual(connector.headers, {})
        self.assertIsNone(connector.auth_token)

    def test_create_stdio_connector(self):
        """Test creating a stdio connector from config."""
        server_config = {
            "command": "python",
            "args": ["-m", "mcp_server"],
            "env": {"DEBUG": "1"},
        }

        connector = create_connector_from_config(server_config)

        self.assertIsInstance(connector, StdioConnector)
        self.assertEqual(connector.command, "python")
        self.assertEqual(connector.args, ["-m", "mcp_server"])
        self.assertEqual(connector.env, {"DEBUG": "1"})

    def test_create_stdio_connector_minimal(self):
        """Test creating a stdio connector with minimal config."""
        server_config = {"command": "python", "args": ["-m", "mcp_server"]}

        connector = create_connector_from_config(server_config)

        self.assertIsInstance(connector, StdioConnector)
        self.assertEqual(connector.command, "python")
        self.assertEqual(connector.args, ["-m", "mcp_server"])
        self.assertIsNone(connector.env)

    def test_create_connector_invalid_config(self):
        """Test creating a connector with invalid config raises ValueError."""
        server_config = {"invalid": "config"}

        with self.assertRaises(ValueError) as context:
            create_connector_from_config(server_config)

        self.assertEqual(str(context.exception), "Cannot determine connector type from config")
