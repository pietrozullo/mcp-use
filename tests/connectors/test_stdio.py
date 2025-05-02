"""
Tests for the StdioConnector class.
"""

import os
import platform
import pytest
from unittest.mock import patch, MagicMock

from mcp_use.connectors.stdio import StdioConnector


class TestStdioConnector:
    """Tests for the StdioConnector class."""

    def test_init(self):
        """Test initialization of StdioConnector."""
        connector = StdioConnector(command="test_command", args=["arg1", "arg2"])
        assert connector.command == "test_command"
        assert connector.args == ["arg1", "arg2"]
        assert connector.env is None

    def test_init_with_resolve_path_disabled(self):
        """Test initialization with resolve_command_path=False."""
        with patch.object(StdioConnector, "_find_command_path") as mock_find:
            connector = StdioConnector(
                command="test_command",
                args=["arg1"],
                resolve_command_path=False
            )
            mock_find.assert_not_called()
            assert connector.command == "test_command"

    def test_init_with_resolve_path_enabled(self):
        """Test initialization with resolve_command_path=True."""
        with patch.object(StdioConnector, "_find_command_path") as mock_find:
            mock_find.return_value = "/resolved/path/test_command"
            connector = StdioConnector(
                command="test_command",
                args=["arg1"],
                resolve_command_path=True
            )
            mock_find.assert_called_once_with("test_command")
            assert connector.command == "/resolved/path/test_command"

    def test_find_command_path_windows(self):
        """Test _find_command_path on Windows."""
        if platform.system() != "Windows":
            pytest.skip("Test only applicable on Windows")

        with patch("platform.system", return_value="Windows"):
            with patch("subprocess.run") as mock_run:
                # Mock the 'where' command result
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "C:\\path\\to\\npx.cmd\n"
                mock_run.return_value = mock_result

                connector = StdioConnector(command="npx", resolve_command_path=False)
                result = connector._find_command_path("npx")

                assert result == "C:\\path\\to\\npx.cmd"
                mock_run.assert_called_with(
                    ["where", "npx"],
                    capture_output=True,
                    text=True,
                    check=False
                )

    def test_find_command_path_macos(self):
        """Test _find_command_path on macOS."""
        if platform.system() != "Darwin":
            pytest.skip("Test only applicable on macOS")

        with patch("shutil.which") as mock_which:
            # Test when shutil.which finds the command
            mock_which.return_value = "/usr/local/bin/npx"

            connector = StdioConnector(command="npx", resolve_command_path=False)
            result = connector._find_command_path("npx")

            assert result == "/usr/local/bin/npx"
            mock_which.assert_called_with("npx")
