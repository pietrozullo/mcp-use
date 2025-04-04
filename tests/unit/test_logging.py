"""
Unit tests for the logging module.
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

from mcp_use.logging import Logger, logger


class TestLogging(unittest.TestCase):
    """Tests for the logging module functionality."""

    def test_logger_instance(self):
        """Test that logger is a properly configured logging.Logger instance."""
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "mcp_use")

    def test_get_logger(self):
        """Test that get_logger returns a logger with the correct name."""
        test_logger = Logger.get_logger("test_module")
        self.assertIsInstance(test_logger, logging.Logger)
        self.assertEqual(test_logger.name, "test_module")

    def test_get_logger_caching(self):
        """Test that get_logger caches loggers."""
        logger1 = Logger.get_logger("test_cache")
        logger2 = Logger.get_logger("test_cache")

        self.assertIs(logger1, logger2)

    @patch("logging.StreamHandler")
    def test_configure_default(self, mock_stream_handler):
        """Test that configure correctly configures logging with default settings."""
        # Set up mocks
        mock_handler = MagicMock()
        mock_stream_handler.return_value = mock_handler

        # Reset the logger's handlers
        root_logger = Logger.get_logger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure logging with default settings
        Logger.configure()

        # Verify stream handler was created
        mock_stream_handler.assert_called_once()

        # Verify formatter was set
        self.assertIsNotNone(mock_handler.setFormatter.call_args)
        formatter = mock_handler.setFormatter.call_args[0][0]
        self.assertEqual(formatter._fmt, Logger.DEFAULT_FORMAT)

    @patch("logging.StreamHandler")
    def test_configure_debug_level(self, mock_stream_handler):
        """Test that configure correctly configures logging with debug level."""
        # Set up mocks
        mock_handler = MagicMock()
        mock_stream_handler.return_value = mock_handler

        # Reset the logger's handlers
        root_logger = Logger.get_logger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure logging with debug level
        Logger.configure(level=logging.DEBUG)

        # Verify level was set
        self.assertEqual(root_logger.level, logging.DEBUG)

        # Verify stream handler was created
        mock_stream_handler.assert_called_once()

    @patch("logging.StreamHandler")
    def test_configure_format(self, mock_stream_handler):
        """Test that configure correctly configures logging format."""
        # Set up mocks
        mock_handler = MagicMock()
        mock_stream_handler.return_value = mock_handler

        # Reset the logger's handlers
        root_logger = Logger.get_logger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure logging with a custom format
        test_format = "%(levelname)s - %(message)s"
        Logger.configure(format_str=test_format)

        # Verify formatter was set with the custom format
        self.assertIsNotNone(mock_handler.setFormatter.call_args)
        formatter = mock_handler.setFormatter.call_args[0][0]
        self.assertEqual(formatter._fmt, test_format)

    @patch("logging.FileHandler")
    def test_configure_file_logging(self, mock_file_handler):
        """Test configuring logging to a file."""
        # Set up mocks
        mock_handler = MagicMock()
        mock_file_handler.return_value = mock_handler

        # Reset the logger's handlers
        root_logger = Logger.get_logger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure logging with a file
        Logger.configure(log_to_file="/tmp/test.log")

        # Verify FileHandler was created
        mock_file_handler.assert_called_once_with("/tmp/test.log")

        # Verify formatter was set
        self.assertIsNotNone(mock_handler.setFormatter.call_args)
