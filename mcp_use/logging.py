"""
Logger module for mcp_use.

This module provides a centralized logging configuration for the mcp_use library,
with customizable log levels and formatters.
"""

import logging
import os
import sys

from langchain.globals import set_debug as langchain_set_debug

# Global debug flag - can be set programmatically or from environment
MCP_USE_DEBUG = False


class Logger:
    """Centralized logger for mcp_use.

    This class provides logging functionality with configurable levels,
    formatters, and handlers.
    """

    # Default log format
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Module-specific loggers
    _loggers = {}

    @classmethod
    def get_logger(cls, name: str = "mcp_use") -> logging.Logger:
        """Get a logger instance for the specified name.

        Args:
            name: Logger name, usually the module name (defaults to 'mcp_use')

        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]

        # Create new logger
        logger = logging.getLogger(name)
        cls._loggers[name] = logger

        return logger

    @classmethod
    def configure(
        cls,
        level: int | str = None,
        format_str: str | None = None,
        log_to_console: bool = True,
        log_to_file: str | None = None,
    ) -> None:
        """Configure the root mcp_use logger.

        Args:
            level: Log level (default: DEBUG if MCP_USE_DEBUG is 2,
            INFO if MCP_USE_DEBUG is 1,
            otherwise WARNING)
            format_str: Log format string (default: DEFAULT_FORMAT)
            log_to_console: Whether to log to console (default: True)
            log_to_file: Path to log file (default: None)
        """
        root_logger = cls.get_logger()

        # Set level based on debug settings if not explicitly provided
        if level is None:
            if MCP_USE_DEBUG == 2:
                level = logging.DEBUG
            elif MCP_USE_DEBUG == 1:
                level = logging.INFO
            else:
                level = logging.WARNING
        elif isinstance(level, str):
            level = getattr(logging, level.upper())

        root_logger.setLevel(level)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set formatter
        formatter = logging.Formatter(format_str or cls.DEFAULT_FORMAT)

        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Add file handler if requested
        if log_to_file:
            # Ensure directory exists
            log_dir = os.path.dirname(log_to_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = logging.FileHandler(log_to_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    @classmethod
    def set_debug(cls, debug_level: int = 2) -> None:
        """Set the debug flag and update the log level accordingly.

        Args:
            debug_level: Debug level (0=off, 1=info, 2=debug)
        """
        global MCP_USE_DEBUG
        MCP_USE_DEBUG = debug_level

        # Update log level for existing loggers
        if debug_level == 2:
            for logger in cls._loggers.values():
                logger.setLevel(logging.DEBUG)
                langchain_set_debug(True)
        elif debug_level == 1:
            for logger in cls._loggers.values():
                logger.setLevel(logging.INFO)
                langchain_set_debug(False)
        else:
            # Reset to default (WARNING)
            for logger in cls._loggers.values():
                logger.setLevel(logging.WARNING)
                langchain_set_debug(False)


# Check environment variable for debug flag
debug_env = os.environ.get("DEBUG", "").lower()
if debug_env == "2":
    MCP_USE_DEBUG = 2
elif debug_env == "1":
    MCP_USE_DEBUG = 1

# Configure default logger
Logger.configure()

logger = Logger.get_logger()
