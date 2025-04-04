"""
Connectors for various MCP transports.

This module provides interfaces for connecting to MCP implementations
through different transport mechanisms.
"""

from .base import ConnectionManager
from .sse import SseConnectionManager
from .stdio import StdioConnectionManager
from .websocket import WebSocketConnectionManager

__all__ = [
    "ConnectionManager",
    "HttpConnectionManager",
    "StdioConnectionManager",
    "WebSocketConnectionManager",
    "SseConnectionManager",
]
