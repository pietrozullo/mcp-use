"""
Agent implementations for using MCP tools.

This module provides ready-to-use agent implementations
that are pre-configured for using MCP tools.
"""

from .base import BaseAgent
from .mcpagent import MCPAgent

__all__ = [
    "BaseAgent",
    "MCPAgent",
]
