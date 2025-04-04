"""
Pytest configuration file.

This module contains pytest fixtures and configuration for all tests.
"""

import os
import sys

import pytest

# Add the parent directory to the path so tests can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Fixture for mock session
@pytest.fixture
def mock_session():
    """Return a mock session object for testing."""
    from unittest.mock import AsyncMock, MagicMock

    # Create mock connector
    connector = MagicMock()
    connector.connect = AsyncMock()
    connector.disconnect = AsyncMock()
    connector.initialize = AsyncMock(return_value={"session_id": "test_session"})
    connector.tools = [{"name": "test_tool"}]
    connector.call_tool = AsyncMock(return_value={"result": "success"})

    return connector


# Register marks
def pytest_configure(config):
    """Register custom pytest marks."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
