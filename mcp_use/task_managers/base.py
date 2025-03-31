"""
Connection management for MCP implementations.

This module provides an abstract base class for different types of connection
managers used in MCP connectors.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..logging import logger

# Type variable for connection types
T = TypeVar("T")


class ConnectionManager(Generic[T], ABC):
    """Abstract base class for connection managers.

    This class defines the interface for different types of connection managers
    used with MCP connectors.
    """

    def __init__(self):
        """Initialize a new connection manager."""
        self._ready_event = asyncio.Event()
        self._done_event = asyncio.Event()
        self._exception: Exception | None = None
        self._connection: T | None = None
        self._task: asyncio.Task | None = None

    @abstractmethod
    async def _establish_connection(self) -> T:
        """Establish the connection.

        This method should be implemented by subclasses to establish
        the specific type of connection needed.

        Returns:
            The established connection.

        Raises:
            Exception: If connection cannot be established.
        """
        pass

    @abstractmethod
    async def _close_connection(self, connection: T) -> None:
        """Close the connection.

        This method should be implemented by subclasses to close
        the specific type of connection.

        Args:
            connection: The connection to close.
        """
        pass

    async def start(self) -> T:
        """Start the connection manager and establish a connection.

        Returns:
            The established connection.

        Raises:
            Exception: If connection cannot be established.
        """
        # Reset state
        self._ready_event.clear()
        self._done_event.clear()
        self._exception = None

        # Create a task to establish and maintain the connection
        self._task = asyncio.create_task(
            self._connection_task(), name=f"{self.__class__.__name__}_task"
        )

        # Wait for the connection to be ready or fail
        await self._ready_event.wait()

        # If there was an exception, raise it
        if self._exception:
            raise self._exception

        # Return the connection
        if self._connection is None:
            raise RuntimeError("Connection was not established")
        return self._connection

    async def stop(self) -> None:
        """Stop the connection manager and close the connection."""
        if self._task and not self._task.done():
            # Cancel the task
            logger.debug(f"Cancelling {self.__class__.__name__} task")
            self._task.cancel()

            # Wait for it to complete
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug(f"{self.__class__.__name__} task cancelled successfully")
            except Exception as e:
                logger.warning(f"Error stopping {self.__class__.__name__} task: {e}")

        # Wait for the connection to be done
        await self._done_event.wait()
        logger.debug(f"{self.__class__.__name__} task completed")

    async def _connection_task(self) -> None:
        """Run the connection task.

        This task establishes and maintains the connection until cancelled.
        """
        logger.debug(f"Starting {self.__class__.__name__} task")
        try:
            # Establish the connection
            self._connection = await self._establish_connection()
            logger.debug(f"{self.__class__.__name__} connected successfully")

            # Signal that the connection is ready
            self._ready_event.set()

            # Wait indefinitely until cancelled
            try:
                # This keeps the connection open until cancelled
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Expected when stopping
                logger.debug(f"{self.__class__.__name__} task received cancellation")
                pass

        except Exception as e:
            # Store the exception
            self._exception = e
            logger.error(f"Error in {self.__class__.__name__} task: {e}")

            # Signal that the connection is ready (with error)
            self._ready_event.set()

        finally:
            # Close the connection if it was established
            if self._connection is not None:
                try:
                    await self._close_connection(self._connection)
                except Exception as e:
                    logger.warning(f"Error closing connection in {self.__class__.__name__}: {e}")
                self._connection = None

            # Signal that the connection is done
            self._done_event.set()
