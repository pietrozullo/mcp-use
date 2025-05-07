import asyncio
import time
from typing import ClassVar

import numpy as np
from fastembed import TextEmbedding
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ...logging import logger
from .base_tool import MCPServerTool


class ToolSearchInput(BaseModel):
    """Input for searching for tools across MCP servers"""

    query: str = Field(description="The search query to find relevant tools")
    top_k: int = Field(
        default=100,
        description="The maximum number of tools to return (defaults to 100)",
    )


class SearchToolsTool(MCPServerTool):
    """Tool for searching for tools across all MCP servers using semantic search."""

    name: ClassVar[str] = "search_mcp_tools"
    description: ClassVar[str] = (
        "Search for relevant tools across all MCP servers using semantic search. "
        "Provide a description of the tool you think you might need to be able to perform "
        "the task you are assigned. Do not be too specific, the search will give you many "
        "options. It is important you search for the tool, not for the goal. "
        "If your first search doesn't yield relevant results, try using different keywords "
        "or more general terms."
    )
    args_schema: ClassVar[type[BaseModel]] = ToolSearchInput

    def __init__(self, server_manager):
        """Initialize with server manager and create a search tool."""
        super().__init__(server_manager)
        self._search_tool = ToolSearchEngine(server_manager=server_manager)

    async def _arun(self, query: str, top_k: int = 100) -> str:
        """Search for tools across all MCP servers using semantic search."""
        # Make sure the index is ready, and if not, allow the search_tools method to handle it
        # No need to manually check or build the index here as the search_tools method will do that

        # Perform search using our search tool instance
        results = await self._search_tool.search_tools(
            query, top_k=top_k, active_server=self.server_manager.active_server
        )
        return self.format_search_results(results)

    def _run(self, query: str, top_k: int = 100) -> str:
        """Synchronous version that raises a NotImplementedError - use _arun instead."""
        raise NotImplementedError("SearchToolsTool requires async execution. Use _arun instead.")

    def format_search_results(self, results: list[tuple[BaseTool, str, float]]) -> str:
        """Format search results in a consistent format."""

        # Only show top_k results
        results = results

        formatted_output = "Search results\n\n"

        for i, (tool, server_name, score) in enumerate(results):
            # Format score as percentage
            if i < 5:
                score_pct = f"{score * 100:.1f}%"
                logger.info(f"{i}: {tool.name} ({score_pct} match)")
            formatted_output += f"[{i + 1}] Tool: {tool.name} ({score_pct} match)\n"
            formatted_output += f"    Server: {server_name}\n"
            formatted_output += f"    Description: {tool.description}\n\n"

        # Add footer with information about how to use the results
        formatted_output += (
            "\nTo use a tool, connect to the appropriate server first, then invoke the tool."
        )

        return formatted_output


class ToolSearchEngine:
    """
    Provides semantic search capabilities for MCP tools.
    Uses vector similarity for semantic search with optional result caching.
    """

    def __init__(self, server_manager=None, use_caching: bool = True):
        """
        Initialize the tool search engine.

        Args:
            server_manager: The ServerManager instance to get tools from
            use_caching: Whether to cache query results
        """
        self.server_manager = server_manager
        self.use_caching = use_caching
        self.is_indexed = False

        # Initialize model components (loaded on demand)
        self.model = None
        self.embedding_function = None

        # Data storage
        self.tool_embeddings = {}  # Maps tool name to embedding vector
        self.tools_by_name = {}  # Maps tool name to tool instance
        self.server_by_tool = {}  # Maps tool name to server name
        self.tool_texts = {}  # Maps tool name to searchable text
        self.query_cache = {}  # Caches search results by query

    def _load_model(self) -> bool:
        """Load the embedding model for semantic search if not already loaded."""
        if self.model is not None:
            return True

        try:
            self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self.embedding_function = lambda texts: list(self.model.embed(texts))
            return True
        except Exception:
            return False

    async def start_indexing(self) -> None:
        """Index the tools from the server manager."""
        if not self.server_manager:
            return

        # Get tools from server manager
        server_tools = self.server_manager._server_tools

        if not server_tools:
            # Try to prefetch tools first
            if hasattr(self.server_manager, "_prefetch_server_tools"):
                await self.server_manager._prefetch_server_tools()
                server_tools = self.server_manager._server_tools

        if server_tools:
            await self.index_tools(server_tools)

    async def index_tools(self, server_tools: dict[str, list[BaseTool]]) -> None:
        """
        Index all tools from all servers for search.

        Args:
            server_tools: dictionary mapping server names to their tools
        """
        # Clear previous indexes
        self.tool_embeddings = {}
        self.tools_by_name = {}
        self.server_by_tool = {}
        self.tool_texts = {}
        self.query_cache = {}
        self.is_indexed = False

        # Collect all tools and their descriptions
        for server_name, tools in server_tools.items():
            for tool in tools:
                # Create text representation for search
                tool_text = f"{tool.name}: {tool.description}"

                # Store tool information
                self.tools_by_name[tool.name] = tool
                self.server_by_tool[tool.name] = server_name
                self.tool_texts[tool.name] = tool_text.lower()  # For case-insensitive search

        if not self.tool_texts:
            return

        # Generate embeddings
        if self._load_model():
            tool_names = list(self.tool_texts.keys())
            tool_texts = [self.tool_texts[name] for name in tool_names]

            try:
                embeddings = self.embedding_function(tool_texts)
                for name, embedding in zip(tool_names, embeddings, strict=True):
                    self.tool_embeddings[name] = embedding

                # Mark as indexed if we successfully embedded tools
                self.is_indexed = len(self.tool_embeddings) > 0
            except Exception:
                return

    def search(self, query: str, top_k: int = 5) -> list[tuple[BaseTool, str, float]]:
        """
        Search for tools that match the query using semantic search.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            list of tuples containing (tool, server_name, score)
        """
        if not self.is_indexed:
            return []

        # Check cache first
        cache_key = f"semantic:{query}:{top_k}"
        if self.use_caching and cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # Ensure model and embeddings exist
        if not self._load_model() or not self.tool_embeddings:
            return []

        # Generate embedding for the query
        try:
            query_embedding = self.embedding_function([query])[0]
        except Exception:
            return []

        # Calculate similarity scores
        scores = {}
        for tool_name, embedding in self.tool_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            scores[tool_name] = float(similarity)

        # Sort by score and get top_k results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Format results
        results = []
        for tool_name, score in sorted_results:
            tool = self.tools_by_name.get(tool_name)
            server_name = self.server_by_tool.get(tool_name)
            if tool and server_name:
                results.append((tool, server_name, score))

        # Cache results
        if self.use_caching:
            self.query_cache[cache_key] = results

        return results

    async def search_tools(self, query: str, top_k: int = 100, active_server: str = None) -> str:
        """
        Search for tools across all MCP servers using semantic search.

        Args:
            query: The search query to find relevant tools
            top_k: Number of top results to return
            active_server: Name of the currently active server (for highlighting)

        Returns:
            String with formatted search results
        """
        # Ensure the index is built or build it
        if not self.is_indexed:
            # Try to build the index
            if self.server_manager and self.server_manager._server_tools:
                await self.index_tools(self.server_manager._server_tools)
            else:
                # If we don't have server_manager or tools, try to index directly
                await self.start_indexing()

            # Wait for indexing to complete (maximum 10 seconds)
            start_time = time.time()
            timeout = 10  # seconds
            while not self.is_indexed and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.5)

            # If still not indexed, return a friendly message
            if not self.is_indexed:
                return (
                    "I'm still preparing the tool index. Please try your search again in a moment. "
                    "This usually takes just a few seconds to complete."
                )

        # If the server manager has an active server but it wasn't provided, use it
        if (
            active_server is None
            and self.server_manager
            and hasattr(self.server_manager, "active_server")
        ):
            active_server = self.server_manager.active_server

        results = self.search(query, top_k=top_k)
        if not results:
            return (
                "No relevant tools found. The search provided no results. "
                "You can try searching again with different keywords. "
                "Try using more general terms or focusing on the capability you need."
            )

        # If there's an active server, mark it in the results
        if active_server:
            # Create a new results list with marked active server
            marked_results = []
            for tool, server_name, score in results:
                # If this is the active server, add "(ACTIVE)" marker
                display_server = (
                    f"{server_name} (ACTIVE)" if server_name == active_server else server_name
                )
                marked_results.append((tool, display_server, score))
            results = marked_results

        # Format and return the results
        return results
