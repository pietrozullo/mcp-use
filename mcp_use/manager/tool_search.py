import re
import time
from typing import Any

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from mcp_use.logging import logger


class ToolSearchInput(BaseModel):
    """Input for searching tools with semantic search"""

    query: str = Field(description="The search query to find relevant tools")
    top_k: int = Field(default=5, description="Number of top results to return")
    use_fast_search: bool = Field(
        default=True, description="Whether to use fast keyword search instead of semantic search"
    )


class ToolSearch:
    """
    Provides efficient semantic and keyword search capabilities for MCP tools across all connectors.

    This implementation uses a hybrid approach with:
    1. Inverted index for fast keyword search
    2. Direct vector similarity for semantic search
    3. Caching of query results for repeated queries
    """

    def __init__(self, embedding_dim: int = 384, use_caching: bool = True):
        """
        Initialize the tool search engine.

        Args:
            embedding_dim: Dimension of embedding vectors
            use_caching: Whether to cache query results
        """
        self.embedding_dim = embedding_dim
        self.use_caching = use_caching

        # Initialize semantic search components lazily (only when needed)
        self.model = None

        # Main data storage
        self.tool_embeddings: dict[str, np.ndarray] = {}
        self.tools_by_name: dict[str, BaseTool] = {}
        self.server_by_tool: dict[str, str] = {}
        self.tool_texts: dict[str, str] = {}  # Store the raw text for keyword search

        # Inverted index for fast keyword search
        self.keyword_index: dict[str, list[str]] = {}  # Maps keywords to tool names

        # Query cache
        self.query_cache: dict[str, list[tuple[BaseTool, str, float]]] = {}

        # Tracking
        self.is_indexed = False
        self.index_build_time = 0.0
        self.search_stats = {
            "keyword_searches": 0,
            "semantic_searches": 0,
            "cache_hits": 0,
            "total_search_time": 0.0,
        }

    def _load_model(self) -> bool:
        """Load the sentence transformer model for semantic search if not already loaded."""
        if self.model is not None:
            return True

        try:
            start_time = time.time()
            # Use a small, efficient model for faster performance
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            load_time = time.time() - start_time
            logger.debug(f"Successfully initialized sentence transformer model in {load_time:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            return False

    def _build_keyword_index(self) -> None:
        """Build an inverted index for keyword search."""
        start_time = time.time()
        self.keyword_index = {}

        # Process each tool
        for tool_name, tool_text in self.tool_texts.items():
            # Extract keywords (words with 3+ characters)
            words = re.findall(r"\b\w{3,}\b", tool_text.lower())

            # Add to inverted index
            for word in words:
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                if tool_name not in self.keyword_index[word]:
                    self.keyword_index[word].append(tool_name)

        build_time = time.time() - start_time
        logger.debug(
            f"Built keyword index with {len(self.keyword_index)} unique "
            f"keywords in {build_time:.2f}s"
        )

    async def index_tools(self, server_tools: dict[str, list[BaseTool]]) -> None:
        """
        Index all tools from all servers for search.

        Args:
            server_tools: dictionary mapping server names to their tools
        """
        start_time = time.time()
        logger.debug(f"Starting tool indexing with {len(server_tools)} servers")

        # Clear previous indexes
        self.tool_embeddings = {}
        self.tools_by_name = {}
        self.server_by_tool = {}
        self.tool_texts = {}
        self.keyword_index = {}
        self.query_cache = {}  # Clear cache when index changes

        # Collect all tools and their descriptions
        for server_name, tools in server_tools.items():
            logger.debug(f"Indexing server '{server_name}' with {len(tools)} tools")
            for tool in tools:
                # Create a rich text representation for better search
                tool_text = f"{tool.name}: {tool.description}"

                # Store tool information
                self.tools_by_name[tool.name] = tool
                self.server_by_tool[tool.name] = server_name
                self.tool_texts[tool.name] = (
                    tool_text.lower()
                )  # Store lowercase for case-insensitive search

        if not self.tool_texts:
            logger.warning("No tools to index for search")
            self.is_indexed = False
            return

        # Build keyword index immediately
        self._build_keyword_index()

        # Generate embeddings (but don't block indexing if it fails)
        if self._load_model():
            tool_names = list(self.tool_texts.keys())
            tool_texts = [self.tool_texts[name] for name in tool_names]

            try:
                logger.debug(f"Generating embeddings for {len(tool_texts)} tools")
                embeddings = self.model.encode(tool_texts)
                for name, embedding in zip(tool_names, embeddings, strict=True):
                    self.tool_embeddings[name] = embedding

                logger.debug(f"Successfully generated embeddings for {len(tool_texts)} tools")
            except Exception as e:
                logger.error(f"Failed to generate embeddings for tools: {e}")
                # Continue without embeddings - we can still use keyword search

        self.is_indexed = True
        self.index_build_time = time.time() - start_time
        logger.info(
            f"Successfully indexed {len(self.tool_texts)} tools in {self.index_build_time:.2f}s"
        )

        # Log more detailed information about the index for debugging
        logger.debug(f"Keyword index size: {len(self.keyword_index)} unique terms")
        logger.debug(f"Tool embeddings size: {len(self.tool_embeddings)} tools")

    def _keyword_search(self, query: str, top_k: int = 5) -> list[tuple[BaseTool, str, float]]:
        """
        Perform a fast keyword-based search on the indexed tools using the inverted index.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            list of tuples containing (tool, server_name, score)
        """
        start_time = time.time()
        logger.debug(f"Performing keyword search with query: '{query}'")

        self.search_stats["keyword_searches"] += 1

        if not self.is_indexed or not self.tool_texts:
            logger.warning("Tool search index not ready for keyword search")
            return []

        # Check cache first
        cache_key = f"keyword:{query}:{top_k}"
        if self.use_caching and cache_key in self.query_cache:
            self.search_stats["cache_hits"] += 1
            logger.debug("Using cached keyword search results")
            return self.query_cache[cache_key]

        # Prepare query
        query = query.lower()

        # Extract query terms
        query_terms = re.findall(r"\b\w{3,}\b", query)

        # Score counter for each tool
        scores: dict[str, float] = {}

        # First check for exact phrase matches
        for tool_name, tool_text in self.tool_texts.items():
            # Higher score for exact phrase match
            if query in tool_text:
                scores[tool_name] = scores.get(tool_name, 0) + 0.8

        # Then check for term matches using inverted index
        for term in query_terms:
            if term in self.keyword_index:
                # Get tools containing this term
                matching_tools = self.keyword_index[term]

                for tool_name in matching_tools:
                    # Score based on where the term appears
                    if term in tool_name.lower():
                        scores[tool_name] = (
                            scores.get(tool_name, 0) + 0.5
                        )  # Higher score for matches in tool name
                    else:
                        scores[tool_name] = (
                            scores.get(tool_name, 0) + 0.3
                        )  # Lower score for matches in description

        # Normalize scores to 0-1 range
        for tool_name in scores:
            scores[tool_name] = min(scores[tool_name], 1.0)

        # Sort by score and get top_k results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Filter out zero scores
        sorted_results = [(name, score) for name, score in sorted_results if score > 0]

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

        search_time = time.time() - start_time
        self.search_stats["total_search_time"] += search_time
        logger.info(f"Keyword search returned {len(results)} results in {search_time:.3f}s")

        return results

    def _semantic_search(self, query: str, top_k: int = 5) -> list[tuple[BaseTool, str, float]]:
        """
        Perform semantic search using direct vector similarity computation.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            list of tuples containing (tool, server_name, score)
        """
        start_time = time.time()
        logger.debug(f"Performing semantic search with query: '{query}'")

        self.search_stats["semantic_searches"] += 1

        # Check cache first
        cache_key = f"semantic:{query}:{top_k}"
        if self.use_caching and cache_key in self.query_cache:
            self.search_stats["cache_hits"] += 1
            logger.debug("Using cached semantic search results")
            return self.query_cache[cache_key]

        # Ensure model and embeddings exist
        if not self._load_model() or not self.tool_embeddings:
            logger.warning("Semantic search not available, falling back to keyword search")
            return self._keyword_search(query, top_k)

        # Generate embedding for the query
        try:
            query_embedding = self.model.encode(query)
        except Exception as e:
            logger.error(f"Failed to encode search query: {e}")
            return self._keyword_search(query, top_k)

        # Direct vector similarity computation
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

        search_time = time.time() - start_time
        self.search_stats["total_search_time"] += search_time
        logger.debug(f"Semantic search returned {len(results)} results in {search_time:.3f}s")

        return results

    def search(
        self, query: str, top_k: int = 5, use_fast_search: bool = True
    ) -> list[tuple[BaseTool, str, float]]:
        """
        Search for tools that match the query using either keyword or semantic search.

        Args:
            query: The search query
            top_k: Number of top results to return
            use_fast_search: Whether to use fast keyword search (True) or semantic search (False)

        Returns:
            list of tuples containing (tool, server_name, score)
        """
        # Ensure the index is built
        if not self.is_indexed:
            logger.warning("Tool search index not ready")
            return []

        # Choose search method
        if use_fast_search:
            return self._keyword_search(query, top_k)
        else:
            return self._semantic_search(query, top_k)

    def hybrid_search(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> list[tuple[BaseTool, str, float]]:
        """
        Perform hybrid search combining keyword and semantic search results.

        Args:
            query: The search query
            top_k: Number of top results to return
            alpha: Weight of keyword search (0.0-1.0). Higher values favor keyword results.

        Returns:
            list of tuples containing (tool, server_name, score)
        """
        # Check cache first
        cache_key = f"hybrid:{query}:{top_k}:{alpha}"
        if self.use_caching and cache_key in self.query_cache:
            self.search_stats["cache_hits"] += 1
            logger.debug("Using cached hybrid search results")
            return self.query_cache[cache_key]

        # Get results from both methods
        keyword_results = self._keyword_search(
            query, top_k=top_k * 2
        )  # Get more for better merging
        semantic_results = (
            self._semantic_search(query, top_k=top_k * 2) if self.tool_embeddings else []
        )

        # If one method fails, use the other
        if not keyword_results and not semantic_results:
            return []
        if not keyword_results:
            return semantic_results[:top_k]
        if not semantic_results:
            return keyword_results[:top_k]

        # Combine results with weighted scores
        combined_scores: dict[str, tuple[BaseTool, str, float]] = {}

        # Process keyword results
        for tool, server, score in keyword_results:
            combined_scores[tool.name] = (tool, server, score * alpha)

        # Process semantic results, adding to keyword scores
        for tool, server, score in semantic_results:
            if tool.name in combined_scores:
                existing_score = combined_scores[tool.name][2]
                combined_scores[tool.name] = (tool, server, existing_score + score * (1 - alpha))
            else:
                combined_scores[tool.name] = (tool, server, score * (1 - alpha))

        # Sort and limit results
        sorted_results = sorted(combined_scores.values(), key=lambda x: x[2], reverse=True)[:top_k]

        # Cache results
        if self.use_caching:
            self.query_cache[cache_key] = sorted_results

        return sorted_results

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the search engine."""
        return {
            "indexed_tools": len(self.tool_texts),
            "indexed_servers": len(set(self.server_by_tool.values())),
            "has_embeddings": len(self.tool_embeddings) > 0,
            "index_build_time": self.index_build_time,
            "keyword_searches": self.search_stats["keyword_searches"],
            "semantic_searches": self.search_stats["semantic_searches"],
            "cache_hits": self.search_stats["cache_hits"],
            "total_search_time": self.search_stats["total_search_time"],
            "cache_size": len(self.query_cache),
        }

    def optimize_for_tool_count(self) -> None:
        """
        Apply optimizations based on the number of indexed tools.

        For larger tool collections, this method applies additional optimizations.
        """
        tool_count = len(self.tool_texts)
        logger.debug(f"Optimizing search for {tool_count} tools")

        # For larger collections, we might want to adjust parameters
        if tool_count > 1000:
            # Potentially pre-compute frequently used queries
            common_queries = ["get", "create", "update", "delete", "list", "search", "find"]
            for query in common_queries:
                # Warm up the cache with common queries
                self._keyword_search(query, top_k=5)

        # Log optimization complete
        logger.debug("Search optimization complete")
