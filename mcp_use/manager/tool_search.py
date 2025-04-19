import time

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from mcp_use.logging import logger


class ToolSearchInput(BaseModel):
    """Input for searching tools with semantic search"""

    query: str = Field(description="The search query to find relevant tools")
    top_k: int = Field(default=5, description="Number of top results to return")


class ToolSearch:
    """
    Provides efficient semantic search capabilities for MCP tools across all connectors.

    This implementation uses vector similarity for semantic search with caching of query results.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        use_caching: bool = True,
        use_fastembed: bool = False,
        fastembed_model: str = "BAAI/bge-small-en-v1.5",
    ):
        """
        Initialize the tool search engine.

        Args:
            embedding_dim: Dimension of embedding vectors
            use_caching: Whether to cache query results
            use_fastembed: Whether to use FastEmbed instead of SentenceTransformer
            fastembed_model: FastEmbed model name to use if use_fastembed is True
        """
        self.embedding_dim = embedding_dim
        self.use_caching = use_caching
        self.use_fastembed = use_fastembed
        self.fastembed_model = fastembed_model

        # Initialize semantic search components lazily (only when needed)
        self.model = None
        self.embedding_function = None

        # Main data storage
        self.tool_embeddings: dict[str, np.ndarray] = {}
        self.tools_by_name: dict[str, BaseTool] = {}
        self.server_by_tool: dict[str, str] = {}
        self.tool_texts: dict[str, str] = {}  # Store the raw text for search

        # Query cache
        self.query_cache: dict[str, list[tuple[BaseTool, str, float]]] = {}

        # Tracking
        self.is_indexed = False
        self.index_build_time = 0.0

    def _load_model(self) -> bool:
        """Load the embedding model for semantic search if not already loaded."""
        if self.model is not None:
            return True

        try:
            start_time = time.time()

            if self.use_fastembed:
                try:
                    from fastembed import TextEmbedding

                    self.model = TextEmbedding(model_name=self.fastembed_model)
                    # FastEmbed's embed returns a generator, so we need to convert to a list
                    self.embedding_function = lambda texts: np.array(list(self.model.embed(texts)))
                except ImportError:
                    logger.error(
                        "Failed to import fastembed. Please install it with 'pip install fastembed'"
                    )
                    return False
            else:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_function = self.model.encode

            load_time = time.time() - start_time
            logger.debug(f"Successfully initialized embedding model in {load_time:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

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

        # Generate embeddings (but don't block indexing if it fails)
        if self._load_model():
            tool_names = list(self.tool_texts.keys())
            tool_texts = [self.tool_texts[name] for name in tool_names]

            try:
                logger.debug(f"Generating embeddings for {len(tool_texts)} tools")
                embeddings = self.embedding_function(tool_texts)
                for name, embedding in zip(tool_names, embeddings, strict=True):
                    self.tool_embeddings[name] = embedding

                logger.debug(f"Successfully generated embeddings for {len(tool_texts)} tools")
            except Exception as e:
                logger.error(f"Failed to generate embeddings for tools: {e}")
                # Continue without embeddings

        self.is_indexed = True
        self.index_build_time = time.time() - start_time
        logger.info(
            f"Successfully indexed {len(self.tool_texts)} tools in {self.index_build_time:.2f}s"
        )

        # Log more detailed information about the index for debugging
        logger.debug(f"Tool embeddings size: {len(self.tool_embeddings)} tools")

    def _semantic_search(self, query: str, top_k: int = 10) -> list[tuple[BaseTool, str, float]]:
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

        # Check cache first
        cache_key = f"semantic:{query}:{top_k}"
        if self.use_caching and cache_key in self.query_cache:
            logger.debug("Using cached semantic search results")
            return self.query_cache[cache_key]

        # Ensure model and embeddings exist
        if not self._load_model() or not self.tool_embeddings:
            logger.warning("Semantic search not available")
            return []

        # Generate embedding for the query
        try:
            query_embedding = self.embedding_function([query])[0]
        except Exception as e:
            logger.error(f"Failed to encode search query: {e}")
            return []

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

        print(f"Query {query}")
        print(f"Results: {results}")

        # Cache results
        if self.use_caching:
            self.query_cache[cache_key] = results

        search_time = time.time() - start_time
        logger.debug(f"Semantic search returned {len(results)} results in {search_time:.3f}s")

        return results

    def search(self, query: str, top_k: int = 5) -> list[tuple[BaseTool, str, float]]:
        """
        Search for tools that match the query using semantic search.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            list of tuples containing (tool, server_name, score)
        """
        # Ensure the index is built
        if not self.is_indexed:
            logger.warning("Tool search index not ready")
            return []

        return self._semantic_search(query, top_k)
