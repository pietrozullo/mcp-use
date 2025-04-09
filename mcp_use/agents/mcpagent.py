"""
MCP: Main integration module with customizable system prompt.

This module provides the main MCPAgent class that integrates all components
to provide a simple interface for using MCP tools with different LLMs.
"""

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain.schema.language_model import BaseLanguageModel

from mcp_use.client import MCPClient
from mcp_use.connectors.base import BaseConnector
from mcp_use.session import MCPSession

from ..logging import logger
from .langchain_agent import LangChainAgent
from .prompts.default import DEFAULT_SYSTEM_PROMPT_TEMPLATE


class MCPAgent:
    """Main class for using MCP tools with various LLM providers.

    This class provides a unified interface for using MCP tools with different LLM providers
    through LangChain's agent framework, with customizable system prompts and conversation memory.
    """

    # Default system prompt template to use if none is provided
    DEFAULT_SYSTEM_PROMPT_TEMPLATE = DEFAULT_SYSTEM_PROMPT_TEMPLATE

    def __init__(
        self,
        llm: BaseLanguageModel,
        client: MCPClient | None = None,
        connectors: list[BaseConnector] | None = None,
        server_name: str | None = None,
        max_steps: int = 5,
        auto_initialize: bool = False,
        memory_enabled: bool = True,
        system_prompt: str | None = None,
        system_prompt_template: str | None = None,
        additional_instructions: str | None = None,
        disallowed_tools: list[str] | None = None,
    ):
        """Initialize a new MCPAgent instance.

        Args:
            llm: The LangChain LLM to use.
            client: The MCPClient to use. If provided, connector is ignored.
            connectors: A list of MCP connectors to use if client is not provided.
            server_name: The name of the server to use if client is provided.
            max_steps: The maximum number of steps to take.
            auto_initialize: Whether to automatically initialize the agent when run is called.
            memory_enabled: Whether to maintain conversation history for context.
            system_prompt: Complete system prompt to use (overrides template if provided).
            system_prompt_template: Template for system prompt with {tool_descriptions} placeholder.
            additional_instructions: Extra instructions to append to the system prompt.
            disallowed_tools: List of tool names that should not be available to the agent.
        """
        self.llm = llm
        self.client = client
        self.connectors = connectors
        self.server_name = server_name
        self.max_steps = max_steps
        self.auto_initialize = auto_initialize
        self.memory_enabled = memory_enabled
        self._initialized = False
        self._conversation_history: list[BaseMessage] = []
        self.disallowed_tools = disallowed_tools or []

        # System prompt configuration
        self.system_prompt = system_prompt
        self.system_prompt_template = system_prompt_template or self.DEFAULT_SYSTEM_PROMPT_TEMPLATE
        self.additional_instructions = additional_instructions

        # Either client or connector must be provided
        if not client and len(connectors) == 0:
            raise ValueError("Either client or connector must be provided")

        self._agent: LangChainAgent | None = None
        self._sessions: dict[str, MCPSession] | None = None
        self._system_message: SystemMessage | None = None

    async def initialize(self) -> None:
        """Initialize the MCP client and agent."""
        # If using client, get or create a session
        if self.client:
            # First try to get existing sessions
            self._sessions = self.client.get_all_active_sessions()

            # If no active sessions exist, create new ones
            if not self._sessions:
                self._sessions = await self.client.create_all_sessions()
            connectors_to_use = [session.connector for session in self._sessions.values()]
        else:
            # Using direct connector
            connectors_to_use = self.connectors
            await [c_to_use.connect() for c_to_use in connectors_to_use]
            await [c_to_use.initialize() for c_to_use in connectors_to_use]
        # Create the system message based on available tools
        await self._create_system_message(connectors_to_use)

        # Create the agent
        self._agent = LangChainAgent(
            connectors=connectors_to_use,
            llm=self.llm,
            max_steps=self.max_steps,
            system_message=(self._system_message.content if self._system_message else None),
            disallowed_tools=self.disallowed_tools,
        )

        # Initialize the agent
        await self._agent.initialize()
        self._initialized = True

    async def _create_system_message(self, connectors: list[BaseConnector]) -> None:
        """Create the system message based on available tools.

        Args:
            connector: The connector with available tools.
        """
        # If a complete system prompt was provided, use it directly
        if self.system_prompt:
            self._system_message = SystemMessage(content=self.system_prompt)
            return

        # Otherwise, build the system prompt from the template and tool descriptions
        tool_descriptions = []
        for connector in connectors:
            tools = connector.tools
            # Generate tool descriptions
            for tool in tools:
                # Skip disallowed tools
                if tool.name in self.disallowed_tools:
                    continue

                # Escape curly braces in the description by doubling them
                # (sometimes e.g. blender mcp they are used in the description)
                description = (
                    f"- {tool.name}: {tool.description.replace('{', '{{').replace('}', '}}')}"
                )
                tool_descriptions.append(description)

        # Format the system prompt template with tool descriptions
        system_prompt = self.system_prompt_template.format(
            tool_descriptions="\n".join(tool_descriptions)
        )

        # Add any additional instructions
        if self.additional_instructions:
            system_prompt += f"\n\n{self.additional_instructions}"

        # Create the system message
        self._system_message = SystemMessage(content=system_prompt)

    def get_conversation_history(self) -> list[BaseMessage]:
        """Get the current conversation history.

        Returns:
            The list of conversation messages.
        """
        return self._conversation_history

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history = []

        # Re-add the system message if it exists
        if self._system_message:
            self._conversation_history = [self._system_message]

    def add_to_history(self, message: BaseMessage) -> None:
        """Add a message to the conversation history.

        Args:
            message: The message to add.
        """
        self._conversation_history.append(message)

    def get_system_message(self) -> SystemMessage | None:
        """Get the current system message.

        Returns:
            The current system message, or None if not set.
        """
        return self._system_message

    def set_system_message(self, message: str) -> None:
        """Set a new system message.

        Args:
            message: The new system message content.
        """
        self._system_message = SystemMessage(content=message)

        # Update the agent if initialized
        if self._agent:
            self._agent.set_system_message(message)

    def set_disallowed_tools(self, disallowed_tools: list[str]) -> None:
        """Set the list of tools that should not be available to the agent.

        This will take effect the next time the agent is initialized.

        Args:
            disallowed_tools: List of tool names that should not be available.
        """
        self.disallowed_tools = disallowed_tools

        # If the agent is already initialized, we need to reinitialize it
        # to apply the changes to the available tools
        if self._initialized:
            logger.info(
                "Agent already initialized. Changes will take effect on next initialization."
            )
            # We don't automatically reinitialize here as it could be disruptive
            # to ongoing operations. The user can call initialize() explicitly if needed.

    def get_disallowed_tools(self) -> list[str]:
        """Get the list of tools that are not available to the agent.

        Returns:
            List of tool names that are not available.
        """
        return self.disallowed_tools

    async def run(
        self,
        query: str,
        max_steps: int | None = None,
        manage_connector: bool = True,
        external_history: list[BaseMessage] | None = None,
    ) -> str:
        """Run a query using the MCP tools.

        This method handles connecting to the MCP server, initializing the agent,
        running the query, and then cleaning up the connection.

        Args:
            query: The query to run.
            max_steps: Optional maximum number of steps to take.
            manage_connector: Whether to handle the connector lifecycle internally.
                If True, this method will connect, initialize, and disconnect from
                the connector automatically. If False, the caller is responsible
                for managing the connector lifecycle.
            external_history: Optional external history to use instead of the
                internal conversation history.

        Returns:
            The result of running the query.
        """
        result = ""
        initialized_here = False

        try:
            # Initialize if needed
            if manage_connector and (not self._initialized or not self._agent):
                logger.info("Initializing agent before running query")
                await self.initialize()
                initialized_here = True
            elif not self._initialized and self.auto_initialize:
                logger.info("Auto-initializing agent before running query")
                await self.initialize()
                initialized_here = True

            # Check if initialization succeeded
            if not self._agent:
                raise RuntimeError("MCP agent failed to initialize")

            # Add the user query to conversation history if memory is enabled
            if self.memory_enabled:
                self.add_to_history(HumanMessage(content=query))

            # Use the provided history or the internal history
            history_to_use = (
                external_history if external_history is not None else self._conversation_history
            )

            # Convert messages to format expected by LangChain
            langchain_history = []
            for msg in history_to_use:
                if isinstance(msg, HumanMessage):
                    langchain_history.append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    langchain_history.append({"type": "ai", "content": msg.content})
                elif isinstance(msg, SystemMessage) and msg != self._system_message:
                    # Include system messages other than the main one
                    # which is already included in the agent's prompt
                    langchain_history.append({"type": "system", "content": msg.content})
                # Other message types can be handled here if needed

            # Run the query with the specified max_steps or default
            logger.info(f"Running query with max_steps={max_steps or self.max_steps}")
            result = await self._agent.run(
                query=query,
                max_steps=max_steps,
                chat_history=langchain_history,
            )

            # Add the response to conversation history if memory is enabled
            if self.memory_enabled:
                self.add_to_history(AIMessage(content=result))

            return result

        except Exception as e:
            logger.error(f"Error running query: {e}")
            # If we initialized in this method and there was an error,
            # make sure to clean up
            if initialized_here and manage_connector:
                logger.info("Cleaning up resources after initialization error")
                await self.close()
            raise

        finally:
            # Clean up resources if we're managing the connector and
            # we're not using a client that manages sessions
            if manage_connector and not self.client and not initialized_here:
                logger.info("Closing agent after query completion")
                await self.close()

    async def close(self) -> None:
        """Close the MCP connection with improved error handling."""
        try:
            if self._agent:
                # Clean up the agent first
                logger.debug("Cleaning up agent")
                self._agent = None

            # If using client with session, close the session through client
            if self.client and self._sessions:
                logger.debug("Closing session through client")
                await self.client.close_all_sessions()
                self._session = None
            # If using direct connector, disconnect
            elif self.connectors:
                for connector in self.connectors:
                    logger.debug("Disconnecting connector")
                    await connector.disconnect()

            self._initialized = False
            logger.info("Agent closed successfully")

        except Exception as e:
            logger.error(f"Error during agent closure: {e}")
            # Still try to clean up references even if there was an error
            self._agent = None
            self._session = None
            self._initialized = False
