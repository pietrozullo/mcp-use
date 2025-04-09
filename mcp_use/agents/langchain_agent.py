"""
LangChain agent implementation for MCP tools with customizable system message.

This module provides a LangChain agent implementation that can use MCP tools
through a unified interface, with support for customizable system messages.
"""

from typing import Any, NoReturn

from jsonschema_pydantic import jsonschema_to_pydantic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.tools import BaseTool, ToolException
from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent
from pydantic import BaseModel

from ..connectors.base import BaseConnector
from ..logging import logger


def _parse_mcp_tool_result(tool_result: CallToolResult) -> str:
    """Parse the content of a CallToolResult into a string.

    Args:
        tool_result: The result object from calling an MCP tool.

    Returns:
        A string representation of the tool result content.

    Raises:
        ToolException: If the tool execution failed, returned no content,
                       or contained unexpected content types.
    """
    if tool_result.isError:
        raise ToolException(f"Tool execution failed: {tool_result.content}")

    if not tool_result.content:
        raise ToolException("Tool execution returned no content")

    decoded_result = ""
    for item in tool_result.content:
        match item.type:
            case "text":
                item: TextContent
                decoded_result += item.text
            case "image":
                item: ImageContent
                decoded_result += item.data  # Assuming data is string-like or base64
            case "resource":
                resource: EmbeddedResource = item.resource
                if hasattr(resource, "text"):
                    decoded_result += resource.text
                elif hasattr(resource, "blob"):
                    # Assuming blob needs decoding or specific handling; adjust as needed
                    decoded_result += (
                        resource.blob.decode()
                        if isinstance(resource.blob, bytes)
                        else str(resource.blob)
                    )
                else:
                    raise ToolException(f"Unexpected resource type: {resource.type}")
            case _:
                raise ToolException(f"Unexpected content type: {item.type}")

    return decoded_result


class LangChainAgent:
    """LangChain agent that can use MCP tools.

    This agent uses LangChain's agent framework to interact with MCP tools
    through a unified interface.
    """

    # Default system message if none is provided
    DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI assistant that can use tools to help users."

    def __init__(
        self,
        connectors: list[BaseConnector],
        llm: BaseLanguageModel,
        max_steps: int = 5,
        system_message: str | None = None,
        disallowed_tools: list[str] | None = None,
    ) -> None:
        """Initialize a new LangChain agent.

        Args:
            connector: The MCP connector to use.
            llm: The LangChain LLM to use.
            max_steps: The maximum number of steps to take.
            system_message: Optional custom system message to use.
            disallowed_tools: List of tool names that should not be available to the agent.
        """
        self.connectors = connectors
        self.llm = llm
        self.max_steps = max_steps
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        self.disallowed_tools = disallowed_tools or []
        self.tools: list[BaseTool] = []
        self.agent: AgentExecutor | None = None

    def set_system_message(self, message: str) -> None:
        """Set a new system message and recreate the agent.

        Args:
            message: The new system message.
        """
        self.system_message = message

        # Recreate the agent with the new system message if it exists
        if self.agent and self.tools:
            self.agent = self._create_agent()
            logger.info("Agent recreated with new system message")

    async def initialize(self) -> None:
        """Initialize the agent and its tools."""
        self.tools = await self._create_langchain_tools()
        self.agent = self._create_agent()

    def fix_schema(self, schema: dict) -> dict:
        """Convert JSON Schema 'type': ['string', 'null'] to 'anyOf' format.

        Args:
            schema: The JSON schema to fix.

        Returns:
            The fixed JSON schema.
        """
        if isinstance(schema, dict):
            if "type" in schema and isinstance(schema["type"], list):
                schema["anyOf"] = [{"type": t} for t in schema["type"]]
                del schema["type"]  # Remove 'type' and standardize to 'anyOf'
            for key, value in schema.items():
                schema[key] = self.fix_schema(value)  # Apply recursively
        return schema

    async def _create_langchain_tools(self) -> list[BaseTool]:
        """Create LangChain tools from MCP tools.

        Returns:
            A list of LangChain tools that wrap MCP tools.
        """
        tools = []
        for connector in self.connectors:
            local_connector = connector  # Capture for closure
            for tool in connector.tools:
                # Skip disallowed tools
                if tool.name in self.disallowed_tools:
                    continue

                class McpToLangChainAdapter(BaseTool):
                    name: str = tool.name or "NO NAME"
                    description: str = tool.description or ""
                    # Convert JSON schema to Pydantic model for argument validation
                    args_schema: type[BaseModel] = jsonschema_to_pydantic(
                        self.fix_schema(tool.inputSchema)  # Apply schema conversion
                    )
                    connector: BaseConnector = local_connector
                    handle_tool_error: bool = True

                    def _run(self, **kwargs: Any) -> NoReturn:
                        """Synchronous run method that always raises an error.

                        Raises:
                            NotImplementedError: Always raises this error because MCP tools
                                only support async operations.
                        """
                        raise NotImplementedError("MCP tools only support async operations")

                    async def _arun(self, **kwargs: Any) -> Any:
                        """Asynchronously execute the tool with given arguments.

                        Args:
                            kwargs: The arguments to pass to the tool.

                        Returns:
                            The result of the tool execution.

                        Raises:
                            ToolException: If tool execution fails.
                        """
                        logger.info(f'MCP tool: "{self.name}" received input: {kwargs}')

                        try:
                            tool_result: CallToolResult = await self.connector.call_tool(
                                self.name, kwargs
                            )
                            try:
                                # Use the helper function to parse the result
                                return _parse_mcp_tool_result(tool_result)
                            except Exception as e:
                                # Log the exception for debugging
                                logger.error(f"Error parsing tool result: {e}")
                                # Shortened line:
                                return (
                                    f"Error parsing result: {e!s};"
                                    f" Raw content: {tool_result.content!r}"
                                )

                        except Exception as e:
                            if self.handle_tool_error:
                                return f"Error executing MCP tool: {str(e)}"
                            raise

                tools.append(McpToLangChainAdapter())

        # Log available tools for debugging
        logger.info(f"Available tools: {[tool.name for tool in tools]}")
        return tools

    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with the configured system message.

        Returns:
            An initialized AgentExecutor.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.system_message,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        return AgentExecutor(
            agent=agent, tools=self.tools, max_iterations=self.max_steps, verbose=False
        )

    async def run(
        self,
        query: str,
        max_steps: int | None = None,
        chat_history: list | None = None,
    ) -> str:
        """Run the agent on a query.

        Args:
            query: The query to run.
            max_steps: Optional maximum number of steps to take.
            chat_history: Optional chat history.

        Returns:
            The result of running the query.

        Raises:
            RuntimeError: If the MCP client is not initialized.
        """
        if not self.agent:
            raise RuntimeError("MCP client is not initialized")

        if max_steps is not None:
            self.agent.max_iterations = max_steps

        # Initialize empty chat history if none provided
        if chat_history is None:
            chat_history = []

        # Invoke with all required variables
        result = await self.agent.ainvoke({"input": query, "chat_history": chat_history})

        return result["output"]
