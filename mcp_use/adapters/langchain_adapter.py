"""
LangChain adapter for MCP tools.

This module provides utilities to convert MCP tools to LangChain tools.
"""

from typing import Any, NoReturn

from jsonschema_pydantic import jsonschema_to_pydantic
from langchain_core.tools import BaseTool, ToolException
from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent
from pydantic import BaseModel

from ..connectors.base import BaseConnector
from ..logging import logger
from .base import BaseAdapter


class LangChainAdapter(BaseAdapter):
    """Adapter for converting MCP tools to LangChain tools."""

    def __init__(self, disallowed_tools: list[str] | None = None) -> None:
        """Initialize a new LangChain adapter.

        Args:
            disallowed_tools: list of tool names that should not be available.
        """
        super().__init__(disallowed_tools)
        self._connector_tool_map: dict[BaseConnector, list[BaseTool]] = {}

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

    def _parse_mcp_tool_result(self, tool_result: CallToolResult) -> str:
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

    def _convert_tool(self, mcp_tool: dict[str, Any], connector: BaseConnector) -> BaseTool:
        """Convert an MCP tool to LangChain's tool format.

        Args:
            mcp_tool: The MCP tool to convert.
            connector: The connector that provides this tool.

        Returns:
            A LangChain BaseTool.
        """
        # Skip disallowed tools
        if mcp_tool.name in self.disallowed_tools:
            return None

        # This is a dynamic class creation, we need to work with the self reference
        adapter_self = self

        class McpToLangChainAdapter(BaseTool):
            name: str = mcp_tool.name or "NO NAME"
            description: str = mcp_tool.description or ""
            # Convert JSON schema to Pydantic model for argument validation
            args_schema: type[BaseModel] = jsonschema_to_pydantic(
                adapter_self.fix_schema(mcp_tool.inputSchema)  # Apply schema conversion
            )
            tool_connector: BaseConnector = connector  # Renamed variable to avoid name conflict
            handle_tool_error: bool = True

            def __repr__(self) -> str:
                return f"MCP tool: {self.name}: {self.description}"

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
                logger.debug(f'MCP tool: "{self.name}" received input: {kwargs}')

                try:
                    tool_result: CallToolResult = await self.tool_connector.call_tool(
                        self.name, kwargs
                    )
                    try:
                        # Use the helper function to parse the result
                        return adapter_self._parse_mcp_tool_result(tool_result)
                    except Exception as e:
                        # Log the exception for debugging
                        logger.error(f"Error parsing tool result: {e}")
                        return f"Error parsing result: {e!s}; Raw content: {tool_result.content!r}"

                except Exception as e:
                    if self.handle_tool_error:
                        return f"Error executing MCP tool: {str(e)}"
                    raise

        return McpToLangChainAdapter()
