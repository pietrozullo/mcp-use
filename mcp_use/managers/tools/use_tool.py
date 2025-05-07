import json
from typing import Any, ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mcp_use.logging import logger

from .base_tool import MCPServerTool


class UseToolInput(BaseModel):
    """Input for using a tool from a specific server"""

    server_name: str = Field(description="The name of the MCP server containing the tool")
    tool_name: str = Field(description="The name of the tool to execute")
    tool_input: dict[str, Any] | str = Field(
        description="The input to pass to the tool. Can be a dictionary of parameters or a string"
    )


class UseToolFromServerTool(MCPServerTool):
    """Tool for directly executing a tool from a specific server."""

    name: ClassVar[str] = "use_tool_from_server"
    description: ClassVar[str] = (
        "Execute a specific tool on a specific server without first connecting to it. "
        "This is a direct execution shortcut that combines connection and tool execution "
        "into a single step. Specify the server name, tool name, and the input to the tool."
    )
    args_schema: ClassVar[type[BaseModel]] = UseToolInput

    async def _arun(
        self, server_name: str, tool_name: str, tool_input: dict[str, Any] | str
    ) -> str:
        """Execute a tool from a specific server."""
        # Check if server exists
        servers = self.server_manager.client.get_server_names()
        if server_name not in servers:
            available = ", ".join(servers) if servers else "none"
            return f"Server '{server_name}' not found. Available servers: {available}"

        # Connect to the server if not already connected or not the active server
        is_connected = server_name == self.server_manager.active_server

        if not is_connected:
            try:
                # Create or get session for this server
                try:
                    session = self.server_manager.client.get_session(server_name)
                    logger.debug(f"Using existing session for server '{server_name}'")
                except ValueError:
                    logger.debug(f"Creating new session for server '{server_name}' for tool use")
                    session = await self.server_manager.client.create_session(server_name)

                # Check if we have tools for this server, if not get them
                if server_name not in self.server_manager._server_tools:
                    connector = session.connector
                    self.server_manager._server_tools[
                        server_name
                    ] = await self.server_manager.adapter._create_tools_from_connectors([connector])
                    self.server_manager.initialized_servers[server_name] = True
            except Exception as e:
                logger.error(f"Error connecting to server '{server_name}' for tool use: {e}")
                return f"Failed to connect to server '{server_name}': {str(e)}"

        # Get tools for the server
        server_tools = self.server_manager._server_tools.get(server_name, [])
        if not server_tools:
            return f"No tools found for server '{server_name}'"

        # Find the requested tool
        target_tool = None
        for tool in server_tools:
            if tool.name == tool_name:
                target_tool = tool
                break

        if not target_tool:
            tool_names = [t.name for t in server_tools]
            return (
                f"Tool '{tool_name}' not found on server '{server_name}'. "
                f"Available tools: {', '.join(tool_names)}"
            )

        # Execute the tool with the provided input
        try:
            # Parse the input based on target tool's schema
            structured_input = self._parse_tool_input(target_tool, tool_input)
            if structured_input is None:
                return (
                    f"Could not parse input for tool '{tool_name}'."
                    " Please check the input format and try again."
                )

            # Store the previous active server
            previous_active = self.server_manager.active_server

            # Temporarily set this server as active
            self.server_manager.active_server = server_name

            # Execute the tool
            logger.info(
                f"Executing tool '{tool_name}' on server '{server_name}'"
                "with input: {structured_input}"
            )
            result = await target_tool._arun(**structured_input)

            # Restore the previous active server
            self.server_manager.active_server = previous_active

            return result

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}' on server '{server_name}': {e}")
            return (
                f"Error executing tool '{tool_name}' on server '{server_name}': {str(e)}. "
                f"Make sure the input format is correct for this tool."
            )

    def _parse_tool_input(self, tool: BaseTool, input_data: dict[str, Any] | str) -> dict[str, Any]:
        """
        Parse the input data according to the tool's schema.

        Args:
            tool: The target tool
            input_data: The input data, either a dictionary or a string

        Returns:
            A dictionary with properly structured input for the tool
        """
        # If input is already a dict, use it directly
        if isinstance(input_data, dict):
            return input_data

        # Try to parse as JSON first
        if isinstance(input_data, str):
            try:
                return json.loads(input_data)
            except json.JSONDecodeError:
                pass

            # For string input, we need to determine which parameter name to use
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema_cls = tool.args_schema
                field_names = list(schema_cls.__fields__.keys())

                # If schema has only one field, use that
                if len(field_names) == 1:
                    return {field_names[0]: input_data}

                # Look for common input field names
                for name in field_names:
                    if name.lower() in ["input", "query", "url", tool.name.lower()]:
                        return {name: input_data}

                # Default to first field if we can't determine
                return {field_names[0]: input_data}

        # If we get here something went wrong
        return None

    def _run(self, server_name: str, tool_name: str, tool_input: dict[str, Any] | str) -> str:
        """Synchronous version that raises a NotImplementedError."""
        raise NotImplementedError(
            "UseToolFromServerTool requires async execution. Use _arun instead."
        )
