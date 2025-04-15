from langchain.schema import SystemMessage
from langchain_core.tools import BaseTool


def generate_tool_descriptions(
    tools: list[BaseTool], disallowed_tools: list[str] | None = None
) -> list[str]:
    """
    Generates a list of formatted tool descriptions, excluding disallowed tools.

    Args:
        tools: The list of available BaseTool objects.
        disallowed_tools: A list of tool names to exclude.

    Returns:
        A list of strings, each describing a tool in the format "- tool_name: description".
    """
    disallowed_set = set(disallowed_tools or [])
    tool_descriptions_list = []
    for tool in tools:
        if tool.name in disallowed_set:
            continue
        # Escape curly braces for formatting
        escaped_desc = tool.description.replace("{", "{{").replace("}", "}}")
        description = f"- {tool.name}: {escaped_desc}"
        tool_descriptions_list.append(description)
    return tool_descriptions_list


def build_system_prompt_content(
    template: str, tool_description_lines: list[str], additional_instructions: str | None = None
) -> str:
    """
    Builds the final system prompt string using a template, tool descriptions,
    and optional additional instructions.

    Args:
        template: The system prompt template string (must contain '{tool_descriptions}').
        tool_description_lines: A list of formatted tool description strings.
        additional_instructions: Optional extra instructions to append.

    Returns:
        The fully formatted system prompt content string.
    """
    tool_descriptions_block = "\n".join(tool_description_lines)
    # Add a check for missing placeholder to prevent errors
    if "{tool_descriptions}" not in template:
        # Handle this case: maybe append descriptions at the end or raise an error
        # For now, let's append if placeholder is missing
        print("Warning: '{tool_descriptions}' placeholder not found in template.")
        system_prompt_content = template + "\n\nAvailable tools:\n" + tool_descriptions_block
    else:
        system_prompt_content = template.format(tool_descriptions=tool_descriptions_block)

    if additional_instructions:
        system_prompt_content += f"\n\n{additional_instructions}"

    return system_prompt_content


def create_system_message(
    tools: list[BaseTool],
    system_prompt_template: str,
    server_manager_template: str,
    use_server_manager: bool,
    disallowed_tools: list[str] | None = None,
    user_provided_prompt: str | None = None,
    additional_instructions: str | None = None,
) -> SystemMessage:
    """
    Creates the final SystemMessage object for the agent.

    Handles selecting the correct template, generating tool descriptions,
    and incorporating user overrides and additional instructions.

    Args:
        tools: List of available tools.
        system_prompt_template: The default system prompt template.
        server_manager_template: The template to use when server manager is active.
        use_server_manager: Flag indicating if server manager mode is enabled.
        disallowed_tools: List of tool names to exclude.
        user_provided_prompt: A complete system prompt provided by the user, overriding templates.
        additional_instructions: Extra instructions to append to the template-based prompt.

    Returns:
        A SystemMessage object containing the final prompt content.
    """
    # If a complete user prompt is given, use it directly
    if user_provided_prompt:
        return SystemMessage(content=user_provided_prompt)

    # Select the appropriate template
    template_to_use = server_manager_template if use_server_manager else system_prompt_template

    # Generate tool descriptions
    tool_description_lines = generate_tool_descriptions(tools, disallowed_tools)

    # Build the final prompt content
    final_prompt_content = build_system_prompt_content(
        template=template_to_use,
        tool_description_lines=tool_description_lines,
        additional_instructions=additional_instructions,
    )

    return SystemMessage(content=final_prompt_content)
