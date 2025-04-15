"""
Basic usage example for mcp_use.

This example demonstrates how to use the mcp_use library with MCPClient
to connect any LLM to MCP tools through a unified interface.

Special Thanks to https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem
for the server.
"""

import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from mcp_use import MCPAgent, MCPClient

config = {
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "THE_PATH_TO_YOUR_DIRECTORY",
            ],
        }
    }
}


async def main():
    """Run the example using a configuration file."""
    # Load environment variables
    load_dotenv()

    # Create MCPClient from config file
    client = MCPClient.from_dict(config)
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o")
    # llm = init_chat_model(model="llama-3.1-8b-instant", model_provider="groq")
    # llm = ChatAnthropic(model="claude-3-")
    # llm = ChatGroq(model="llama3-8b-8192")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query
    result = await agent.run(
        "Hello can you give me a list of files and directories in the current directory",
        max_steps=30,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    # Run the appropriate example
    asyncio.run(main())
