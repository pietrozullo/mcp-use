"""
HTTP Example for mcp_use.

This example demonstrates how to use the mcp_use library with MCPClient
to connect to an MCP server running on a specific HTTP port.

Before running this example, you need to start the Playwright MCP server
in another terminal with:

    npx @playwright/mcp@latest --port 8931

This will start the server on port 8931. Resulting in the config you find below.
Of course you can run this with any server you want at any URL.

Special thanks to https://github.com/microsoft/playwright-mcp for the server.

"""

import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from mcp_use import MCPAgent, MCPClient


async def main():
    """Run the example using a configuration file."""
    # Load environment variables
    load_dotenv()

    config = {"mcpServers": {"http": {"url": "http://localhost:8931/sse"}}}

    # Create MCPClient from config file
    client = MCPClient.from_dict(config)

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query
    result = await agent.run(
        "Find the best restaurant in San Francisco USING GOOGLE SEARCH",
        max_steps=30,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    # Run the appropriate example
    asyncio.run(main())
