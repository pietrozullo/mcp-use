"""
This example shows how to test the different functionalities of MCPs using the MCP server from
anthropic.
"""

import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from mcp_use import MCPAgent, MCPClient

everything_server = {
    "mcpServers": {
        "everything": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-everything"]}
    }
}


async def main():
    """Run the example using a configuration file."""
    load_dotenv()
    client = MCPClient(config=everything_server)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    result = await agent.run(
        """
        Hello, you are a tester can you please answer the follwing questions:
        - Which resources do you have access to?
        - Which prompts do you have access to?
        - Which tools do you have access to?
        """,
        max_steps=30,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    # Run the appropriate example
    asyncio.run(main())
