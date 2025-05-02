"""
Bright Data Example for mcp_use with Google Gemini.
Please make sure to install Bright Data MCP Server
https://www.npmjs.com/package/@brightdata/mcp

"""
import asyncio
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from mcp_use import MCPAgent, MCPClient

async def main():
    # Load .env if needed
    load_dotenv()

    # Configure MCP Server
    config = {
        "mcpServers": {
            "Bright Data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.environ["BRIGHT_DATA_KEY"]
                }
            }
        }
    }
    client = MCPClient.from_dict(config)

    # Set up the Gemini LLM (replace model name if needed)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # Create MCP Agent
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run query
    result = await agent.run(
        "Find the best restaurant in San Francisco USING GOOGLE SEARCH",
        max_steps=30,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    asyncio.run(main())
