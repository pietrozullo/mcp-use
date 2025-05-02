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
            "mcpServers":
            {
                "airbnb": {
                    "command": "npx",
                    "args": ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
                },
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

    # Run a query that uses tools from multiple servers
    result = await agent.run(
        "Search for a nice place to stay in Barcelona on Airbnb, "
        "then use Google Search to find nearby restaurants and attractions."
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
