import asyncio
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_perplexity import ChatPerplexity
from mcp_use import MCPAgent, MCPClient

async def run_perplexity_example():
    """
    Run an example using Perplexity via MCP and LangChain.
    
    This example loads the Perplexity MCP configuration from 'perplexity_mcp.json',
    instantiates a ChatPerplexity LLM using LangChain (with model "sonar-pro" and a specified API key),
    and creates an MCP agent that uses the LangChain prompt template to query Perplexity.
    """
    # Load environment variables from .env (ensure PPLX_API_KEY is set)
    load_dotenv()
    
    # Create an MCPClient using the configuration file for Perplexity
    config_path = os.path.join(os.path.dirname(__file__), "perplexity_mcp.json")
    client = MCPClient.from_config_file(config_path)
    
    # Initialize the Perplexity LLM using LangChain integration.
    llm = ChatPerplexity(
        model="sonar-pro",
        temperature=0.2,
        pplx_api_key=os.getenv("PPLX_API_KEY"),
    )
    
    # (Optional) Create a chat prompt template with a specific instruction.
    prompt_template = ChatPromptTemplate.from_template(
        "Provide a comprehensive overview of the latest updates on the Perplexity Model Context Protocol."
        "Include key findings, product updates, and experimental breakthroughs."
    )
    
    # Create an MCPAgent using the client and LLM with a set maximum number of steps.
    agent = MCPAgent(llm=llm, client=client, max_steps=30)
    
    try:
        # Run the agent using the formatted prompt from the template.
        result = await agent.run(prompt_template.format(), max_steps=30)
        print(f"\nResult: {result}")
    finally:
        # Ensure all sessions are properly closed to free up resources.
        if client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(run_perplexity_example())
