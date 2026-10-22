import asyncio

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerStreamableHttp, create_static_tool_filter
from dotenv import load_dotenv

load_dotenv()


async def main():
    # External public MCP server (no auth) over Streamable HTTP
    deepwiki_server = MCPServerStreamableHttp(
        name="DeepWiki",
        params={
            "url": "https://mcp.deepwiki.com/mcp",
            "timeout": 30,
        },
        client_session_timeout_seconds=60,
        # read_wiki_contents returns the whole wiki and blows the context window
        tool_filter=create_static_tool_filter(allowed_tool_names=["ask_wiki_question"]),
    )

    await deepwiki_server.connect()  # abre a conexão com o servidor MCP
    try:
        agent = Agent(
            name="Assistant",
            instructions="Use the DeepWiki tools to answer questions about GitHub repositories.",
            model=LitellmModel(model="anthropic/claude-haiku-4-5"),
            mcp_servers=[deepwiki_server],
        )

        tools = await deepwiki_server.list_tools()
        print("Available tools:", [tool.name for tool in tools])

        question = "What is the openai/openai-agents-python repository about? Answer briefly."
        print(f"Running: {question}")
        result = await Runner.run(agent, question)
        print(result.final_output)
    finally:
        await deepwiki_server.cleanup()  # fecha a conexão, mesmo se der erro


if __name__ == "__main__":
    asyncio.run(main())