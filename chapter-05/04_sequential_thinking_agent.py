import asyncio

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerStdio


async def main():
    # Instancie os servidores primeiro…
    thinking_srv = MCPServerStdio(
        name="sequential-thinking",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        },
    )

    instructions = """
Você é um assistente de planejamento prestativo.
    """
    agent = Agent(
        name="Assistente",
        model=LitellmModel(model="anthropic/claude-haiku-4-5"),
        instructions=instructions,
        mcp_servers=[thinking_srv],
    )

    async with thinking_srv:
        tools = await thinking_srv.list_tools()
        print("Ferramentas disponíveis:", tools)
        goal = """
Descubra e liste as ferramentas e funções que você tem disponíveis.
"""
        print("Executando...", goal)
        result = await Runner.run(agent, goal)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
