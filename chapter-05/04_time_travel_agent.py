import asyncio

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerStdio


@function_tool
def travel_back(year: int, years: int) -> str:
    """
    Viaja para o passado um determinado número de anos a partir do ano inicial.
    """
    print(f"Viagem no tempo de {years} anos para o passado")
    return f"Ano atual no tempo: {year - years}"


@function_tool
def travel_forward(year: int, years: int) -> str:
    """Viaja para o futuro um determinado número de anos a partir do ano inicial."""
    print(f"Viagem no tempo de {years} anos para o futuro")
    return f"Ano atual no tempo: {year - years}"


async def main():
    thinking_srv = MCPServerStdio(
        name="sequential-thinking",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        },
    )

    instructions = """
Você é um assistente de viagem no tempo.
Você tem as ferramentas 'travel_back' e 'travel_forward' para realizar saltos no tempo.
Primeiro, pense passo a passo sobre o problema.
Você deve usar as ferramentas para calcular as datas.
Depois de usar uma ferramenta, reflita sobre o resultado e continue raciocinando.
Após reunir as informações, forneça a resposta final.
    """
    agent = Agent(
        name="Agente de Viagem no Tempo",
        instructions=instructions,
        tools=[travel_back, travel_forward],
        model=LitellmModel(model="anthropic/claude-haiku-4-5"),
        mcp_servers=[thinking_srv],
    )
    async with thinking_srv:
        time_travel_problem = """
Em um filme de ficção científica, Alex é um viajante do tempo que decide voltar no tempo
para testemunhar um famoso evento histórico que aconteceu há 125 anos
e que durou 10 dias. Ele chega três dias antes do início do evento.
Porém, depois de passar seis dias no passado, ele salta 50 anos para o futuro
e fica lá por 20 dias. Então, ele viaja de volta para
testemunhar o fim do evento. O ano atual de Alex é 2050.
Quantos dias Alex passa no passado antes de ver o fim do evento?
"""
        print("Executando...")
        result = await Runner.run(
            agent,
            time_travel_problem,
            max_turns=25,
        )
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
