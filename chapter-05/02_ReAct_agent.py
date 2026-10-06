import asyncio

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel


# Define duas ferramentas para cálculos de tempo
@function_tool
def travel_back(year: int, years: int) -> int:
    """Viaja para o passado um determinado número de anos a partir do ano inicial."""
    return year - years


@function_tool
def travel_forward(year: int, years: int) -> int:
    """Viaja para o futuro um determinado número de anos a partir do ano inicial."""
    return year + years


# Cria um agente equipado com as ferramentas
react_agent = Agent(
    name="TimeTravelerReAct",
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    instructions=(
        "Você é um assistente de viagem no tempo. Você tem as ferramentas 'travel_back' e 'travel_forward' para realizar saltos no tempo. "
        "Primeiro, pense passo a passo sobre o problema. Se necessário, use as ferramentas para calcular as datas. "
        "Depois de usar uma ferramenta, reflita sobre o resultado e continue raciocinando. "
        "Depois de reunir as informações, forneça a resposta final."
    ),
    tools=[travel_back, travel_forward],
)

# Um problema de viagem no tempo que exige o uso das ferramentas
problem = (
    "Estou no ano de 2050. "
    "Viajo 25 anos para o passado, depois viajo 10 anos para o futuro "
    "e, por fim, volto mais 5 anos para o passado. Em que ano estou agora?"
)

result = asyncio.run(Runner.run(react_agent, input=problem))
print(result.final_output)
