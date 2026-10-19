import asyncio

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from pydantic import BaseModel

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Base de conhecimento simples em memória
_special_knowledge_db = [
    "O motor Nebula Forge gira anéis de antimatéria para controlar a gravidade.",
    "O Solaris Glacier absorve calor, liberando fogo-gelo luminescente ao entardecer.",
    "As árvores Quantum Bark emitem fótons emaranhados que guiam migrações noturnas de insetos.",
    "O tecido Aether Silk se autotece, reparando microrrasgos em catorze milissegundos.",
    "Os recifes Chrono Coral rebobinam as correntes de água por três minutos a cada solstício.",
]

# benchmarks para o conhecimento especial
_benchmarks = [
    {"q": "O motor Nebula Forge gira o quê?", "a": "antimatéria", "wrong": "fusão"},
    {
        "q": "O Solaris Glacier libera o que luminescente?",
        "a": "fogo-gelo",
        "wrong": "lava",
    },
    {"q": "As árvores Quantum Bark emitem o quê?", "a": "fótons", "wrong": "esporos"},
    {"q": "O tecido Aether Silk repara o quê?", "a": "microrrasgos", "wrong": "fios"},
    {"q": "Os recifes Chrono Coral rebobinam o quê?", "a": "correntes", "wrong": "marés"},
]

_last_context = ""  # Variável global para armazenar o último contexto


@function_tool
def search_knowledge_by_keyword(query: str) -> dict:
    """
    Busca fatos relevantes na base de conhecimento.
    param query: A palavra-chave única a ser buscada.
    """
    global _last_context
    matches = [doc for doc in _special_knowledge_db if query.lower() in doc.lower()]
    print(f"Encontrados {len(matches)} resultados para '{query}'")
    _last_context = "\n".join(matches)
    return {"status": "ok", "context": _last_context}


agent = Agent(
    name="RAG Agent",
    instructions="""
Você é um agente de conhecimento aumentado por recuperação (RAG).
Divida a consulta do usuário em partes menores, se necessário,
para buscar o contexto relevante para a consulta do usuário.
Responda sempre em português.
""",
    tools=[search_knowledge_by_keyword],
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
)


@function_tool
def get_last_context() -> str:
    """
    Recupera o último contexto usado pelo agente de grounding.
    """
    global _last_context
    if not _last_context:
        return "Nenhum contexto disponível."
    return _last_context


class GroundedAnswer(BaseModel):
    """Modelo de saída do agente de grounding."""

    is_answer_grounded: bool
    feedback: str


grounding_agent = Agent(
    name="Grounding Agent",
    instructions="""
Você é um agente de grounding (verificação de fundamentação).
Sua tarefa é avaliar a correção das respostas
com base na pergunta fornecida, no contexto utilizado
e na resposta gerada.
Escreva o feedback em português.
""",
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    output_type=GroundedAnswer,
    tools=[get_last_context],
)


for benchmark in _benchmarks:
    question = benchmark["q"]
    answer = benchmark["a"]
    wrong_answer = benchmark["wrong"]
    result = asyncio.run(Runner.run(agent, input=question)).final_output.strip()
    grounding_input = dict(
        pergunta=question,
        resposta=result,
    )
    print("" + "=" * 40)
    grounding = asyncio.run(
        Runner.run(grounding_agent, input=str(grounding_input))
    ).final_output
    print(f"Fundamentada: {grounding.is_answer_grounded}")
    print(f"Feedback: {grounding.feedback}")
    print(f"Pergunta: {question}")
    print(f"Resposta: {result}")
