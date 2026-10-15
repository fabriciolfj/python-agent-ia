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
    "A Geleira Solaris absorve calor, liberando fogo-gelo luminescente ao entardecer.",
    "As árvores Quantum Bark emitem fótons emaranhados que guiam migrações noturnas de insetos.",
    "O tecido Aether Silk se autorrepara, costurando microrrasgos em catorze milissegundos.",
    "Os recifes Chrono Coral rebobinam as correntes de água três minutos a cada solstício.",
]

# Benchmarks para o conhecimento especial
_benchmarks = [
    {"q": "O motor Nebula Forge gira o quê?", "a": "antimatéria", "wrong": "fusão"},
    {
        "q": "A Geleira Solaris libera o quê luminescente?",
        "a": "fogo-gelo",
        "wrong": "lava",
    },
    {"q": "As árvores Quantum Bark emitem o quê?", "a": "fótons", "wrong": "esporos"},
    {"q": "O tecido Aether Silk repara o quê?", "a": "microrrasgos", "wrong": "fios"},
    {"q": "Os recifes Chrono Coral rebobinam o quê?", "a": "correntes", "wrong": "marés"},
]


@function_tool
def search_knowledge_by_keyword(query: str) -> dict:
    """
    Pesquisa fatos relevantes na base de conhecimento.
    param query: A palavra-chave única a ser pesquisada.
    """
    matches = [doc for doc in _special_knowledge_db if query.lower() in doc.lower()]
    print(f"Encontradas {len(matches)} correspondências para '{query}'")
    return {"status": "ok", "context": "\n".join(matches)}


agent = Agent(
    name="Agente RAG",
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    instructions="""
Você é um agente de conhecimento aumentado por recuperação (RAG).
Divida a consulta do usuário em partes menores, se necessário,
para buscar o contexto relevante para a consulta do usuário.
""",
    tools=[search_knowledge_by_keyword]
)


class EvaluationOutput(BaseModel):
    """Modelo de saída do agente de avaliação."""

    is_correct: bool
    feedback: str


evaluation_agent = Agent(
    name="Agente de Avaliação",
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    instructions="""
Você é um agente de avaliação.
Sua tarefa é avaliar as respostas fornecidas pelo Agente RAG.
Você vai comparar a resposta com o termo-chave
da resposta esperada para validar se está correta.
""",
    output_type=EvaluationOutput,
)

for benchmark in _benchmarks:
    question = benchmark["q"]
    answer = benchmark["a"]
    wrong_answer = benchmark["wrong"]
    result = asyncio.run(Runner.run(agent, input=question)).final_output.strip()
    evaluation_input = dict(
        question=question,
        answer=result,
        expected_key_term=answer,
        wrong_answer=wrong_answer,
    )
    print("" + "=" * 40)
    evaluation = asyncio.run(
        Runner.run(evaluation_agent, input=str(evaluation_input))
    ).final_output
    print(f"Avaliação: {evaluation.is_correct}")
    print(f"Feedback: {evaluation.feedback}")
    print(f"Pergunta: {question}")
    print(f"Resposta: {result}")
    if evaluation.is_correct:
        print(f"Correta -> {answer}")
    else:
        print(f"Incorreta -> Esperado: {answer}")
        if wrong_answer:
            print(f"A resposta errada era: {wrong_answer}")
