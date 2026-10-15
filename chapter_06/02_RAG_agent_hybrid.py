import uuid
from pathlib import Path

import chromadb
import tiktoken
from agents import Agent, Runner, function_tool, set_tracing_disabled  # OpenAI Agents SDK
from agents.exceptions import MaxTurnsExceeded
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Desativa o tracing (ele envia dados para a OpenAI e exige OPENAI_API_KEY)
set_tracing_disabled(True)

# ------------------------------------------------------------------
# 1. Carregar + dividir o roteiro em chunks
# ------------------------------------------------------------------
script_text = Path("chapter_06/sample_documents/back_to_the_future.txt").read_text(
    encoding="utf-8"
)


def simple_chunk(text, max_tokens=200):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words, chunk, chunks = text.split(), [], []
    for w in words:
        if len(tokenizer.encode(" ".join(chunk + [w]))) > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = [w]
        else:
            chunk.append(w)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


docs = simple_chunk(script_text, max_tokens=200)

# ------------------------------------------------------------------
# 2. Criar (ou conectar a) uma coleção do Chroma com embeddings da OpenAI
# ------------------------------------------------------------------
client = chromadb.PersistentClient(
    path="./chapter_06/chroma_script_store"  # em disco para reutilizarmos depois
)
collection_name = "bttf_script"

# Tenta obter a coleção existente primeiro
try:
    collection = client.get_collection(collection_name)
except Exception:
    # A coleção não existe, cria sem função de embedding
    collection = client.create_collection(name=collection_name)

# Popula apenas uma vez (pula se já estiver populada)
if collection.count() == 0:
    collection.add(ids=[str(uuid.uuid4()) for _ in docs], documents=docs)


# ------------------------------------------------------------------
# 3. Definir as ferramentas de busca para a abordagem híbrida
# ------------------------------------------------------------------
@function_tool
def search_script_with_vector_similarity(query: str, top_k: int = 3) -> str:
    """Busca no roteiro usando similaridade vetorial (busca semântica)."""
    res = collection.query(query_texts=[query], n_results=top_k)
    if res and "documents" in res and res["documents"] and res["documents"][0]:
        # Formata os resultados com separação clara e marcadores de referência
        formatted_results = []
        for i, doc in enumerate(res["documents"][0]):
            formatted_results.append(f"[Referência {i + 1}]\n{doc}")
        return "\n\n".join(formatted_results)
    return "Nenhum documento relevante encontrado."


@function_tool
def search_script_with_keyword(query: str, top_k: int = 3) -> str:
    """Busca no roteiro usando correspondência básica de palavras-chave."""
    # Converte a consulta para minúsculas para busca sem diferenciar maiúsculas
    query_lower = query.lower()

    # Divide a consulta em palavras individuais (remove palavras curtas)
    query_words = [
        word.strip() for word in query_lower.split() if len(word.strip()) > 2
    ]

    if not query_words:
        return "Nenhum documento relevante encontrado."

    # Pontua cada documento com base nas correspondências de palavras-chave
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        score = 0

        # Conta correspondências exatas da frase (maior peso)
        if query_lower in doc_lower:
            score += 10

        # Conta correspondências de palavras individuais
        for word in query_words:
            # Conta as ocorrências de cada palavra
            word_count = doc_lower.count(word)
            score += word_count * 2

        # Aumenta a pontuação de documentos com várias palavras da consulta
        words_found = sum(1 for word in query_words if word in doc_lower)
        if len(query_words) > 1 and words_found > 1:
            score += words_found

        if score > 0:
            scored_docs.append((score, doc))

    # Ordena por pontuação (decrescente) e retorna os top_k
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Formata os resultados com marcadores de referência claros
    if scored_docs:
        formatted_results = []
        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            formatted_results.append(f"[Referência {i + 1}]\n{doc}")
        return "\n\n".join(formatted_results)
    return "Nenhum documento relevante encontrado."


# ------------------------------------------------------------------
# 4. Construir o agente com capacidades de busca híbrida
# ------------------------------------------------------------------
agent = Agent(
    name="Agente do Roteiro",
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    instructions="""
Você responde perguntas sobre o filme *De Volta para o Futuro*.
Você tem acesso a duas ferramentas de busca:
1. `search_script_with_vector_similarity` - Use para
   buscas semânticas/conceituais
2. `search_script_with_keyword` - Use para buscas de
   palavras/frases exatas
Use as duas ferramentas quando apropriado para obter
resultados abrangentes.

O roteiro está em inglês: ao buscar, use termos em inglês.
Responda sempre em português.

Faça no máximo 4 buscas no total. Se os trechos não
trouxerem a resposta, diga que ela não foi encontrada no
roteiro em vez de continuar buscando.

IMPORTANTE: Você deve sempre fornecer referências exatas
do roteiro.
- Cite trechos relevantes diretamente dos resultados
  da busca
- Use os marcadores [Referência X] para citar trechos
  específicos
- Não invente informações que não estejam no roteiro
- Se o roteiro não contiver a informação solicitada,
  diga isso claramente
- Sempre fundamente suas respostas no texto real do
  roteiro fornecido pelas ferramentas de busca

Formate suas respostas assim:
Com base no roteiro:
[Citação da Referência 1]: "texto exato do roteiro"
[Citação da Referência 2]: "texto exato do roteiro"

Sua interpretação: [sua análise com base nas
citações]
""",
    tools=[search_script_with_vector_similarity, search_script_with_keyword],
)


# ------------------------------------------------------------------
# 5. Testar a busca híbrida com várias consultas
# ------------------------------------------------------------------
queries = [
    "Onde o Doc diz para o Marty encontrá-lo, e a que horas?",
    "O que acontece à 1:15 da manhã?",
    "um minuto e quinze segundos",
    "Quem é o Doc Brown?",
    "máquina do tempo",
    "DeLorean",
]

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"CONSULTA: {query}")
    print("=" * 60)
    try:
        result = Runner.run_sync(agent, query)
    except MaxTurnsExceeded:
        print("\n--- RESPOSTA ---\n O agente excedeu o limite de turnos.")
        continue
    print("\n--- RESPOSTA ---\n", result.final_output)
