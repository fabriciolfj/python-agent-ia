import uuid
from pathlib import Path

import chromadb
import tiktoken
from agents import Agent, Runner, function_tool, set_tracing_disabled  # OpenAI Agents SDK
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Desativa o tracing (ele envia dados para a OpenAI e exige OPENAI_API_KEY)
set_tracing_disabled(True)

# ------------------------------------------------------------------
# 1. Carrega o roteiro e divide em pedaços (chunks)
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
# 2. Cria (ou conecta a) uma coleção do Chroma com a função de embedding padrão
# ------------------------------------------------------------------
client = chromadb.PersistentClient(
    path="./chapter_06/chroma_script_store"  # em disco, para reutilizar depois
)
collection_name = "bttf_script"

# Tenta obter a coleção existente primeiro
try:
    collection = client.get_collection(collection_name)
except Exception:
    # A coleção não existe, cria sem especificar função de embedding
    collection = client.create_collection(name=collection_name)

# Popula apenas uma vez (pula se já estiver populada)
if collection.count() == 0:
    collection.add(ids=[str(uuid.uuid4()) for _ in docs], documents=docs)


# ------------------------------------------------------------------
# 3. Define uma ferramenta de busca semântica que consulta o Chroma
# ------------------------------------------------------------------
@function_tool
def search_script(query: str, top_k: int = 3) -> str:
    res = collection.query(query_texts=[query], n_results=top_k)
    if res and "documents" in res and res["documents"] and res["documents"][0]:
        return "\n\n".join(res["documents"][0])  # junta os melhores trechos
    return "Nenhum documento relevante encontrado."


# ------------------------------------------------------------------
# 4. Monta o agente
# ------------------------------------------------------------------
agent = Agent(
    name="Agente do Roteiro",
    model=LitellmModel(model="anthropic/claude-haiku-4-5"),
    instructions=(
        "Você responde perguntas sobre o filme *De Volta para o Futuro*.\n"
        "Quando necessário, chame a ferramenta `search_script` para buscar trechos, "
        "depois cite ou parafraseie esses trechos na sua resposta.\n"
        "O roteiro está em inglês, então faça as buscas em inglês, "
        "mas responda sempre em português.\n"
        "Faça no máximo 3 buscas. Se os trechos não trouxerem a resposta, "
        "diga que ela não foi encontrada no roteiro em vez de continuar buscando."
    ),
    tools=[search_script],
)

# ------------------------------------------------------------------
# 5. Faz as perguntas
# ------------------------------------------------------------------
query = "Onde o Doc diz para o Marty encontrá-lo, e a que horas?"
result = Runner.run_sync(agent, query)
print("\n--- RESPOSTA ---\n", result.final_output)

query = "O que acontece à 1:15 da manhã?"
result = Runner.run_sync(agent, query)
print("\n--- RESPOSTA ---\n", result.final_output)
