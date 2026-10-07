from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb

# Carrega a chave de API do arquivo .env
load_dotenv()
ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434/v1')
client = OpenAI(base_url=ollama_url, api_key='ollama')


def get_embedding(text, model="nomic-embed-text:latest"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Documentos de exemplo
documents = [
    "O céu é azul e bonito.",
    "Adoro este céu azul e bonito!",
    "A raposa marrom rápida pula sobre o cachorro preguiçoso.",
    "O café da manhã de um rei tem salsichas, presunto, bacon, ovos, torrada e feijão",
    "Eu adoro ovos verdes, presunto, salsichas e bacon!",
    "A raposa marrom é rápida e o cachorro azul é preguiçoso!",
    "O céu está muito azul e o céu está muito bonito hoje",
    "O cachorro é preguiçoso, mas a raposa marrom é rápida!"
]

# Gera os embeddings de cada documento
embeddings = [get_embedding(doc) for doc in documents]
ids = [f"id{i}" for i in range(len(documents))]

#cria o cliente do banco de dados chroma
chroma_client = chromadb.Client()
#cria uma coleção
collection = chroma_client.create_collection(name="documents")

collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=ids
)

def query_chromadb(query, top_n=2):
    """Retorna o texto dos top_n resultados da coleção do ChromaDB
    """
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )
    return [(id, score, text) for id, score, text in
            zip(results['ids'][0], results['distances'][0], results['documents'][0])]


# Loop de entrada para as consultas de busca
while True:
    query = input("Digite uma consulta de busca (ou 'sair' para encerrar): ")
    if query.lower() == 'sair':
        break
    top_n = int(input("Quantos resultados principais você deseja ver? "))
    search_results = query_chromadb(query, top_n)

    print("Documentos mais relevantes:")
    for id, score, text in search_results:
        print(f"ID:{id} TEXTO: {text} DISTÂNCIA: {round(score, 2)}")

    print("\n")
