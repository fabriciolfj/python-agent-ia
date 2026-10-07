from openai import OpenAI
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

# Carrega as configurações do Ollama do arquivo .env (padrão: Ollama local)
load_dotenv()
ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
# O Ollama expõe uma API compatível com a OpenAI; a chave de API é exigida pelo cliente, mas ignorada
client = OpenAI(base_url=ollama_base_url, api_key='ollama')

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

# Converte os embeddings em um array numpy para o PCA
embeddings_array = np.array(embeddings)

print(embeddings_array.shape)

# Aplicando PCA para reduzir as dimensões para 3
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings_array)

# Criando um gráfico 3D com o Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_embeddings[:,0],
    y=reduced_embeddings[:,1],
    z=reduced_embeddings[:,2],
    mode='markers+text',
    text=documents,  # Adiciona os textos dos documentos ao passar o mouse
    hoverinfo='text',  # Mostra apenas o texto ao passar o mouse
    marker=dict(
        size=12,
        color=list(range(len(documents))),  # Atribui uma cor única a cada documento
        opacity=0.8
    )
)])

# Adicionando títulos e rótulos ao gráfico
fig.update_layout(title="Gráfico 3D dos Embeddings dos Documentos",
                  scene=dict(
                      xaxis_title='Componente PCA 1',
                      yaxis_title='Componente PCA 2',
                      zaxis_title='Componente PCA 3'
                  ))

fig.show()
