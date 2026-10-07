import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Passo 1: Vetorizar com o TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Passo 2: Armazenar os vetores em um banco de dados vetorial simples (aqui, uma lista)
vector_database = X.toarray()

# Passo 3: Função de busca por similaridade de cosseno
def cosine_similarity_search(query, database, vectorizer, top_n=5):
    query_vec = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(query_vec, database)[0]
    top_indices = np.argsort(-similarities)[:top_n]  # Índices dos n melhores
    return [(idx, similarities[idx]) for idx in top_indices]

# Loop de entrada para as consultas de busca
while True:
    query = input("Digite uma consulta de busca (ou 'sair' para encerrar): ")
    if query.lower() == 'sair':
        break
    top_n = int(input("Quantos resultados principais você deseja ver? "))
    search_results = cosine_similarity_search(query, vector_database, vectorizer, top_n)

    print("Documentos mais relevantes:")
    for idx, score in search_results:
        print(f"- {documents[idx]} (Pontuação: {score:.4f})")

    print("\n")

