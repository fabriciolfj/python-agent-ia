import plotly.graph_objects as go
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

# Vetorização usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Calculando a similaridade de cosseno
cosine_similarities = cosine_similarity(X)

while True:
    # Entrada do usuário para selecionar um documento
    selected_document_index = input(f"Digite o número de um documento (0-{len(documents)-1}) ou 'sair' para encerrar: ").strip()

    if selected_document_index.lower() == 'sair':
        break

    if not selected_document_index.isdigit() or not 0 <= int(selected_document_index) < len(documents):
        print("Entrada inválida. Digite um número de documento válido.")
        continue

    selected_document_index = int(selected_document_index)

    # Extraindo as pontuações de similaridade do documento selecionado
    selected_document_similarities = cosine_similarities[selected_document_index]

    # Trunca textos longos para os rótulos do eixo x
    x_axis_labels = [doc[:50] + "..." if len(doc) > 50 else doc for doc in documents]

    # Plotando a similaridade de cosseno
    fig = go.Figure([go.Bar(x=x_axis_labels,
                            y=selected_document_similarities)])

    fig.update_layout(title=f"Similaridades de cosseno de '{documents[selected_document_index][:50] + '...' if len(documents[selected_document_index]) > 50 else documents[selected_document_index]}' com os demais",
                      xaxis_title="Documento",
                      yaxis_title="Similaridade de cosseno",
                      xaxis={'tickangle': 45})  # Rotaciona os rótulos do eixo x para melhor leitura

    fig.show()
