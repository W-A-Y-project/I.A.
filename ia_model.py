# ia_model.py

from wayIa import process_pdf  # Importa a função para processar o PDF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Função para treinar a IA
def train_ia_model(pdf_paths, labels):
    # Vamos processar o texto de cada PDF
    processed_texts = [process_pdf(pdf_path)[0] for pdf_path in pdf_paths]
    
    # Converte o texto para uma representação numérica que a IA entende (usando TF-IDF)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    
    # Divide os dados em treino e teste pra avaliar a IA
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
    
    # Cria o modelo de IA (Naive Bayes) e treina ele com os dados de treino
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Testa o modelo com os dados de teste e mostra os resultados
    y_pred = model.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_pred))  # Mostra a porcentagem de acerto
    print(classification_report(y_test, y_pred))  # Detalha o desempenho em cada categoria
    
    # Retorna o modelo treinado e o vetor que converte o texto para números
    return model, vectorizer

if __name__ == "__main__":
    # Aqui você coloca os PDFs que quer usar pra treinar a IA
    pdf_files = ["colocar o pdf", "colocar o pdf"]  # Exemplo de caminhos dos PDFs
    labels = [1, 0]  # Exemplo de rótulos (1 para verdadeiro, 0 para falso)

    # Treina a IA com os PDFs e os rótulos definidos acima
    model, vectorizer = train_ia_model(pdf_files, labels)
