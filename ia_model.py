import os
import joblib
from wayIA import process_pdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def save_model(model, vectorizer, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Modelo salvo em: {model_path}")
    print(f"Vectorizador salvo em: {vectorizer_path}")


def load_model(model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Modelo e vetorizador carregados com sucesso!")
    return model, vectorizer


def train_ia_model(pdf_paths, labels, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    processed_texts = [process_pdf(pdf_path) for pdf_path in pdf_paths]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    save_model(model, vectorizer, model_path, vectorizer_path)
    return model, vectorizer


def classify_new_pdf(pdf_path, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    model, vectorizer = load_model(model_path, vectorizer_path)
    processed_text = process_pdf(pdf_path)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)
    return "Verdadeiro" if prediction[0] == 1 else "Falso"

if __name__ == "__main__":
    pdf_files = [
        "C:/Users/u23523/Downloads/bo.pdf",
        "C:/Users/u23523/Downloads/cartaIfood.pdf",
        "C:/Users/u23523/Downloads/Furto Simples.pdf",
        "C:/Users/u23523/Downloads/Acidente de Trânsito com Vítima.pdf",
        "C:/Users/u23523/Downloads/Carta de Apresentação Falsa.pdf"
    ]
    labels = [1, 0, 1, 1, 0]  
    
  
    train_ia_model(pdf_files, labels)


    result = classify_new_pdf("/path/to/new_pdf.pdf")
    print(f"Resultado da classificação: {result}")
