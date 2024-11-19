from wayIa import process_pdf 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

def train_ia_model(pdf_paths, labels):
    processed_texts = [process_pdf(pdf_path) for pdf_path in pdf_paths]  
    
    print(len(processed_texts), len(labels))
    
    if len(processed_texts) != len(labels):
        raise ValueError("O número de textos processados e rótulos não coincide.")
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_pred)) 
    print(classification_report(y_test, y_pred))  
    return model, vectorizer


if __name__ == "__main__":
    pdf_files = ["/Users/u23523/I.A/boTeste.pdf",
                  "/Users/u23523/Downloads/cartaIfood.pdf",
                  "/Users/u23523/Downloads/Acidente de Trânsito com Vítima.pdf",
                  "/Users/u23523/Downloads/Furto Simples.pdf",
                  "/Users/u23523/Downloads/Carta de Apresentação Falsa.pdf"] 
    labels = [1, 0, 1, 1, 0]  
    model, vectorizer = train_ia_model(pdf_files, labels)