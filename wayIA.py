import PyPDF2
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from symspellpy import SymSpell, Verbosity
import importlib.resources

# Configurações iniciais
nlp = spacy.load('pt_core_news_lg')

# Baixar stopwords se necessário
try:
    stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')

# Dicionário de abreviações
abbreviation_dict = {
    'DP': 'Delegacia de Polícia',
    'BO': 'Boletim de Ocorrência',
    'CPF': 'Cadastro de Pessoa Física',
    'RG': 'Registro Geral',
}

# Lista de marcas de veículos
car_brands = [
    'Aston Martin', 'Audi', 'BMW', 'BYD', 'CAOA Chery', 'Chevrolet',
    'Citroën', 'Effa', 'Ferrari', 'Fiat', 'Ford', 'Foton', 'GWM',
    'Honda', 'Hyundai', 'Iveco', 'JAC', 'Jaguar', 'Jeep', 'Kia',
    'Lamborghini', 'Land Rover', 'Lexus', 'Maserati', 'McLaren',
    'Mercedes-AMG', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Neta',
    'Nissan', 'Peugeot', 'Porsche', 'RAM', 'Renault', 'Rolls-Royce',
    'Seres', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen'
]

def normalize_abbreviations(text):
    for abbr, full_form in abbreviation_dict.items():
        text = re.sub(r'\b' + abbr + r'\b', full_form, text)
    return text

def load_symspell():
    symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    with importlib.resources.path("symspellpy", "frequency_dictionary_pt_br.txt") as dictionary_path:
        symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    with importlib.resources.path("symspellpy", "frequency_bigramdictionary_pt_br.txt") as bigram_path:
        symspell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    return symspell

def correct_spelling(text, symspell):
    suggestions = symspell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

symspell = load_symspell()

def read_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])
        return text
    except Exception as e:
        print(f"Erro ao ler o PDF: {e}")
        return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s,.]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and token.lemma_ not in stopwords.words('portuguese')])

def classify_vehicle(entities):
    return [(entity, 'VEÍCULO' if any(brand.lower() in entity.lower() for brand in car_brands) else label) for entity, label in entities]

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text.lower(), ent.label_) for ent in doc.ents]
    unique_entities = list({entity[0]: entity for entity in entities}.values())
    unique_entities = classify_vehicle(unique_entities)
    return [(entity, 'PESSOA' if label == 'PER' else label) for entity, label in unique_entities]

def process_pdf(pdf_path):
    text = read_pdf(pdf_path)
    if not text:
        return "", []

    cleaned_text = clean_text(text)
    corrected_text = correct_spelling(cleaned_text, symspell)
    normalized_text = normalize_abbreviations(corrected_text)
    lemmatized_text = lemmatize_text(normalized_text)

    entities = extract_entities(lemmatized_text)
    entities = [(text, label) for text, label in entities if text != 'dp']
    return lemmatized_text, entities

def vectorize_text(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

# Teste de processamento do PDF
pdf_file = r"Z:\I.A\exemplo2.pdf"
processed_text, extracted_entities = process_pdf(pdf_file)

print("Entidades extraídas: ", extracted_entities)

# Vetorização
vectors, vectorizer = vectorize_text([processed_text])
print(vectors.toarray())
print(vectorizer.get_feature_names_out())
