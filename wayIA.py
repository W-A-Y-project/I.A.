import PyPDF2
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from symspellpy.symspellpy import SymSpell
import importlib.resources

# Usa o português do spaCy
nlp = spacy.load('pt_core_news_lg')

# Recursos do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Dicionário de abreviações e as formas completas
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
        text = re.sub(r'\b' + abbr + r'\b', full_form, text)  # Normaliza abreviações
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
            text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() or ''  # Caso a extração falhe
        return text
    except Exception as e:
        print(f"Erro ao ler o PDF: {e}")
        return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove múltiplos espaços
    text = text.strip()  # Remove espaços no início e no fim
    text = re.sub(r'[^\w\s,.]', '', text)  # Remove pontuação desnecessária
    text = re.sub(r'\d+', '', text)  # Remove números, se não forem relevantes
    return text

def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and token.lemma_ not in stopwords.words('portuguese')])

def classify_vehicle(entities):
    classified_entities = []
    for i, (entity, label) in enumerate(entities):
        # Verifica se a entidade é uma marca
        for brand in car_brands:
            if brand.lower() in entity.lower():  # Ignora maiúsculas/minúsculas
                classified_entities.append((entity, 'VEÍCULO'))
                break
        else:
            classified_entities.append((entity, label))  # Mantém original se não for um veículo
    return classified_entities

def separate_person_info(entity):
    if 'cpf' in entity and 'rg' in entity:
        return entity.replace('cpf', '').replace('rg', '').strip(), 'CPF e RG'
    return entity, None

# Extrai as entidades
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text.lower(), ent.label_) for ent in doc.ents]
    unique_entities = list({entity[0]: entity for entity in entities}.values())  # Remove duplicatas
    unique_entities = classify_vehicle(unique_entities)  # Classifica veículos

    processed_entities = []
    for entity, label in unique_entities:
        if label == 'PER':
            processed_entities.append((entity, 'PESSOA')) # Classifica PER como PESSOA
        else:
            processed_entities.append((entity, label))
    return processed_entities


def process_pdf(pdf_path):
    text = read_pdf(pdf_path)
    cleaned_text = clean_text(text)
    corrected_text = correct_spelling(cleaned_text, symspell)
    normalized_text = normalize_abbreviations(corrected_text)
    lemmatized_text = lemmatize_text(normalized_text)

    entities = extract_entities(lemmatized_text)
    # Filtrar entidades 
    entities = [(text, label) for text, label in entities if text != 'dp']
    return lemmatized_text, entities

# Função da Vetorização
def vectorize_text(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

# PDF que vai ser lido
pdf_file = r"Z:\I.A\exemplo2.pdf"
processed_text, extracted_entities = process_pdf(pdf_file)

# Exibe as entidades extraídas
print("Entidades extraídas: ", extracted_entities)

# Vetorização
vectors, vectorizer = vectorize_text([processed_text])
print(vectors.toarray())

feature_names = vectorizer.get_feature_names_out()
print(feature_names)
