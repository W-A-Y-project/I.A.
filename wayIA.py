import PyPDF2
import spacy
import nltk
from nltk.corpus import stopwords
import re
from spellchecker import SpellChecker

nlp = spacy.load('pt_core_news_lg')
nltk.download('punkt')
nltk.download('stopwords')

abbreviation_dict = {
    'DP': 'Delegacia de Polícia',
    'BO': 'Boletim de Ocorrência',
    'CPF': 'Cadastro de Pessoa Física',
    'RG': 'Registro Geral',
}

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

def correct_spelling(text):
    spell = SpellChecker(language='pt')
    words = text.split()
    corrected_words = [spell.correction(word) if word else word for word in words]
    corrected_words = [word if word is not None else '' for word in corrected_words] 
    return ' '.join(corrected_words)


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
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s,.]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and token.lemma_ not in stopwords.words('portuguese')])

def classify_vehicle(entities):
    classified_entities = []
    for entity, label in entities:
        if any(brand.lower() in entity.lower() for brand in car_brands):
            classified_entities.append((entity, 'VEÍCULO'))
        else:
            classified_entities.append((entity, label))
    return classified_entities

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text.lower(), ent.label_) for ent in doc.ents]
    unique_entities = list({entity[0]: entity for entity in entities}.values())
    unique_entities = classify_vehicle(unique_entities)
    return [(entity, 'PESSOA' if label == 'PER' else label) for entity, label in unique_entities]

def process_pdf(pdf_path):
    text = read_pdf(pdf_path)
    cleaned_text = clean_text(text)
    corrected_text = correct_spelling(cleaned_text)
    normalized_text = normalize_abbreviations(corrected_text)
    lemmatized_text = lemmatize_text(normalized_text)
    return lemmatized_text
