import PyPDF2
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from symspellpy.symspellpy import SymSpell
import importlib.resources

# Carrega o modelo de linguagem em português do spaCy
nlp = spacy.load('pt_core_news_lg')

# Baixa os recursos do NLTK, como as palavras paradas (stopwords)
nltk.download('punkt')
nltk.download('stopwords')

# Dicionário de abreviações que vamos normalizar (ex: 'DP' vira 'Delegacia de Polícia')
abbreviation_dict = {
    'DP': 'Delegacia de Polícia',
    'BO': 'Boletim de Ocorrência',
    'CPF': 'Cadastro de Pessoa Física',
    'RG': 'Registro Geral',
}

# Lista de marcas de carros (para identificar veículos no texto)
car_brands = [
    'Aston Martin', 'Audi', 'BMW', 'BYD', 'CAOA Chery', 'Chevrolet',
    'Citroën', 'Effa', 'Ferrari', 'Fiat', 'Ford', 'Foton', 'GWM',
    'Honda', 'Hyundai', 'Iveco', 'JAC', 'Jaguar', 'Jeep', 'Kia',
    'Lamborghini', 'Land Rover', 'Lexus', 'Maserati', 'McLaren',
    'Mercedes-AMG', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Neta',
    'Nissan', 'Peugeot', 'Porsche', 'RAM', 'Renault', 'Rolls-Royce',
    'Seres', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen'
]

# Função que substitui as abreviações pelo texto completo
def normalize_abbreviations(text):
    for abbr, full_form in abbreviation_dict.items():
        text = re.sub(r'\b' + abbr + r'\b', full_form, text)  # Substitui a abreviação pelo texto
    return text

# Função para carregar o corretor ortográfico (SymSpell)
def load_symspell():
    symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    with importlib.resources.path("symspellpy", "frequency_dictionary_pt_br.txt") as dictionary_path:
        symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    with importlib.resources.path("symspellpy", "frequency_bigramdictionary_pt_br.txt") as bigram_path:
        symspell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    return symspell

# Função para corrigir erros de digitação no texto
def correct_spelling(text, symspell):
    suggestions = symspell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# Carrega o corretor ortográfico
symspell = load_symspell()

# Função para ler o conteúdo de um PDF e extrair o texto
def read_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])
        return text
    except Exception as e:
        print(f"Erro ao ler o PDF: {e}")
        return ""

# Função para limpar o texto (remover espaços extras, pontuação desnecessária, números, etc.)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove múltiplos espaços
    text = text.strip()  # Remove espaços no começo e no fim
    text = re.sub(r'[^\w\s,.]', '', text)  # Remove pontuação que não é necessária
    text = re.sub(r'\d+', '', text)  # Remove números (se não forem importantes)
    return text

# Função para "reduzir" as palavras para sua forma base (ex: "falando" vira "falar")
def lemmatize_text(text):
    doc = nlp(text)  # Usa o spaCy para processar o texto
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and token.lemma_ not in stopwords.words('portuguese')])

# Função para identificar se a entidade é um veículo e classificar como 'VEÍCULO'
def classify_vehicle(entities):
    classified_entities = []
    for entity, label in entities:
        # Verifica se o nome do veículo está na lista de marcas
        if any(brand.lower() in entity.lower() for brand in car_brands):
            classified_entities.append((entity, 'VEÍCULO'))  # Marca como VEÍCULO
        else:
            classified_entities.append((entity, label))  # Mantém a classificação original
    return classified_entities

# Função para extrair as entidades do texto processado
def extract_entities(text):
    doc = nlp(text)  # Usa o spaCy para encontrar as entidades
    entities = [(ent.text.lower(), ent.label_) for ent in doc.ents]  # Extrai as entidades
    unique_entities = list({entity[0]: entity for entity in entities}.values())  # Remove duplicatas
    unique_entities = classify_vehicle(unique_entities)  # Classifica veículos
    # Se a entidade for um "PER" (pessoa), marca como 'PESSOA'
    return [(entity, 'PESSOA' if label == 'PER' else label) for entity, label in unique_entities]

# Função principal para processar o PDF
def process_pdf(pdf_path):
    # Lê o PDF, limpa, corrige e faz a lematização do texto
    text = read_pdf(pdf_path)
    cleaned_text = clean_text(text)
    corrected_text = correct_spelling(cleaned_text, symspell)
    normalized_text = normalize_abbreviations(corrected_text)
    lemmatized_text = lemmatize_text(normalized_text)
    
    # Extrai as entidades do texto
    entities = extract_entities(lemmatized_text)
    # Remove 'dp' das entidades (caso exista)
    entities = [(text, label) for text, label in entities if text != 'dp']
    return lemmatized_text, entities
