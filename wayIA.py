import PyPDF2
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources

# Usa o português do spaCy
nlp = spacy.load('pt_core_news_sm')

# Recursos do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Dicionário de abreviações e as formas completas (É PARA ADICIONAR MAIS CONFORME O NECESSÁRIO!!!!!!!)
abbreviation_dict = {
    'DP': 'Delegacia de Polícia',
    'BO': 'Boletim de Ocorrência',
    'CPF': 'Cadastro de Pessoa Física',
    'RG': 'Registro Geral',

}

def normalize_abbreviations(text):
    # Substitui as abreviações para as palavras
    for abbr, full_form in abbreviation_dict.items():
        text = re.sub(r'\b' + abbr + r'\b', full_form, text)  # Substitui as abreviações exatas
    return text

# Correção ortográfica do SymSpell
def load_symspell():
    # Cria uma instância do SymSpell
    symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    # Carrega o dicionário de palavras frequentes
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_pt_br.txt")
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_pt_br.txt")
    
    symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    symspell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    return symspell

def correct_spelling(text, symspell):
    # Corrige o texto
    suggestions = symspell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# Inicia o SymSpell
symspell = load_symspell()

def read_pdf(pdf_path):
    # Lê o PDF e retorna o tezto
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def clean_text(text):
    # Remove caracteres especiais
    text = re.sub(r'\s+', ' ', text)  # Remove espaços desnecessários
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
    text = re.sub(r'\d+', '', text)  # Remove os números
    return text

def tokenize_text(text):
    # Tokeniza o texto
    return nltk.word_tokenize(text)

def remove_stopwords(words):
    # Remove as stopwords
    stop_words = set(stopwords.words('portuguese'))
    return [word for word in words if word.lower() not in stop_words]

def lemmatize_text(text):
    # lematiza o texto usando o spaCy
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

def process_pdf(pdf_path):
    # Aplica as funcionalidades no pdf
    text = read_pdf(pdf_path)
    cleaned_text = clean_text(text)
    corrected_text = correct_spelling(cleaned_text, symspell)  # Corrige a ortografia do texto
    normalized_text = normalize_abbreviations(corrected_text)  # Normaliza abreviações
    lemmatized_text = lemmatize_text(normalized_text)  # Lematiza o texto
    return lemmatized_text  # Retorna o texto processado

def vectorize_text(texts):
    # Vetoriza o texto usando TF-IDF.
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

# PDF que vai ser lido
pdf_file = r"Z:\I.A\Design sem nome.pdf"
processed_text = process_pdf(pdf_file)

# Vetorização
vectors, vectorizer = vectorize_text([processed_text])  # Passando uma lista de textos
print(vectors.toarray())  # Exibe a matriz de TF-IDF

feature_names = vectorizer.get_feature_names_out()
print(feature_names)
