import PyPDF2
import nltk
from nltk.corpus import stopwords
nltk.download('punkt_tab')

def remove_stopwords(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    words = nltk.word_tokenize(text)

    nltk.download('stopwords')
    stop_words = set(stopwords.words('portuguese'))

    filtered_words = [word for word in words if word not in stop_words]

    return filtered_words


pdf_file = "/Users/u23523/Downloads/edital_bloco2_versaoretificada09ago2024.pdf"
resultado = remove_stopwords(pdf_file)
print(resultado)