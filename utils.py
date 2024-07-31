import PyPDF2
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
import nltk
import pandas as pd

# Initialize spaCy model
def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    return nlp

nlp = load_spacy_model()

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.

    Parameters:
    - file_path: Path to the PDF file.

    Returns:
    - text: Extracted text from the PDF.
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() if page.extract_text() else ""
        return text
    except FileNotFoundError:
        return ""

def preprocess_text(text, lang='en'):
    """
    Preprocess text by tokenizing, lowercasing, removing stopwords, stemming, and lemmatizing.

    Parameters:
    - text: Input text to preprocess.
    - lang: Language of the text.

    Returns:
    - preprocessed_text: Preprocessed text.
    """
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    
    stop_words = set(stopwords.words('english')) if lang == 'en' else set()
    tokens = [word for word in tokens if word not in stop_words]

    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    preprocessed_text = " ".join(lemmatized_tokens)
    return preprocessed_text

def extract_data_with_gpt(input_text):
    """
    Extract customer names, product names, and review texts from the input text.
    
    Parameters:
    - input_text: The text to extract data from.
    
    Returns:
    - extracted_data: customer_name, product_name, review_text
    """
    try:
        customer_names = []
        product_names = []
        review_texts = []

        doc = nlp(input_text)
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

        action_product_pattern = re.compile(r'(bought|got|tried|used|tested)\s+(.+?)[.,]', re.IGNORECASE)
        entries = re.split(r'Entry\s+\d+:', input_text)

        for entry in entries[1:]:
            customer_name = next((name for name in names if name in entry), 'NULL')
            customer_names.append(customer_name)
            
            match_product = action_product_pattern.search(entry)
            product_name = match_product.group(2).strip() if match_product else 'NULL'
            product_names.append(product_name)
            
            review_match = re.search(r'\.\s*(.*)', entry)
            review_text = review_match.group(1).strip() if review_match else 'NULL'
            review_texts.append(review_text)
        
        max_len = max(len(customer_names), len(product_names), len(review_texts))
        customer_names += ['NULL'] * (max_len - len(customer_names))
        product_names += ['NULL'] * (max_len - len(product_names))
        review_texts += ['NULL'] * (max_len - len(review_texts))
        
        return customer_names, product_names, review_texts
    
    except Exception as e:
        print(f"Error in extract_data_with_gpt: {e}")
        return ['NULL'], ['NULL'], ['NULL']

def create_dataframe(customer_names, product_names, review_texts, best_sentiments):
    data = {
        "Customer Name": customer_names,
        "Product Name": product_names,
        "Review Text": review_texts,
        "Sentiment": best_sentiments
    }
    df = pd.DataFrame(data)
    return df

def detect_language(text):
    """
    Detect the language of the input text.

    Parameters:
    - text: Input text to detect language.

    Returns:
    - lang: Detected language code.
    """
    return 'en'  # Assuming 'en' for now
