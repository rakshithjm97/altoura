import PyPDF2
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
import nltk
import pandas as pd

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

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

    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    combined_tokens = set(stemmed_tokens + lemmatized_tokens)
    preprocessed_text = " ".join(combined_tokens)
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
            customer_name = None
            for name in names:
                if name in entry:
                    customer_name = name
                    break
            customer_names.append(customer_name if customer_name else 'NULL')
            
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
    
    except AttributeError as ae:
        print(f"AttributeError in extract_data_with_gpt: {ae}")
        return None
    except Exception as e:
        print(f"Error in extract_data_with_gpt: {e}")
        return None

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
