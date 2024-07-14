from langdetect import detect
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK components
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, lang='en'):
    """
    Preprocess text by tokenizing, lowercasing, removing stopwords, stemming, and lemmatizing.

    Parameters:
    - text: Input text to preprocess.
    - lang: Language of the text.

    Returns:
    - preprocessed_text: Preprocessed text.
    """
    tokens = word_tokenize(text.lower())

    if lang == 'en':
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    combined_tokens = set(stemmed_tokens + lemmatized_tokens)
    preprocessed_text = " ".join(combined_tokens)
    return preprocessed_text

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
