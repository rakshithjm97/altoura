# sentiment_analysis.py

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import spacy
from preprocessing import preprocess_text
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_text
from transformers import pipeline


# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the sentiment analysis pipeline with a specified model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize Transformers sentiment analysis
transformers_pipeline = pipeline("sentiment-analysis")
# Sentiment analysis functions

def determine_sentiment_textblob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

def determine_sentiment_vader(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound > 0.05:
        return "positive"
    elif compound < -0.05:
        return "negative"
    else:
        return "neutral"



# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

def determine_sentiment_spacy(text):
    doc = nlp(text)
    # Initialize sentiment scores
    sentiment = {'pos': 0.0, 'neg': 0.0, 'neutral': 0.0}
    count = 0
    # Iterate through each token in the document
    for token in doc:
        # Check if the token has a sentiment score
        if token.sentiment:
            # Update sentiment scores
            sentiment['pos'] += token.sentiment.pos
            sentiment['neg'] += token.sentiment.neg
            count += 1

    if count > 0:
        # Average the sentiment scores
        sentiment['pos'] /= count
        sentiment['neg'] /= count

    # Determine the overall sentiment label based on average scores
    if sentiment['pos'] > sentiment['neg']:
        return "positive"
    elif sentiment['neg'] > sentiment['pos']:
        return "negative"
    else:
        return "neutral"

def determine_sentiment_transformers(text):
    result = transformers_pipeline(text)[0]
    label = result['label']
    if label == 'POSITIVE':
        return "positive"
    elif label == 'NEGATIVE':
        return "negative"
    else:
        return "neutral"

true_labels = [
    ("John Doe", "AwesomeWidget", "This widget is absolutely amazing! It has made my life so much easier. Highly recommended.", "positive"),
    ("Jane Smith", "MediocreWidget", "It's okay, but not as great as I expected. It works fine, but there are better options out there.", "neutral"),
    ("Bob Johnson", "TerribleWidget", "This is the worst widget I've ever used. It broke after just one week. Very disappointed.", "negative"),
    ("Alice Brown", "SuperWidget", "It is fantastic. It exceeded all her expectations.", "positive"),
    ("Charlie Davis", "BasicWidget", "He found it decent but lacking some features compared to others.", "neutral"),
    ("Emily Evans", "DeluxeWidget", "She was impressed with its performance and reliability.", "positive"),
    ("Frank Green", "SimpleWidget", "He found it to be very user-friendly and efficient.", "positive"),
    ("Grace Harris", "PremiumWidget", "She mentioned it was worth every penny.", "positive"),
    ("Henry White", "AdvancedWidget", "It was good but had some issues.", "neutral"),
    ("Ivy Black", "EconomyWidget", "She thought it was below average in terms of quality.", "negative")
]

#  approach to convert model names to sentiment labels
def compare_models_best_sentiment(dataset):
    true_labels_list = [label for _, _, _, label in dataset]
    texts = [text for _, _, text, _ in dataset]
    preprocessed_texts = [preprocess_text(text) for text in texts]

    predictions_textblob = [determine_sentiment_textblob(text) for text in preprocessed_texts]
    predictions_vader = [determine_sentiment_vader(text) for text in preprocessed_texts]
    predictions_transformers = [determine_sentiment_transformers(text) for text in preprocessed_texts]
    predictions_spacy = [determine_sentiment_spacy(text) for text in preprocessed_texts]

    best_sentiments = []
    for textblob, vader, transformers, spacy in zip(predictions_textblob, predictions_vader, predictions_transformers, predictions_spacy):
        # Determine the "best" sentiment label among the models
        sentiments = [textblob, vader, transformers, spacy]
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
        best_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        best_sentiments.append(best_sentiment)

    return best_sentiments

def compare_models_accuracy(dataset):
    true_labels_list = [label for _, _, _, label in dataset]
    texts = [text for _, _, text, _ in dataset]
    preprocessed_texts = [preprocess_text(text) for text in texts]

    predictions_textblob = [determine_sentiment_textblob(text) for text in preprocessed_texts]
    predictions_vader = [determine_sentiment_vader(text) for text in preprocessed_texts]
    predictions_transformers = [determine_sentiment_transformers(text) for text in preprocessed_texts]
    predictions_spacy = [determine_sentiment_spacy(text) for text in preprocessed_texts]

    accuracy_textblob = accuracy_score(true_labels_list, predictions_textblob)
    accuracy_vader = accuracy_score(true_labels_list, predictions_vader)
    accuracy_transformers = accuracy_score(true_labels_list, predictions_transformers)
    accuracy_spacy = accuracy_score(true_labels_list, predictions_spacy)

    print("TextBlob Classification Report:")
    print(classification_report(true_labels_list, predictions_textblob))
    print("VADER Classification Report:")
    print(classification_report(true_labels_list, predictions_vader))
    print("Transformers Classification Report:")
    print(classification_report(true_labels_list, predictions_transformers))
    print("spaCy Classification Report:")
    print(classification_report(true_labels_list, predictions_spacy))

    return {
        "TextBlob": accuracy_textblob,
        "VADER": accuracy_vader,
        "Transformers": accuracy_transformers,
        "spaCy": accuracy_spacy
}
if __name__ == "__main__":
    best_sentiments = compare_models_best_sentiment(true_labels)
    print("Best Sentiments:", best_sentiments)
