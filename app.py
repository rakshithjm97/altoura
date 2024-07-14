import streamlit as st
import tempfile
from utils import extract_text_from_pdf, extract_data_with_gpt, create_dataframe, preprocess_text, detect_language
from sentiment_analysis import determine_sentiment_textblob, determine_sentiment_spacy, determine_sentiment_vader, determine_sentiment_transformers
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def main():
    st.title("PDF Sentiment Analysis and Model Comparison")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
        
        text = extract_text_from_pdf(file_path)
        
        lang = detect_language(text)
        st.write(f"Detected Language: {lang}")
        
        extracted_data = extract_data_with_gpt(text)
        
        st.write("Extracted Data:", extracted_data)
        
        if isinstance(extracted_data, tuple) and len(extracted_data) == 3:
            customer_names, product_names, review_texts = extracted_data
        else:
            st.error("Error: Unexpected format from extract_data_with_gpt")
            return
        
        # Perform sentiment analysis using different methods
        best_sentiments = []
        for review_text in review_texts:
            preprocessed_text = preprocess_text(review_text, lang)
            sentiments = {
                'TextBlob': determine_sentiment_textblob(preprocessed_text),
                'VADER': determine_sentiment_vader(preprocessed_text),
                'Transformers': determine_sentiment_transformers(preprocessed_text),
                'spaCy': determine_sentiment_spacy(preprocessed_text)
            }
            # Determine the most frequent sentiment
            sentiment_counts = {sentiment: list(sentiments.values()).count(sentiment) for sentiment in sentiments.values()}
            best_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            best_sentiments.append(best_sentiment)
        
        # Create DataFrame with extracted data and best sentiments
        df = create_dataframe(customer_names, product_names, review_texts, best_sentiments)
        
        st.write("Extracted Customer Names:")
        st.write(customer_names)
        st.write("Extracted Product Names:")
        st.write(product_names)
        st.write("Extracted Review Texts:")
        st.write(review_texts)
        st.write("Best Sentiments Analysis Results:")
        st.write(df)

if __name__ == "__main__":
    main()
