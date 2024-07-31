# setup.py
import spacy
from spacy.cli import download

def install_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")

if __name__ == "__main__":
    install_spacy_model()
