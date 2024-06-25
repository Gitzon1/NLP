import spacy
from nltk.stem import PorterStemmer

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize the stemmer
stemmer = PorterStemmer()

# Example text
text = "Life is a Goal."

# Step 1: Tokenization, Stop Word Removal, and Lemmatization
doc = nlp(text)
tokens = [token.text for token in doc]
filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

# Step 2: Stemming
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

print("Tokens:", tokens)
print("Filtered Tokens:", filtered_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)
print("Stemmed Tokens:", stemmed_tokens)
