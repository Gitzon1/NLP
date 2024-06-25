from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Example text
text = "Life is a Goal"

# Step 1: Tokenization and Stop Word Removal
tokens = simple_preprocess(text)
filtered_tokens = [token for token in tokens if token not in STOPWORDS]
print("Filtered Tokens:", filtered_tokens)

# Step 2: Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("Stemmed Tokens:", stemmed_tokens)

# Step 3: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("Lemmatized Tokens:", lemmatized_tokens)
