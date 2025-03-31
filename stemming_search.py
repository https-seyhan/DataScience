import spacy
from nltk.stem import PorterStemmer
from spacy.tokens import Doc

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Function to stem the words in a SpaCy Doc
def stem_tokens(doc):
    return [stemmer.stem(token.text) for token in doc]

# Add custom component to the pipeline
def custom_pipeline_component(doc):
    stemmed_tokens = stem_tokens(doc)
    return Doc(doc.vocab, words=stemmed_tokens)

# Add component after the tokenizer
nlp.add_pipe(custom_pipeline_component, name='stemmer', last=True)

# Example text
text = "The boys are running and ran faster than their friends."

# Process the text
doc = nlp(text)

# Display stemmed words
print([token.text for token in doc])

# Example of a stemming-based search
def search_stemmed(doc, query):
    query_stem = stemmer.stem(query)
    return [token.text for token in doc if token.text == query_stem]

# Searching for the stem of 'run'
print(search_stemmed(doc, 'run'))  # Output will match 'run', 'ran', 'running'
