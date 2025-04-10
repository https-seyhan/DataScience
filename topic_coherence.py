from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

# Prepare input
texts = [['human', 'interface', 'computer'], ...]  # Tokenized docs
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# For LDA / NMF: provide the actual model
coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()
