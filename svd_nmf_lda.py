import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 📚 Sample corpus
documents = [
    "AI is transforming the construction industry.",
    "Generative AI models are used for construction schedule optimization.",
    "Machine learning improves construction safety.",
    "Football is a popular sport worldwide.",
    "The World Cup is the most watched football event.",
    "Messi and Ronaldo are famous football players."
]

# 🧼 Preprocessing
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(documents)
terms = vectorizer.get_feature_names_out()

count_vectorizer = CountVectorizer(stop_words='english')
count = count_vectorizer.fit_transform(documents)
count_terms = count_vectorizer.get_feature_names_out()

# 🔹 SVD (LSA)
svd = TruncatedSVD(n_components=2, random_state=42)
svd.fit(tfidf)
print("\n🔷 SVD (LSA) Topics")
for i, comp in enumerate(svd.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:5]
    print(f"Topic {i+1}: {[t[0] for t in sorted_terms]}")

# 🔹 NMF
nmf = NMF(n_components=2, random_state=42)
nmf.fit(tfidf)
print("\n🔶 NMF Topics")
for i, topic in enumerate(nmf.components_):
    print(f"Topic {i+1}: {[terms[i] for i in topic.argsort()[-5:][::-1]]}")

# 🔹 LDA
# Convert count matrix to gensim corpus
corpus = [count[i].nonzero()[1] for i in range(count.shape[0])]
gensim_corpus = [list(zip(doc, [1]*len(doc))) for doc in corpus]
dictionary = corpora.Dictionary([[count_terms[i] for i in doc] for doc in corpus])

lda = LdaModel(corpus=gensim_corpus, id2word=dictionary, num_topics=2, random_state=42)
print("\n🟡 LDA Topics")
for i, topic in lda.print_topics():
    print(f"Topic {i+1}: {topic}")
