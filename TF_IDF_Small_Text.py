from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["Construction scheduling AI", "Machine learning in real estate", "NLP for assumption identification"]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# Get important words
feature_names = vectorizer.get_feature_names_out()
for i, text in enumerate(texts):
    scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print(f"Top keywords for text {i+1}: {sorted_scores[:3]}")
