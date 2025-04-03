from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "A worker suffered a head injury due to a falling object on site.",
    "A patient experienced an allergic reaction after medication was incorrectly administered.",
    "An employee slipped on a wet floor, resulting in a minor wrist fracture."
]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

feature_names = vectorizer.get_feature_names_out()
for i, text in enumerate(texts):
    scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print(f"Top keywords for incident {i+1}: {sorted_scores[:3]}")
