import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats

# --- Sample data ---
df1 = pd.DataFrame({'address': ['123 Main Street, Sydney', '45 Queen Rd, Melbourne', '10 George St, Brisbane']})
df2 = pd.DataFrame({'address': ['123 Main St, Sydney', '46 Queen Road, Melbourne', '10 George Street, Brisbane']})

# --- TF-IDF Vectorization ---
all_addresses = pd.concat([df1['address'], df2['address']])
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
tfidf_matrix = vectorizer.fit_transform(all_addresses)

# Split vectors for the two sets
tfidf_df1 = tfidf_matrix[:len(df1)]
tfidf_df2 = tfidf_matrix[len(df1):]

# --- Cosine Similarity ---
similarity_matrix = cosine_similarity(tfidf_df1, tfidf_df2)

# --- Bayesian Thresholding ---
# Prior belief: matches are rare (e.g., 5%)
prior_match = 0.05
prior_non_match = 1 - prior_match

# Likelihood function (estimated from domain knowledge or tuned empirically)
def likelihood(similarity, match=True):
    # Use Beta distributions as likelihood models
    if match:
        return stats.beta.pdf(similarity, a=5, b=1)  # matches are usually >0.7
    else:
        return stats.beta.pdf(similarity, a=1, b=5)  # non-matches peak near 0

# Posterior probability of match for each pair
posterior_matrix = np.zeros_like(similarity_matrix)

for i in range(similarity_matrix.shape[0]):
    for j in range(similarity_matrix.shape[1]):
        sim = similarity_matrix[i, j]
        likelihood_match = likelihood(sim, match=True)
        likelihood_non_match = likelihood(sim, match=False)
        
        numerator = likelihood_match * prior_match
        denominator = numerator + (likelihood_non_match * prior_non_match)
        
        posterior = numerator / denominator
        posterior_matrix[i, j] = posterior

# --- Output matches with high posterior ---
threshold = 0.90  # posterior probability threshold
matches = []

for i in range(posterior_matrix.shape[0]):
    for j in range(posterior_matrix.shape[1]):
        if posterior_matrix[i, j] >= threshold:
            matches.append({
                'df1_index': i,
                'df2_index': j,
                'df1_address': df1.iloc[i]['address'],
                'df2_address': df2.iloc[j]['address'],
                'similarity': similarity_matrix[i, j],
                'posterior': posterior_matrix[i, j]
            })

# Convert to DataFrame
matches_df = pd.DataFrame(matches)
print(matches_df)
