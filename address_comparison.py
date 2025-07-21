import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

def clean_address(address):
    """Clean address strings by lowercasing and removing punctuation."""
    if pd.isna(address):
        return ""
    return str(address).lower().replace(',', '').replace('.', '').strip()

def bayesian_similarity(score, prior_match=0.5, evidence_strength=10):
    """Calculate Bayesian posterior probability of address match."""
    # Prior probability of a match
    prior_odds = prior_match / (1 - prior_match)
    
    # Likelihood ratio based on cosine similarity score
    # High similarity (score close to 1) supports match, low similarity supports non-match
    likelihood_ratio = (score ** evidence_strength) / ((1 - score) ** evidence_strength)
    
    # Posterior odds
    posterior_odds = prior_odds * likelihood_ratio
    
    # Convert odds to probability
    posterior_prob = posterior_odds / (1 + posterior_odds)
    return posterior_prob

def compare_addresses(df1, df2, address_col1, address_col2):
    """Compare addresses between two DataFrames using TF-IDF and Bayesian probability."""
    # Clean addresses
    addresses1 = df1[address_col1].apply(clean_address)
    addresses2 = df2[address_col2].apply(clean_address)
    
    # Combine addresses for vectorization
    all_addresses = pd.concat([addresses1, addresses2], ignore_index=True)
    
    # Vectorize addresses using TF-IDF
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(all_addresses)
    
    # Split back into two matrices
    n1 = len(addresses1)
    tfidf1 = tfidf_matrix[:n1]
    tfidf2 = tfidf_matrix[n1:]
    
    # Compute cosine similarities (vectorized)
    cosine_similarities = np.array([1 - cosine(tfidf1[i].toarray().flatten(), 
                                              tfidf2[i].toarray().flatten()) 
                                   for i in range(min(n1, len(addresses2)))])
    
    # Apply Bayesian similarity
    probabilities = np.array([bayesian_similarity(score) for score in cosine_similarities])
    
    # Create result DataFrame
    results = pd.DataFrame({
        'address1': addresses1[:len(probabilities)],
        'address2': addresses2[:len(probabilities)],
        'cosine_similarity': cosine_similarities,
        'match_probability': probabilities
    })
    
    return results

# Example usage
if __name__ == "__main__":
    # Sample DataFrames
    df1 = pd.DataFrame({
        'address': [
            '123 Main St, Springfield',
            '456 Oak Ave, Boston',
            '789 Pine Rd, Seattle'
        ]
    })
    
    df2 = pd.DataFrame({
        'address': [
            '123 Main Street, Springfield',
            '456 Oak Avenue, Boston',
            '101 Elm St, Seattle'
        ]
    })
    
    # Compare addresses
    result = compare_addresses(df1, df2, 'address', 'address')
    print(result)
