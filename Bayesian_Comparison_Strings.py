import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import re
from collections import defaultdict

class AddressComparator:
    def __init__(self, df1, df2, address_col1='address', address_col2='address'):
        """
        Initialize the AddressComparator with two dataframes containing address columns.
        
        Args:
            df1 (pd.DataFrame): First dataframe containing addresses
            df2 (pd.DataFrame): Second dataframe containing addresses
            address_col1 (str): Column name for addresses in df1
            address_col2 (str): Column name for addresses in df2
        """
        self.df1 = df1.copy()
        self.df2 = df2.copy()
        self.address_col1 = address_col1
        self.address_col2 = address_col2
        
        # Preprocess addresses
        self.df1['processed_address'] = self.df1[address_col1].apply(self._preprocess_address)
        self.df2['processed_address'] = self.df2[address_col2].apply(self._preprocess_address)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        
        # Fit vectorizer on all addresses
        all_addresses = pd.concat([
            self.df1['processed_address'], 
            self.df2['processed_address']
        ])
        self.vectorizer.fit(all_addresses)
        
        # Prior probability (can be adjusted based on domain knowledge)
        self.prior_prob = 0.01  # Assume 1% chance any two random addresses match
        
    def _preprocess_address(self, address):
        """Standardize address format for better comparison"""
        if not isinstance(address, str):
            return ""
            
        # Convert to lowercase
        address = address.lower()
        
        # Remove special characters except numbers, letters, and spaces
        address = re.sub(r'[^a-z0-9\s]', '', address)
        
        # Standardize common address components
        replacements = {
            'street': 'st',
            'avenue': 'ave',
            'road': 'rd',
            'boulevard': 'blvd',
            'drive': 'dr',
            'court': 'ct',
            'lane': 'ln',
            'highway': 'hwy',
            'apartment': 'apt',
            'suite': 'ste',
            'number': 'no'
        }
        
        for full, abbrev in replacements.items():
            address = re.sub(r'\b' + full + r'\b', abbrev, address)
        
        # Remove extra whitespace
        address = ' '.join(address.split())
        
        return address
    
    def _calculate_similarity_metrics(self, address1, address2):
        """Calculate multiple similarity metrics between two addresses"""
        if not address1 or not address2:
            return {
                'cosine_sim': 0,
                'levenshtein_sim': 0,
                'token_sort_ratio': 0,
                'partial_ratio': 0
            }
        
        # TF-IDF cosine similarity
        vec1 = self.vectorizer.transform([address1])
        vec2 = self.vectorizer.transform([address2])
        cosine_sim = cosine_similarity(vec1, vec2)[0][0]
        
        # Levenshtein similarity (normalized)
        levenshtein_sim = fuzz.ratio(address1, address2) / 100
        
        # Token sort ratio
        token_sort_ratio = fuzz.token_sort_ratio(address1, address2) / 100
        
        # Partial ratio
        partial_ratio = fuzz.partial_ratio(address1, address2) / 100
        
        return {
            'cosine_sim': cosine_sim,
            'levenshtein_sim': levenshtein_sim,
            'token_sort_ratio': token_sort_ratio,
            'partial_ratio': partial_ratio
        }
    
    def _calculate_likelihood(self, metrics):
        """
        Calculate likelihood that addresses match given similarity metrics.
        This uses naive Bayes with empirically estimated probabilities.
        """
        # These probabilities should be calibrated on your specific data
        # Here we use example values - you should adjust based on your data
        
        # P(metric | match)
        p_cosine_match = np.where(metrics['cosine_sim'] > 0.8, 0.9, 
                                np.where(metrics['cosine_sim'] > 0.6, 0.7, 0.1))
        p_lev_match = np.where(metrics['levenshtein_sim'] > 0.8, 0.95, 
                             np.where(metrics['levenshtein_sim'] > 0.6, 0.75, 0.15))
        p_token_match = np.where(metrics['token_sort_ratio'] > 0.85, 0.92, 
                               np.where(metrics['token_sort_ratio'] > 0.7, 0.8, 0.2))
        p_partial_match = np.where(metrics['partial_ratio'] > 0.9, 0.96, 
                                  np.where(metrics['partial_ratio'] > 0.7, 0.85, 0.25))
        
        # P(metric | no match)
        p_cosine_nomatch = np.where(metrics['cosine_sim'] > 0.8, 0.05, 
                                  np.where(metrics['cosine_sim'] > 0.6, 0.2, 0.9))
        p_lev_nomatch = np.where(metrics['levenshtein_sim'] > 0.8, 0.02, 
                               np.where(metrics['levenshtein_sim'] > 0.6, 0.15, 0.85))
        p_token_nomatch = np.where(metrics['token_sort_ratio'] > 0.85, 0.03, 
                                 np.where(metrics['token_sort_ratio'] > 0.7, 0.25, 0.8))
        p_partial_nomatch = np.where(metrics['partial_ratio'] > 0.9, 0.01, 
                                   np.where(metrics['partial_ratio'] > 0.7, 0.2, 0.75))
        
        # Calculate likelihood ratio (naive Bayes assumption)
        likelihood_ratio = (
            (p_cosine_match * p_lev_match * p_token_match * p_partial_match) / 
            (p_cosine_nomatch * p_lev_nomatch * p_token_nomatch * p_partial_nomatch)
        )
        
        return likelihood_ratio
    
    def calculate_match_probability(self, address1, address2):
        """
        Calculate Bayesian probability that two addresses match.
        
        Args:
            address1 (str): Address from first dataset
            address2 (str): Address from second dataset
            
        Returns:
            float: Probability [0, 1] that the addresses match
        """
        # Preprocess addresses
        processed1 = self._preprocess_address(address1)
        processed2 = self._preprocess_address(address2)
        
        # Calculate similarity metrics
        metrics = self._calculate_similarity_metrics(processed1, processed2)
        
        # Calculate likelihood ratio
        likelihood_ratio = self._calculate_likelihood(metrics)
        
        # Apply Bayes' theorem
        posterior_prob = (likelihood_ratio * self.prior_prob) / \
                        (likelihood_ratio * self.prior_prob + (1 - self.prior_prob))
        
        return posterior_prob
    
    def compare_all_addresses(self, threshold=0.8):
        """
        Compare all addresses between the two dataframes and return potential matches.
        
        Args:
            threshold (float): Probability threshold to consider a match
            
        Returns:
            pd.DataFrame: DataFrame of potential matches with probabilities
        """
        matches = []
        
        # This can be optimized for large datasets (e.g., with blocking or indexing)
        for idx1, row1 in self.df1.iterrows():
            for idx2, row2 in self.df2.iterrows():
                prob = self.calculate_match_probability(
                    row1[self.address_col1], 
                    row2[self.address_col2]
                )
                
                if prob >= threshold:
                    matches.append({
                        'df1_index': idx1,
                        'df2_index': idx2,
                        'address1': row1[self.address_col1],
                        'address2': row2[self.address_col2],
                        'match_probability': prob,
                        **self._calculate_similarity_metrics(
                            row1['processed_address'],
                            row2['processed_address']
                        )
                    })
        
        return pd.DataFrame(matches).sort_values('match_probability', ascending=False)

# Example usage
if __name__ == "__main__":
    # Example data
    data1 = {
        'id': [1, 2, 3, 4],
        'address': [
            "123 Main St, Apt 4B, Springfield",
            "456 Oak Avenue, Springfield, IL 62704",
            "789 Pine Rd, Unit 12, Boston, MA",
            "321 Elm Street, Chicago, IL"
        ]
    }
    
    data2 = {
        'id': [101, 102, 103, 104],
        'address': [
            "123 Main Street, Apartment 4B, Springfield",
            "456 Oak Ave, Springfield IL 62704",
            "999 Pine Road, Boston Massachusetts",
            "321 Elm St, Chicago Illinois"
        ]
    }
    
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    # Initialize comparator
    comparator = AddressComparator(df1, df2)
    
    # Compare all addresses
    matches = comparator.compare_all_addresses(threshold=0.7)
    
    print("Potential address matches:")
    print(matches[['address1', 'address2', 'match_probability']])
