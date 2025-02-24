import numpy as np
import scipy.stats as stats

# Assume mean death counts (lambda) for three years
lambda_values = [50, 55, 60]  # Example average death counts per year

# Define a range of death counts to evaluate probabilities
k_values = np.arange(30, 90)  # Considering a reasonable range around the mean

# Compute Poisson probabilities for each year
poisson_probs = {f'Year {i+1} (λ={l})': stats.poisson.pmf(k_values, l) 
                 for i, l in enumerate(lambda_values)}

# Display some example probabilities
poisson_probs_summary = {year: probs[:10] for year, probs in poisson_probs.items()}  # Show first 10 values
poisson_probs_summary
