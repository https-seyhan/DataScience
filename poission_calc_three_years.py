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


import matplotlib.pyplot as plt

# Plot the Poisson distributions for each year
plt.figure(figsize=(10, 6))

for i, (year, probs) in enumerate(poisson_probs.items()):
    plt.plot(k_values, probs, marker='o', linestyle='-', label=year)

plt.xlabel("Death Count (k)")
plt.ylabel("Probability P(X = k)")
plt.title("Poisson Distribution of Death Counts Over Three Years")
plt.legend()
plt.grid(True)
plt.show()
