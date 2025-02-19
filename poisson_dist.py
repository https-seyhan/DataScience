import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import Counter

# Example data (death counts)
death_counts = np.array([3, 2, 1, 4, 2, 3, 5, 3, 2, 4, 1, 2, 3, 2, 4])

# Mean and variance
mean_count = np.mean(death_counts)
var_count = np.var(death_counts, ddof=1)  # Sample variance

print(f"Mean: {mean_count}, Variance: {var_count}")

# Check if mean ≈ variance
if np.isclose(mean_count, var_count, atol=0.1):
    print("Mean is approximately equal to variance, indicating Poisson distribution.")
else:
    print("Mean and variance differ significantly, suggesting a different distribution.")

# Chi-Square Test
observed_counts = Counter(death_counts)
total_samples = len(death_counts)
# Compute Poisson probabilities
poisson_probs = [stats.poisson.pmf(k, mean_count) for k in observed_counts.keys()]

# Scale to match total observed count
expected_counts = np.array(poisson_probs) * total_samples
expected_counts *= sum(observed_counts.values()) / sum(expected_counts)  # Ensure sums match exactly

# Perform Chi-Square test
chi2_stat, p_value = stats.chisquare(list(observed_counts.values()), expected_counts)

print(f"Chi-Square Statistic: {chi2_stat}, p-value: {p_value}")
if p_value > 0.05:
    print("Failed to reject Poisson hypothesis (p > 0.05).")
else:
    print("Rejected Poisson hypothesis (p < 0.05), suggesting another distribution.")


print(f"Chi-Square Statistic: {chi2_stat}, p-value: {p_value}")
if p_value > 0.05:
    print("Failed to reject Poisson hypothesis (p > 0.05).")
else:
    print("Rejected Poisson hypothesis (p < 0.05), suggesting another distribution.")

# Histogram with Poisson overlay
plt.hist(death_counts, bins=np.arange(0, max(death_counts)+2)-0.5, density=True, alpha=0.6, color='g', label="Observed")
x = np.arange(0, max(death_counts)+1)
plt.plot(x, stats.poisson.pmf(x, mean_count), 'bo', markersize=8, label="Poisson Fit")
plt.xlabel("Death Count")
plt.ylabel("Probability")
plt.legend()
plt.show()
