import matplotlib.pyplot as plt
from scipy.stats import binom

n = 10
p = 0.5

# Create an array of possible values of k
k_values = list(range(n + 1))

# Calculate the probabilities for each k
probabilities = [binom.pmf(k, n, p) for k in k_values]

# Plot the PMF graph
plt.bar(k_values, probabilities)
plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability')
plt.title('Binomial Distribution PMF')
plt.show()