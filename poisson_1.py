import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Simulating data
np.random.seed(42)
n = 1000  # Number of observations

# Generating a feature (e.g., age or exposure time)
age = np.random.randint(20, 90, size=n)

# Simulating the rate of death occurrences (Lambda) based on age
true_lambda = np.exp(0.01 * age - 4)  # Small effect, rare occurrences
deaths = np.random.poisson(lam=true_lambda)  # Poisson distributed deaths

# Creating a DataFrame
data = pd.DataFrame({'age': age, 'deaths': deaths})

# Adding a constant for the Poisson regression model
data['intercept'] = 1

# Fit a Poisson regression model
poisson_model = sm.GLM(data['deaths'], data[['intercept', 'age']],
                        family=sm.families.Poisson()).fit()

# Print the summary
print(poisson_model.summary())

# Predict the expected death occurrences
data['predicted_deaths'] = poisson_model.predict()

# Visualizing actual vs predicted deaths
plt.scatter(data['age'], data['deaths'], alpha=0.5, label='Actual deaths')
plt.scatter(data['age'], data['predicted_deaths'], alpha=0.5, label='Predicted deaths', color='red')
plt.xlabel('Age')
plt.ylabel('Death occurrences')
plt.legend()
plt.show()
