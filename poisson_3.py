import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Simulating data
np.random.seed(42)
n = 1000  # Number of observations

# Continuous predictor (age)
age = np.random.randint(20, 90, size=n)

# Categorical predictors
smoker = np.random.choice([0, 1], size=n, p=[0.7, 0.3])  # 30% are smokers
gender = np.random.choice([0, 1], size=n, p=[0.5, 0.5])  # 50% Male, 50% Female
region = np.random.choice(["North", "South", "West"], size=n, p=[0.4, 0.3, 0.3])  # 3 Regions

# One-hot encode 'region' variable
region_dummies = pd.get_dummies(region, prefix='region', drop_first=True)

# True lambda for Poisson distribution
true_lambda = np.exp(0.01 * age + 0.5 * smoker - 0.3 * gender +
                     0.2 * region_dummies['region_South'] + 0.4 * region_dummies['region_West'] - 4)

# Generate Poisson-distributed deaths
deaths = np.random.poisson(lam=true_lambda)

# Create DataFrame
data = pd.DataFrame({'age': age, 'smoker': smoker, 'gender': gender, 'deaths': deaths})
data = pd.concat([data, region_dummies], axis=1)  # Add encoded region variables
data['intercept'] = 1  # Intercept term

# Fit Poisson Regression model
poisson_model = sm.GLM(data['deaths'],
                        data[['intercept', 'age', 'smoker', 'gender', 'region_South', 'region_West']],
                        family=sm.families.Poisson()).fit()

# Print summary
print(poisson_model.summary())

# Predict deaths
data['predicted_deaths'] = poisson_model.predict()

# Plot actual vs predicted deaths
plt.scatter(data['age'], data['deaths'], alpha=0.5, label='Actual deaths')
plt.scatter(data['age'], data['predicted_deaths'], alpha=0.5, label='Predicted deaths', color='red')
plt.xlabel('Age')
plt.ylabel('Death occurrences')
plt.legend()
plt.show()
