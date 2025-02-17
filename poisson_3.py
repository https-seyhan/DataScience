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

# Convert 'region' to categorical type
region = pd.Categorical(region)

# One-hot encode 'region' variable
region_dummies = pd.get_dummies(region, prefix='region', drop_first=True)

# Ensure region_dummies are numeric
region_dummies = region_dummies.astype(int)

# True lambda for Poisson distribution
true_lambda = np.exp(0.01 * age + 0.5 * smoker - 0.3 * gender +
                     0.2 * region_dummies.get('region_South', 0) +
                     0.4 * region_dummies.get('region_West', 0) - 4)

# Generate Poisson-distributed deaths
deaths = np.random.poisson(lam=true_lambda)

# Create DataFrame
data = pd.DataFrame({'age': age, 'smoker': smoker, 'gender': gender, 'deaths': deaths})

# Merge region_dummies into main dataframe
data = pd.concat([data, region_dummies], axis=1)

# Add intercept column
data['intercept'] = 1

# Check if all columns are numeric
print(data.dtypes)  # Debugging step

# Fit Poisson Regression model
poisson_model = sm.GLM(data['deaths'],
                        data[['intercept', 'age', 'smoker', 'gender', 'region_South', 'region_West']],
                        family=sm.families.Poisson()).fit()

# Print summary
print(poisson_model.summary())
