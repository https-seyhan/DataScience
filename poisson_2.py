import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Simulating data
np.random.seed(42)
n = 1000  # Number of observations

# Generating a continuous predictor (age)
age = np.random.randint(20, 90, size=n)

# Categorical predictor: Smoker (Yes=1, No=0)
smoker = np.random.choice([0, 1], size=n, p=[0.7, 0.3])  # 30% are smokers

# Simulating the death rate (lambda) based on age and smoking status
true_lambda = np.exp(0.01 * age + 0.5 * smoker - 4)  # Smokers have higher risk

# Generating Poisson-distributed deaths
deaths = np.random.poisson(lam=true_lambda)

# Creating a DataFrame
data = pd.DataFrame({'age': age, 'smoker': smoker, 'deaths': deaths})

# Adding a constant for the Poisson regression model
data['intercept'] = 1

# Fit a Poisson regression model with categorical predictor
poisson_model = sm.GLM(data['deaths'], data[['intercept', 'age', 'smoker']],
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
