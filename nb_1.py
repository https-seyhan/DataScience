import pandas as pd
import statsmodels.api as sm

# Sample data
df = pd.DataFrame({
    'deaths': [3, 1, 4, 2, 0, 1],
    'age': [30, 40, 35, 50, 45, 60],
    'gender': ['M', 'F', 'F', 'M', 'F', 'M']
})

# Convert categorical variables
X = pd.get_dummies(df[['age', 'gender']], drop_first=True)
X = sm.add_constant(X)

# Target variable
y = df['deaths']

# Fit Negative Binomial regression
model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
print(model.summary())
