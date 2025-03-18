import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n = 500
df = pd.DataFrame({
    'age': np.random.randint(20, 80, size=n),
    'gender': np.random.choice(['M', 'F'], size=n),
    'smoking_status': np.random.choice(['Non-smoker', 'Former', 'Current'], size=n)
})

# Simulate count of deaths using a Negative Binomial process
# Assume log-linear relationship
# true coefficients (for data generation)
intercept = 1.0
coef_age = 0.02
coef_gender = {'M': -0.3, 'F': 0.0}  # Reference is F
coef_smoking = {'Non-smoker': 0.0, 'Former': 0.4, 'Current': 0.7}

# Linear predictor
mu = np.exp(
    intercept
    + coef_age * df['age']
    + df['gender'].map(coef_gender)
    + df['smoking_status'].map(coef_smoking)
)

# Negative binomial dispersion parameter (alpha)
alpha = 1.5

# Simulate deaths using Negative Binomial distribution
df['deaths'] = np.random.negative_binomial(n=1/alpha, p=1 / (1 + mu * alpha))

# Preview data
print(df.head())

# ============================
# Fit Negative Binomial GLM
# ============================
model = smf.glm(
    formula='deaths ~ age + C(gender) + C(smoking_status)',
    data=df,
    family=sm.families.NegativeBinomial()
)
result = model.fit()
print(result.summary())

# ============================
# Incidence Rate Ratios (IRR)
# ============================
irr = np.exp(result.params)
print("\nIncidence Rate Ratios (IRR):")
print(irr)

# ============================
# Predict and plot results
# ============================
df['predicted_deaths'] = result.predict(df)

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x='predicted_deaths', y='deaths', data=df, alpha=0.6)
plt.plot([0, df['deaths'].max()], [0, df['deaths'].max()], 'r--')
plt.xlabel("Predicted Deaths")
plt.ylabel("Actual Deaths")
plt.title("Actual vs Predicted Deaths (Negative Binomial Regression)")
plt.tight_layout()
plt.show()

📌 Explanation of Causal Effects (Again, Approximate):
Variable	IRR Interpretation
Age	IRR > 1 → deaths increase with age (e.g., IRR 1.02 = 2% more deaths per year)
Gender (M vs F)	IRR < 1 → Males have fewer deaths than females
Smoking (Current vs Non-smoker)	IRR > 1 → Current smokers have higher death risk
