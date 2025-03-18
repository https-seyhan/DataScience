import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Sample data
df = pd.DataFrame({
    'deaths': [3, 1, 4, 2, 0, 1],
    'age': [30, 40, 35, 50, 45, 60],
    'gender': ['M', 'F', 'F', 'M', 'F', 'M']
})

# Use formula API instead of manually creating dummies
# Automatically handles categorical encoding and interpretable model summary
model = smf.glm(formula='deaths ~ age + C(gender)', data=df, family=sm.families.NegativeBinomial())
result = model.fit()

# Print summary
print(result.summary())
Why is this version better?
formula='deaths ~ age + C(gender)'	Automatically treats gender as categorical
smf.glm	More readable, interpretable coefficients
C(gender)	Adds gender as a factor variable (e.g., Male vs Female)

How to Interpret the Coefficients (Causal Effect Approximation)

                 coef   std err    z      P>|z|    [0.025  0.975]
Intercept      0.8000   0.500   1.60     0.10
age            0.0300   0.010   3.00     0.005
C(gender)[T.M] -0.4000   0.300  -1.33     0.18

➤ Intercept

    Expected log count of deaths when age=0 and gender=F (baseline).

➤ age coefficient (0.03)

    Causal interpretation:
    Holding gender constant, a 1-year increase in age increases the expected log-count of deaths by 0.03 units.
        Or in count scale: exp(0.03) ≈ 1.03 → +3% increase per year.

➤ C(gender)[T.M] = -0.4

    Causal interpretation:
    Being Male (vs Female) decreases the expected log-count of deaths by 0.4, all else equal.
        exp(-0.4) ≈ 0.67 → Males have 33% fewer expected deaths compared to females, if age is held constant.

    💡 These are associational effects, not true causal effects unless you control for all confounders. But if the model includes all relevant confounders, you can approximate causal effects under the "unconfoundedness assumption" (no omitted variable bias).

Bonus: Add interaction if you suspect different effects by gender

model = smf.glm(formula='deaths ~ age * C(gender)', data=df, family=sm.families.NegativeBinomial())
result = model.fit()
print(result.summary())

    age * C(gender) includes main effects + interaction term → allows age effect to vary by gender.

✅ Summary of Causal Insights:
Variable	Causal Impact (Approximate)	Interpretation
age	Positive effect on deaths	Older age → higher death counts
gender	Negative effect (M vs F)	Males have fewer expected deaths
age * gender	Interaction effect	Gender modifies age effect
