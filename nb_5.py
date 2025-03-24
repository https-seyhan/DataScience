import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from scipy import stats

# Manually load Quine dataset (since get_rdataset fails in some environments)
data = {
    'Eth': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
    'Sex': ['F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'M'],
    'Age': ['F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0'],
    'Lrn': ['AL', 'AL', 'SL', 'SL', 'AL', 'SL', 'AL', 'SL', 'AL', 'SL'],
    'Days': [2, 11, 14, 5, 12, 32, 20, 15, 8, 4]
}
df = pd.DataFrame(data)

# Fit Negative Binomial Model
nb_model = smf.glm(formula='Days ~ Age + Sex + Eth + Lrn',
                   data=df,
                   family=sm.families.NegativeBinomial()).fit()

# Fit Poisson Model
poisson_model = smf.glm(formula='Days ~ Age + Sex + Eth + Lrn',
                        data=df,
                        family=sm.families.Poisson()).fit()

# AIC Comparison
print(f"Poisson AIC: {poisson_model.aic:.2f}")
print(f"Negative Binomial AIC: {nb_model.aic:.2f}")

# Likelihood Ratio Test
llf_nb = nb_model.llf
llf_pois = poisson_model.llf
lr_stat = 2 * (llf_nb - llf_pois)
p_value = stats.chi2.sf(lr_stat, df=1)
print(f"Likelihood Ratio Statistic: {lr_stat:.2f}, p-value: {p_value:.4f}")

# Pseudo R-squared (McFadden)
null_model = smf.glm(formula='Days ~ 1', data=df, family=sm.families.NegativeBinomial()).fit()
pseudo_r2 = 1 - nb_model.deviance / null_model.deviance
print(f"McFadden’s Pseudo R²: {pseudo_r2:.4f}")

# Prediction Metrics
predicted_counts = nb_model.predict(df)
actual_counts = df['Days']
rmse = np.sqrt(mean_squared_error(actual_counts, predicted_counts))
mae = mean_absolute_error(actual_counts, predicted_counts)
correlation = np.corrcoef(actual_counts, predicted_counts)[0, 1]

print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, Correlation: {correlation:.2f}")

# Cross-Validation RMSE
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_list = []

for train_idx, test_idx in kf.split(df):
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    cv_model = smf.glm(formula='Days ~ Age + Sex + Eth + Lrn',
                       data=train_data,
                       family=sm.families.NegativeBinomial()).fit()
    preds = cv_model.predict(test_data)
    actuals = test_data['Days']
    rmse_cv = np.sqrt(mean_squared_error(actuals, preds))
    rmse_list.append(rmse_cv)

print(f"Cross-Validated RMSEs: {rmse_list}")
print(f"Average Cross-Validated RMSE: {np.mean(rmse_list):.2f}")

# Plot Actual vs Predicted
plt.scatter(actual_counts, predicted_counts, alpha=0.6, color='teal')
plt.plot([0, max(actual_counts)], [0, max(predicted_counts)], 'r--')
plt.xlabel('Actual Days')
plt.ylabel('Predicted Days')
plt.title('Negative Binomial: Actual vs Predicted')
plt.grid(True)
plt.show()

# Residual Histogram
residuals = actual_counts - predicted_counts
plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
