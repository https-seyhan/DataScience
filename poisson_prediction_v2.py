import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Sample dataset
data = pd.DataFrame({
    'Deaths': [10, 15, 25, 30, 50, 65, 80, 90, 120, 150],  # Target variable
    'Population': [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000],  # Numerical
    'Avg_Age': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75],  # Numerical
    'Region': ['Urban', 'Urban', 'Rural', 'Rural', 'Urban', 'Urban', 'Rural', 'Rural', 'Urban', 'Urban'],  # Categorical
    'Cause_of_Death': ['Heart Attack', 'Stroke', 'Accident', 'Heart Attack', 'Cancer',
                       'Stroke', 'Accident', 'Cancer', 'Heart Attack', 'Stroke']  # Text
})

# Convert categorical variables into dummy variables
data = pd.get_dummies(data, columns=['Region'], drop_first=True)

# Convert text variable using TF-IDF
vectorizer = TfidfVectorizer()
cause_tfidf = vectorizer.fit_transform(data['Cause_of_Death']).toarray()

# Convert TF-IDF array to DataFrame
cause_tfidf_df = pd.DataFrame(cause_tfidf, columns=vectorizer.get_feature_names_out())

# Merge TF-IDF data back to main DataFrame
data = pd.concat([data.drop(columns=['Cause_of_Death']), cause_tfidf_df], axis=1)

# Standardize numerical variables
scaler = StandardScaler()
data[['Population', 'Avg_Age']] = scaler.fit_transform(data[['Population', 'Avg_Age']])

# Define Poisson Regression Model
formula = 'Deaths ~ Population + Avg_Age + Region_Urban + ' + ' + '.join(cause_tfidf_df.columns)
model = smf.glm(formula=formula, data=data, family=sm.families.Poisson()).fit()

# Model Summary
print(model.summary())

# Make Predictions
new_data = pd.DataFrame({
    'Population': [3500, 5500, 7500], 
    'Avg_Age': [48, 63, 72], 
    'Region_Urban': [1, 0, 1],  # Urban = 1, Rural = 0
    'heart attack': [1, 0, 1],  # One-hot encoded text variable (example)
    'stroke': [0, 1, 0], 
    'accident': [0, 0, 0], 
    'cancer': [0, 0, 0]
})
new_data[['Population', 'Avg_Age']] = scaler.transform(new_data[['Population', 'Avg_Age']])  # Standardize
new_data['Predicted_Deaths'] = model.predict(new_data)

# Display Predictions
print(new_data[['Population', 'Avg_Age', 'Region_Urban', 'Predicted_Deaths']])

# Check Overdispersion
deviance = model.deviance  # Residual deviance
df_resid = model.df_resid  # Degrees of freedom

print(f"Residual Deviance: {deviance}")
print(f"Degrees of Freedom: {df_resid}")
print(f"Ratio (Deviance / DF): {deviance / df_resid}")

# Rule of thumb: If Ratio >> 1, overdispersion is present

import statsmodels.api as sm

nb_model = smf.glm(formula=formula, data=data, family=sm.families.NegativeBinomial()).fit()

print(nb_model.summary())

from statsmodels.tsa.stattools import grangercausalitytests

# Assume we have a time-series dataset with Deaths and Population
time_series_data = data[['Deaths', 'Population']].copy()
time_series_data['Deaths_Lag1'] = time_series_data['Deaths'].shift(1)
time_series_data.dropna(inplace=True)

# Perform Granger Causality Test
grangercausalitytests(time_series_data, maxlag=2, verbose=True)


import statsmodels.sandbox.regression.gmm as gmm

# Assume 'Region' is an instrument (correlated with Population but not with errors in Deaths)
iv_model = sm.OLS(data['Deaths'], sm.add_constant(data[['Population', 'Avg_Age', 'Region_Urban']])).fit()

print(iv_model.summary())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

vectorizer = TfidfVectorizer()
cause_tfidf = vectorizer.fit_transform(data['Cause_of_Death'])

# Reduce dimensionality
svd = TruncatedSVD(n_components=2)
cause_embeddings = svd.fit_transform(cause_tfidf)

# Add to DataFrame
data['Cause1'] = cause_embeddings[:, 0]
data['Cause2'] = cause_embeddings[:, 1]


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X = data.drop(columns=['Deaths'])
y = data['Deaths']

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Poisson Regression
train_model = smf.glm(formula=formula, data=pd.concat([X_train, y_train], axis=1), family=sm.families.Poisson()).fit()

# Predict
y_pred = train_model.predict(X_test)

# Evaluate with MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")



