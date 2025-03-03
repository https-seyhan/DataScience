import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.count_model import NegativeBinomial
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Sample Data with Missing Values
data = pd.DataFrame({
    'TextFeature': [
        "Minor safety issue reported",
        "Risk of falling objects is high",
        "Severe flooding at the site",
        "Some safety concerns raised",
        "Potential hazard identified",
        "Low-impact site risk identified",
        "Workers reported some risk",
        "Major accident risk detected",
        "Hazardous area spotted",
        "Worksite safety moderate"
    ],
    'RiskLevel': ['Low', 'Medium', None, 'Medium', 'Medium', 'Low', None, 'High', 'High', 'Medium'],
    'Severity': ['Minor', 'Moderate', 'Severe', 'Moderate', 'Severe', 'Minor', 'Moderate', 'Severe', 'Severe', 'Moderate'],
    'Priority': ['Low', 'Normal', 'Urgent', 'Normal', 'High', 'Low', 'Normal', 'Urgent', 'High', 'Normal'],
    'DeathCount': [1, 4, 20, 3, 7, 0, 2, 19, 15, 5]  # Target variable
})

# **Step 1: Impute Missing Values**
imputer = SimpleImputer(strategy="most_frequent")
data[['RiskLevel']] = imputer.fit_transform(data[['RiskLevel']])

# **Step 2: Encode Ordered Categorical Variables**
encoders = {
    'RiskLevel': OrdinalEncoder(categories=[['Low', 'Medium', 'High']]),
    'Severity': OrdinalEncoder(categories=[['Minor', 'Moderate', 'Severe']]),
    'Priority': OrdinalEncoder(categories=[['Low', 'Normal', 'High', 'Urgent']])
}

for col, encoder in encoders.items():
    data[col] = encoder.fit_transform(data[[col]]).astype(int)

# **Step 3: Prepare Features and Target**
X = data[['RiskLevel', 'Severity', 'Priority']]
y = data['DeathCount']

# **Step 4: Fit Negative Binomial Regression**
X = sm.add_constant(X)  # Add intercept
nb_model = NegativeBinomial(y, X).fit()

# **Step 5: Predict Death Counts and Compute Probabilities**
predicted_deaths = nb_model.predict(X)
death_probabilities = predicted_deaths / np.sum(predicted_deaths)  # Normalize to get probabilities

# **Step 6: Store Results**
data['Predicted_Deaths'] = predicted_deaths
data['Death_Probability'] = death_probabilities

# **Step 7: Display Results**
print(nb_model.summary())
print(data[['TextFeature', 'RiskLevel', 'Severity', 'Priority', 'Predicted_Deaths', 'Death_Probability']])
