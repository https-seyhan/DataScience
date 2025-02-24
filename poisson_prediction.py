import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Sample dataset with categorical and numerical independent variables
data = pd.DataFrame({
    'Deaths': [10, 15, 25, 30, 50, 65, 80, 90, 120, 150],  # Dependent variable (count data)
    'Population': [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000],  # Numerical
    'Avg_Age': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75],  # Numerical
    'Region': ['Urban', 'Urban', 'Rural', 'Rural', 'Urban', 'Urban', 'Rural', 'Rural', 'Urban', 'Urban']  # Categorical
})

# Encode categorical variables
data = pd.get_dummies(data, columns=['Region'], drop_first=True)  # Convert 'Region' into binary (Urban = 1, Rural = 0)

# Define the Poisson regression model
model = smf.glm(formula='Deaths ~ Population + Avg_Age + Region_Urban', 
                data=data, 
                family=sm.families.Poisson()).fit()

# Predicting death counts for new data
new_data = pd.DataFrame({
    'Population': [3500, 5500, 7500], 
    'Avg_Age': [48, 63, 72], 
    'Region_Urban': [1, 0, 1]  # Urban = 1, Rural = 0
})
new_data['Predicted_Deaths'] = model.predict(new_data)

# Display the predictions
new_data
