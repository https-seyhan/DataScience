import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

# Sample Dataset with Missing Values
data = {
    'OrderedCategory': ['Low', np.nan, 'High', 'Medium', np.nan, 'Low', 'Medium', 'High', np.nan, 'Medium'],
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
    ]
}

df = pd.DataFrame(data)

# Encode Ordered Category (Low -> Medium -> High)
ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['OrderedCategory_encoded'] = ordinal_encoder.fit_transform(df[['OrderedCategory']])

# Split Data into Known and Missing
df_known = df.dropna(subset=['OrderedCategory'])
df_missing = df[df['OrderedCategory'].isna()]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')  # Remove common stop words
X = vectorizer.fit_transform(df_known['TextFeature']).toarray()
y = df_known['OrderedCategory_encoded'].ravel()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost Classifier
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False)

# Hyperparameter Grid
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [3, 5, 7],  # Tree depth
    'learning_rate': [0.01, 0.1, 0.2],  # Step size
    'subsample': [0.8, 1.0]  # % of data used per tree
}

# Grid Search for Best Hyperparameters
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best Model from Grid Search
best_model = grid_search.best_estimator_

# Predict Missing Values
X_missing = vectorizer.transform(df_missing['TextFeature']).toarray()
predicted_labels = best_model.predict(X_missing)

# Convert Predicted Encoded Labels Back to Categories
df_missing['ImputedCategory'] = ordinal_encoder.inverse_transform(predicted_labels.reshape(-1, 1))

# Merge Imputed Values Back
df.loc[df['OrderedCategory'].isna(), 'OrderedCategory'] = df_missing['ImputedCategory']
print(df[['TextFeature', 'OrderedCategory']])

# Feature Importance (Top Words)
feature_importance = best_model.feature_importances_
important_words = np.array(vectorizer.get_feature_names_out())[np.argsort(feature_importance)[::-1][:10]]
print("Top 10 Important Words:", important_words)

#✅ Handles imbalanced data better (if certain risk levels appear less frequently).
#✅ Faster training and prediction (especially for large datasets).
#✅ Feature importance analysis to understand which words impact the prediction most.

#✅ Hyperparameter tuning ensures the best model is selected.
#✅ XGBoost handles imbalanced categories well.
#✅ Feature importance analysis shows which words drive predictions.
#✅ Lightweight, scalable, and fast for deployment.
