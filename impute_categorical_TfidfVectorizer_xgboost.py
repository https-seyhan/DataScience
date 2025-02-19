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

# Handle Missing Values Before Encoding
df['OrderedCategory'].fillna('Unknown', inplace=True)

# Encode Ordered Category (Low -> Medium -> High -> Unknown)
ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Unknown']], handle_unknown='use_encoded_value', unknown_value=-1)
df['OrderedCategory_encoded'] = ordinal_encoder.fit_transform(df[['OrderedCategory']])

# Convert Text to TF-IDF Features
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['TextFeature'])

# Convert to NumPy Array
X_embeddings = X_tfidf.toarray()

# Separate Known and Missing Data
df_known = df[df['OrderedCategory'] != 'Unknown']
df_missing = df[df['OrderedCategory'] == 'Unknown']

X_known = X_embeddings[df_known.index]
y_known = df_known['OrderedCategory_encoded'].ravel()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.2, random_state=42)

# Define XGBoost Model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss', use_label_encoder=False)

# Hyperparameter Grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Grid Search for Best Hyperparameters
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best Model from Grid Search
best_model = grid_search.best_estimator_

# Predict Missing Values
X_missing = X_embeddings[df_missing.index]
predicted_labels = best_model.predict(X_missing).reshape(-1, 1)
print(predicted_labels)
print(predicted_labels.reshape(-1, 1))
# Convert Predicted Encoded Labels Back to Categories
df_missing = df_missing.copy()
df_missing['ImputedCategory'] = ordinal_encoder.inverse_transform(predicted_labels.reshape(-1, 1))

# Ensure Correct Index Alignment Before Merging
df.loc[df_missing.index, 'OrderedCategory'] = df_missing['ImputedCategory'].values
print(df[['TextFeature', 'OrderedCategory']])
