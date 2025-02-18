import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Sample Dataset
data = {
    'Category1': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'C', 'B', 'A'],
    'Category2': ['X', 'Y', 'X', 'Y', 'Z', 'Z', 'X', 'X', 'Y', 'Z'],
    'TextFeature': [
        "The project deadline is tight",
        "Construction schedule is delayed",
        "Meeting with stakeholders",
        "Material shortage on site",
        "New safety regulations",
        "Subcontractor issues reported",
        "Piling work completed",
        "Weather affecting work progress",
        "Site access restriction",
        "Equipment malfunction reported"
    ],
    'Target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Encode Categorical Variables
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_cats = encoder.fit_transform(df[['Category1', 'Category2']])
cat_features = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(['Category1', 'Category2']))

# Convert Text Data into Numerical Representation (TF-IDF)
vectorizer = TfidfVectorizer(max_features=10)
text_features = vectorizer.fit_transform(df['TextFeature']).toarray()
text_features_df = pd.DataFrame(text_features, columns=vectorizer.get_feature_names_out())

# Merge Features
X = pd.concat([cat_features, text_features_df], axis=1)
y = df['Target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Make Predictions and Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
