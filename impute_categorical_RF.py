import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

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
vectorizer = TfidfVectorizer(max_features=100)  # Convert text to numerical format
X = vectorizer.fit_transform(df_known['TextFeature']).toarray()
y = df_known['OrderedCategory_encoded'].ravel()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict Missing Values
X_missing = vectorizer.transform(df_missing['TextFeature']).toarray()
predicted_labels = rf_model.predict(X_missing)

# Convert Predicted Encoded Labels Back to Categories
df_missing['ImputedCategory'] = ordinal_encoder.inverse_transform(predicted_labels.reshape(-1, 1))

# Merge Imputed Values Back
df.loc[df['OrderedCategory'].isna(), 'OrderedCategory'] = df_missing['ImputedCategory']
print(df[['TextFeature', 'OrderedCategory']])
