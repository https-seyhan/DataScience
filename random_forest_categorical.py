import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Create Sample Dataset
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

# Step 2: Encode Categorical Variables (One-Hot Encoding)
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_cats = encoder.fit_transform(df[['Category1', 'Category2']])
cat_features = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(['Category1', 'Category2']))

# Step 3: Convert Text Data into Numerical Representation (TF-IDF)
vectorizer = TfidfVectorizer(max_features=10)
text_features = vectorizer.fit_transform(df['TextFeature']).toarray()
text_features_df = pd.DataFrame(text_features, columns=vectorizer.get_feature_names_out())

# Step 4: Merge Features
X = pd.concat([cat_features, text_features_df], axis=1)
y = df['Target']

# Step 5: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make Predictions and Evaluate Model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#What This Code Does

#    Loads categorical and text data.
#    Encodes categorical variables using One-Hot Encoding.
#    Converts text data into numerical vectors using TF-IDF.
#    Merges categorical and text features into one dataset.
#    Splits the dataset into training and test sets.
#    Trains a Random Forest Classifier on the transformed data.
#    Evaluates the model using a classification report.
