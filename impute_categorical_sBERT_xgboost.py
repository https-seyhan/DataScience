import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
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

# Load Pretrained sBERT Model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast

# Convert Text to sBERT Embeddings
df['TextEmbedding'] = df['TextFeature'].apply(lambda x: sbert_model.encode(x))

# Convert Embeddings to NumPy Array
X_embeddings = np.vstack(df['TextEmbedding'].values)

# Separate Known and Missing Data
df_known = df.dropna(subset=['OrderedCategory'])
df_missing = df[df['OrderedCategory'].isna()]

X_known = np.vstack(df_known['TextEmbedding'].values)
y_known = df_known['OrderedCategory_encoded'].ravel()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.2, random_state=42)

# Define XGBoost Model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False)

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
X_missing = np.vstack(df_missing['TextEmbedding'].values)
predicted_labels = best_model.predict(X_missing)

# Convert Predicted Encoded Labels Back to Categories
df_missing['ImputedCategory'] = ordinal_encoder.inverse_transform(predicted_labels.reshape(-1, 1))

# Merge Imputed Values Back
df.loc[df['OrderedCategory'].isna(), 'OrderedCategory'] = df_missing['ImputedCategory']
print(df[['TextFeature', 'OrderedCategory']])

#use sBERT (Sentence-BERT) instead of TF-IDF, we'll replace the text vectorization step with pre-trained sBERT embeddings.

#✅ Captures semantic meaning (understands similar phrases better).
#✅ Handles contextual differences (e.g., "Severe hazard" vs. "High risk" are close in meaning).
#✅ Works well with small datasets where TF-IDF might fail due to sparse text.
#Convert text into embeddings using sentence-transformers (all-MiniLM-L6-v2).
#Train an XGBoost model for missing value imputation.
#Predict missing values and update the dataset.
#✅ Leverages pre-trained sBERT for meaningful text embeddings.
#✅ XGBoost fine-tuned with GridSearchCV ensures optimal performance.
#✅ Works well for small datasets where deep learning fine-tuning isn't feasible.
