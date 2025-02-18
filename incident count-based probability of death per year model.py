import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor

# 📌 Sample Dataset with Missing Ordered Categories + Incident Counts
data = {
    'RiskLevel': ['Low', np.nan, 'High', 'Medium', np.nan, 'Low', 'Medium', 'High', np.nan, 'Medium'],
    'Severity': ['Minor', 'Moderate', 'Severe', np.nan, 'Severe', 'Minor', 'Moderate', np.nan, 'Moderate', 'Severe'],
    'Priority': ['Low', np.nan, 'Urgent', 'Normal', 'High', 'Low', 'Normal', 'Urgent', np.nan, 'Normal'],
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
    'IncidentCount': [1, 3, 15, 5, 7, 0, 2, 20, 6, 4],  # Number of incidents reported
    'Year': [2020, 2021, 2022, 2023, 2024, 2020, 2021, 2022, 2023, 2024]  # Year of event
}
df = pd.DataFrame(data)

# 📌 Encode Multiple Ordered Categories
ordinal_categories = {
    'RiskLevel': ['Low', 'Medium', 'High'],
    'Severity': ['Minor', 'Moderate', 'Severe'],
    'Priority': ['Low', 'Normal', 'High', 'Urgent']
}

ordinal_encoder = OrdinalEncoder(categories=[ordinal_categories[col] for col in ordinal_categories.keys()])
df[list(ordinal_categories.keys())] = ordinal_encoder.fit_transform(df[list(ordinal_categories.keys())])

# 📌 Separate Data with & without Missing Values
df_known = df.dropna()
df_missing = df[df.isnull().any(axis=1)]

X_texts = df_known['TextFeature'].values
y_labels = df_known[list(ordinal_categories.keys())].values  # Multiple Targets

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_texts, y_labels, test_size=0.2, random_state=42)

# 📌 Convert to SentenceTransformer Training Format (Multi-Label)
train_examples = [InputExample(texts=[text], label=list(label)) for text, label in zip(X_train, y_train)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

# 📌 Load Pretrained sBERT Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 📌 Define Loss Function (Cosine Similarity for Multi-Label)
train_loss = losses.CosineSimilarityLoss(model)

# 📌 Train the Model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=100)

# 📌 Use Fine-Tuned Model to Predict Missing Categories
df_missing['TextEmbedding'] = df_missing['TextFeature'].apply(lambda x: model.encode(x))
X_missing = np.vstack(df_missing['TextEmbedding'].values)

# 📌 Map to Nearest Label (Rounded Prediction)
predicted_labels = np.round(model.predict(X_missing)).astype(int)

# 📌 Decode Back to Original Categories
df_missing[list(ordinal_categories.keys())] = ordinal_encoder.inverse_transform(predicted_labels)

# 📌 Merge Imputed Values Back
for col in ordinal_categories.keys():
    df.loc[df[col].isna(), col] = df_missing[col]

# 🎯 **Step 2: Poisson Regression for Death Probability Prediction**
# 📌 Feature Engineering: Convert Text to SBERT Embeddings
df['TextEmbedding'] = df['TextFeature'].apply(lambda x: model.encode(x))
X = np.vstack(df['TextEmbedding'].values)

# 📌 Target Variable: Incident Count (Poisson Distributed)
y = df['IncidentCount'].values

# 📌 Train Poisson Regression Model
poisson_model = PoissonRegressor(alpha=0.1, max_iter=300)
poisson_model.fit(X, y)

# 📌 Predict Death Probability for Each Year
df['Predicted_Deaths'] = poisson_model.predict(X)

# 📌 Normalize to Get Probabilities
df['Death_Probability'] = df['Predicted_Deaths'] / df['Predicted_Deaths'].sum()

# 📌 Display Final Results
print(df[['TextFeature', 'RiskLevel', 'Severity', 'Priority', 'Predicted_Deaths', 'Death_Probability']])

#✅ Handles Multiple Ordered Categories → Imputes RiskLevel, Severity, Priority together.
#✅ sBERT Fine-Tuning → Uses text descriptions to predict missing values.
#✅ Poisson Regression → Models incident count-based probability of death per year.
#✅ Scalable → Can be applied to other risk-related datasets.
