import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# 📌 Sample Dataset with Missing Ordered Categories
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
    ]
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

print(df[['TextFeature', 'RiskLevel', 'Severity', 'Priority']])
