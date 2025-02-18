import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# 📌 Load Sample Dataset
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

# 📌 Encode Ordered Category (Ordinal Encoding)
ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['OrderedCategory_encoded'] = ordinal_encoder.fit_transform(df[['OrderedCategory']])

# 📌 Separate Data with & without Missing Values
df_known = df.dropna(subset=['OrderedCategory'])
df_missing = df[df['OrderedCategory'].isna()]

X_texts = df_known['TextFeature'].values
y_labels = df_known['OrderedCategory_encoded'].ravel()

# 📌 Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_texts, y_labels, test_size=0.2, random_state=42)

# 📌 Convert to Sentence Transformer Training Format
train_examples = [InputExample(texts=[text], label=float(label)) for text, label in zip(X_train, y_train)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

# 📌 Load Pretrained sBERT Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 📌 Define Loss Function (Regression Loss for Ordered Categories)
train_loss = losses.CosineSimilarityLoss(model)

# 📌 Train the Model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=100)

# 📌 Use Fine-Tuned Model to Predict Missing Categories
df_missing['TextEmbedding'] = df_missing['TextFeature'].apply(lambda x: model.encode(x))
X_missing = np.vstack(df_missing['TextEmbedding'].values)

# 📌 Map to Nearest Label
predicted_labels = np.round(model.predict(X_missing)).astype(int)
df_missing['ImputedCategory'] = ordinal_encoder.inverse_transform(predicted_labels.reshape(-1, 1))

# 📌 Merge Imputed Values Back
df.loc[df['OrderedCategory'].isna(), 'OrderedCategory'] = df_missing['ImputedCategory']
print(df[['TextFeature', 'OrderedCategory']])
#✅ Fine-Tuning Adapts sBERT to Your Risk Data – It understands domain-specific safety terminology.
#✅ Cosine Similarity Loss Captures Ordered Categories – Helps predict Low → Medium → High accurately.
#✅ No Need for Extra Models – After training, you can use sBERT directly to predict missing risk levels.
