import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from tensorflow import keras

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

# Split into Known and Missing Data
df_known = df.dropna(subset=['OrderedCategory'])  # Rows where OrderedCategory is known
df_missing = df[df['OrderedCategory'].isna()]    # Rows where OrderedCategory is missing

# BERT Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_texts(texts, tokenizer, max_length=30):
    encodings = tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="tf")
    return encodings

# Process Text Data
text_encodings = encode_texts(df_known['TextFeature'], tokenizer)

# Load Pretrained BERT Model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Extract Text Embeddings
bert_embeddings = bert_model(text_encodings['input_ids'], attention_mask=text_encodings['attention_mask'])[1].numpy()

# Prepare Labels
y = df_known['OrderedCategory_encoded'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(bert_embeddings, y, test_size=0.2, random_state=42)

# Define Neural Network for Classification
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(bert_embeddings.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 classes: Low, Medium, High
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# Predict Missing Values
missing_text_encodings = encode_texts(df_missing['TextFeature'], tokenizer)
missing_bert_embeddings = bert_model(missing_text_encodings['input_ids'], attention_mask=missing_text_encodings['attention_mask'])[1].numpy()

# Predict Categories
predicted_categories = model.predict(missing_bert_embeddings)
predicted_labels = np.argmax(predicted_categories, axis=1)

# Convert Encoded Predictions to Labels
df_missing['ImputedCategory'] = ordinal_encoder.inverse_transform(predicted_labels.reshape(-1, 1))

# Merge Imputed Values
df.loc[df['OrderedCategory'].isna(), 'OrderedCategory'] = df_missing['ImputedCategory']
print(df[['TextFeature', 'OrderedCategory']])
