import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader

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
cat_features = torch.tensor(encoded_cats, dtype=torch.float32)

# Tokenizer & Model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to Tokenize Text
def encode_texts(texts, tokenizer, max_length=20):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return encoded['input_ids'], encoded['attention_mask']

# Encode Text Data
input_ids, attention_masks = encode_texts(df['TextFeature'].tolist(), tokenizer)

# Define Dataset Class
class TextDataset(Dataset):
    def __init__(self, input_ids, attention_masks, cat_features, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.cat_features = cat_features
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.cat_features[idx], self.labels[idx]

# Train/Test Split
X_train_ids, X_test_ids, X_train_masks, X_test_masks, X_train_cats, X_test_cats, y_train, y_test = train_test_split(
    input_ids, attention_masks, cat_features, df['Target'], test_size=0.2, random_state=42
)

# Create Datasets and DataLoaders
train_dataset = TextDataset(X_train_ids, X_train_masks, X_train_cats, y_train.tolist())
test_dataset = TextDataset(X_test_ids, X_test_masks, X_test_cats, y_test.tolist())

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# Define Model
class BERTClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.fc1 = nn.Linear(768 + 5, 128)  # 768 from BERT + categorical features
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, cat_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        combined_features = torch.cat((bert_output, cat_features), dim=1)
        x = torch.relu(self.fc1(combined_features))
        x = self.sigmoid(self.fc2(x))
        return x

# Instantiate Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model).to(device)

# Define Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training Loop
for epoch in range(3):  # Only 3 epochs for demonstration
    for input_ids, attention_mask, cat_features, labels in train_loader:
        input_ids, attention_mask, cat_features, labels = input_ids.to(device), attention_mask.to(device), cat_features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, cat_features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Training Complete!")
