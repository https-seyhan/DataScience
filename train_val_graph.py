import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# 1. Generate synthetic dataset
X, y = make_classification(n_samples=3000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                  random_state=42, stratify=y)

# 2. Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
}

results = {}

# 3. Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    results[name] = {"Train": train_acc, "Validation": val_acc}
    print(f"{name}: Train Acc = {train_acc:.4f}, Validation Acc = {val_acc:.4f}")

# 4. Plot results
labels = list(results.keys())
train_scores = [results[m]["Train"] for m in labels]
val_scores = [results[m]["Validation"] for m in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, train_scores, width, label='Train Accuracy')
ax.bar(x + width/2, val_scores, width, label='Validation Accuracy')

ax.set_ylabel('Accuracy')
ax.set_title('Training vs Validation Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0.5, 1.05)
plt.show()
