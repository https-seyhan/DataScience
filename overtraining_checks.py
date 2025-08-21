import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb

# 1. Create synthetic dataset
X, y = make_classification(n_samples=5000, n_features=20, n_informative=15, 
                           n_classes=2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, solver='lbfgs'),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200),
    "LightGBM": lgb.LGBMClassifier(n_estimators=200)
}

plt.figure(figsize=(15, 5))

# 3. Loop through models and plot learning curves
for i, (name, model) in enumerate(models.items(), 1):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.subplot(1, 3, i)
    plt.plot(train_sizes, train_mean, 'o-', label="Training accuracy")
    plt.plot(train_sizes, val_mean, 'o-', label="Validation accuracy")
    plt.title(name)
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
