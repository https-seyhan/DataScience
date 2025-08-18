import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. Generate synthetic dataset
X, y = make_classification(n_samples=3000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature names (optional, otherwise use f0, f1, ...)
feature_names = [f"Feature {i}" for i in range(X.shape[1])]

# 2. Train models
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="logloss", random_state=42).fit(X_train, y_train)
lgb = LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42).fit(X_train, y_train)

# 3. Extract feature importances
importances = {
    "Random Forest": rf.feature_importances_,
    "XGBoost": xgb.feature_importances_,
    "LightGBM": lgb.feature_importances_
}

# 4. Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, (name, imp) in zip(axes, importances.items()):
    sorted_idx = np.argsort(imp)[::-1][:10]   # top 10
    ax.barh(np.array(feature_names)[sorted_idx][::-1], imp[sorted_idx][::-1])
    ax.set_title(f"{name} Top 10 Features")
    ax.set_xlabel("Importance")

plt.tight_layout()
plt.show()
