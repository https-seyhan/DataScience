import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# 1. Create synthetic dataset
X, y = make_classification(n_samples=5000, n_features=20, n_informative=15, 
                           n_classes=2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Define models (with early stopping for boosting)
log_reg = LogisticRegression(max_iter=200, solver='lbfgs')
xgb_clf = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=200
)
lgb_clf = lgb.LGBMClassifier(n_estimators=200)

# Train XGBoost with early stopping
xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=False
)
xgb_val_acc = xgb_clf.score(X_val, y_val)

# Train LightGBM with early stopping
lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=False
)
lgb_val_acc = lgb_clf.score(X_val, y_val)

# Logistic Regression doesn't support early stopping directly
log_reg.fit(X_train, y_train)
log_val_acc = log_reg.score(X_val, y_val)

models = {
    f"Logistic Regression (val_acc={log_val_acc:.3f})": log_reg,
    f"XGBoost (best_iter={xgb_clf.best_iteration}, val_acc={xgb_val_acc:.3f})": xgb_clf,
    f"LightGBM (best_iter={lgb_clf.best_iteration_}, val_acc={lgb_val_acc:.3f})": lgb_clf
}

plt.figure(figsize=(15, 5))

# 3. Loop through models and plot learning curves
for i, (name, model) in enumerate(models.items(), 1):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
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
