import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -------------------------
# 1. Synthetic Data
# -------------------------
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=15, n_redundant=5,
    n_classes=8, n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# For neural net
y_train_nn = to_categorical(y_train, num_classes=8)
y_test_nn = to_categorical(y_test, num_classes=8)

# -------------------------
# 2. Random Forest Learning Curve
# -------------------------
rf_train_acc, rf_test_acc = [], []
estimators = [10, 50, 100, 200, 300, 400]

for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    rf_train_acc.append(accuracy_score(y_train, rf.predict(X_train)))
    rf_test_acc.append(accuracy_score(y_test, rf.predict(X_test)))

# -------------------------
# 3. XGBoost Learning Curve
# -------------------------
xgb_train_acc, xgb_test_acc = [], []

for n in estimators:
    xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=8,
                                  n_estimators=n, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_train_acc.append(accuracy_score(y_train, xgb_model.predict(X_train)))
    xgb_test_acc.append(accuracy_score(y_test, xgb_model.predict(X_test)))

# -------------------------
# 4. LightGBM Learning Curve
# -------------------------
lgb_train_acc, lgb_test_acc = [], []

for n in estimators:
    lgb_model = lgb.LGBMClassifier(objective="multiclass", num_class=8,
                                   n_estimators=n, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_train_acc.append(accuracy_score(y_train, lgb_model.predict(X_train)))
    lgb_test_acc.append(accuracy_score(y_test, lgb_model.predict(X_test)))

# -------------------------
# 5. Neural Network Learning Curve
# -------------------------
nn = Sequential([
    Dense(64, input_dim=20, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(8, activation="softmax")
])
nn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = nn.fit(X_train, y_train_nn, validation_data=(X_test, y_test_nn),
                 epochs=30, batch_size=64, verbose=0)

# -------------------------
# 6. Plot Learning Curves
# -------------------------
plt.figure(figsize=(16,10))

# Random Forest
plt.subplot(2,2,1)
plt.plot(estimators, rf_train_acc, label="Train Acc")
plt.plot(estimators, rf_test_acc, label="Test Acc")
plt.title("Random Forest Overfitting Check")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()

# XGBoost
plt.subplot(2,2,2)
plt.plot(estimators, xgb_train_acc, label="Train Acc")
plt.plot(estimators, xgb_test_acc, label="Test Acc")
plt.title("XGBoost Overfitting Check")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()

# LightGBM
plt.subplot(2,2,3)
plt.plot(estimators, lgb_train_acc, label="Train Acc")
plt.plot(estimators, lgb_test_acc, label="Test Acc")
plt.title("LightGBM Overfitting Check")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()

# Neural Net
plt.subplot(2,2,4)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Neural Net Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

What you’ll see:

RF, XGB, LGB: Train vs Test accuracy as the number of trees increases.

NN: Train vs Validation accuracy per epoch.

👉 Overfitting pattern = training accuracy keeps climbing while test accuracy plateaus or drops.
