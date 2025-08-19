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
# 2. Random Forest
# -------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"Random Forest - Train Acc: {rf_train_acc:.3f}, Test Acc: {rf_test_acc:.3f}")

# -------------------------
# 3. XGBoost
# -------------------------
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=8,
                              n_estimators=200, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="mlogloss", verbose=False)
xgb_train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
xgb_test_acc = accuracy_score(y_test, xgb_model.predict(X_test))

print(f"XGBoost - Train Acc: {xgb_train_acc:.3f}, Test Acc: {xgb_test_acc:.3f}")

# -------------------------
# 4. LightGBM
# -------------------------
lgb_model = lgb.LGBMClassifier(objective="multiclass", num_class=8,
                               n_estimators=200, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=42)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="multi_logloss", verbose=-1)
lgb_train_acc = accuracy_score(y_train, lgb_model.predict(X_train))
lgb_test_acc = accuracy_score(y_test, lgb_model.predict(X_test))

print(f"LightGBM - Train Acc: {lgb_train_acc:.3f}, Test Acc: {lgb_test_acc:.3f}")

# -------------------------
# 5. Neural Network
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

nn_train_acc = history.history["accuracy"][-1]
nn_test_acc = history.history["val_accuracy"][-1]

print(f"Neural Net - Train Acc: {nn_train_acc:.3f}, Test Acc: {nn_test_acc:.3f}")

# -------------------------
# 6. Plot Learning Curves
# -------------------------
plt.figure(figsize=(12,6))
plt.plot(history.history["accuracy"], label="NN Train Acc")
plt.plot(history.history["val_accuracy"], label="NN Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Neural Net Learning Curve (Overfitting Check)")
plt.legend()
plt.show()

What this does:

Prints train vs. test accuracy for each model (overfitting check).

Plots Neural Net learning curve (train vs. val accuracy per epoch).

👉 Overfitting is visible if:

Training accuracy >> Validation accuracy.

Validation accuracy stagnates or decreases while training keeps improving.
