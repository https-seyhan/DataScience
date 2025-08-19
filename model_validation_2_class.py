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
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# 1. Synthetic Binary Data
# -------------------------
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=15, n_redundant=5,
    n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# For neural net (one-hot encoding)
y_train_nn = to_categorical(y_train, num_classes=2)
y_test_nn = to_categorical(y_test, num_classes=2)

# -------------------------
# 2. Random Forest
# -------------------------
rf_train_acc, rf_test_acc = [], []
estimators = [10, 50, 100, 200, 300, 400]

for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    rf_train_acc.append(accuracy_score(y_train, rf.predict(X_train)))
    rf_test_acc.append(accuracy_score(y_test, rf.predict(X_test)))

# -------------------------
# 3. XGBoost with Early Stopping
# -------------------------
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="logloss",
    n_estimators=1000, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=30,
    verbose=False
)
xgb_best_iter = xgb_model.best_iteration
xgb_train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
xgb_test_acc = accuracy_score(y_test, xgb_model.predict(X_test))
xgb_results = xgb_model.evals_result()

# -------------------------
# 4. LightGBM with Early Stopping
# -------------------------
lgb_model = lgb.LGBMClassifier(
    objective="binary", n_estimators=1000, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
)
lgb_best_iter = lgb_model.best_iteration_
lgb_train_acc = accuracy_score(y_train, lgb_model.predict(X_train))
lgb_test_acc = accuracy_score(y_test, lgb_model.predict(X_test))
lgb_results = lgb_model.evals_result_

# -------------------------
# 5. Neural Network with Early Stopping
# -------------------------
nn = Sequential([
    Dense(64, input_dim=20, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(2, activation="softmax")
])
nn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = nn.fit(
    X_train, y_train_nn,
    validation_data=(X_test, y_test_nn),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=0
)

nn_train_acc = history.history["accuracy"][-1]
nn_val_acc = history.history["val_accuracy"][-1]

# -------------------------
# 6. Results
# -------------------------
print("Random Forest:")
for n, tr, te in zip(estimators, rf_train_acc, rf_test_acc):
    print(f"  n_estimators={n}: Train={tr:.3f}, Test={te:.3f}")

print(f"\nXGBoost: Best Iter={xgb_best_iter}, Train={xgb_train_acc:.3f}, Test={xgb_test_acc:.3f}")
print(f"LightGBM: Best Iter={lgb_best_iter}, Train={lgb_train_acc:.3f}, Test={lgb_test_acc:.3f}")
print(f"Neural Net: Train={nn_train_acc:.3f}, Val={nn_val_acc:.3f} (stopped at {len(history.history['loss'])} epochs)")

# -------------------------
# 7. Plot Learning Curves
# -------------------------
plt.figure(figsize=(14,8))

# Random Forest
plt.subplot(2,2,1)
plt.plot(estimators, rf_train_acc, label="Train Acc")
plt.plot(estimators, rf_test_acc, label="Test Acc")
plt.title("Random Forest Overfitting Check")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()

# XGBoost logloss curve
plt.subplot(2,2,2)
plt.plot(xgb_results["validation_0"]["logloss"], label="Train Logloss")
plt.plot(xgb_results["validation_1"]["logloss"], label="Test Logloss")
plt.axvline(xgb_best_iter, color="red", linestyle="--", label="Best Iter")
plt.title("XGBoost Learning Curve (Early Stopping)")
plt.xlabel("Iteration")
plt.ylabel("Logloss")
plt.legend()

# LightGBM logloss curve
plt.subplot(2,2,3)
plt.plot(lgb_results["training"]["binary_logloss"], label="Train Logloss")
plt.plot(lgb_results["valid_1"]["binary_logloss"], label="Test Logloss")
plt.axvline(lgb_best_iter, color="red", linestyle="--", label="Best Iter")
plt.title("LightGBM Learning Curve (Early Stopping)")
plt.xlabel("Iteration")
plt.ylabel("Logloss")
plt.legend()

# Neural Net accuracy curve
plt.subplot(2,2,4)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Neural Net Learning Curve (Early Stopping)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
What this does

RandomForest: plots train vs test accuracy as n_estimators increases.

XGBoost / LightGBM: train up to 1000 trees but stop early if validation loss doesn’t improve.

Neural Net: uses EarlyStopping to prevent overfitting, plotting train vs validation accuracy.

Plots: Show where overfitting starts (when train improves but validation/test stagnates or worsens).
