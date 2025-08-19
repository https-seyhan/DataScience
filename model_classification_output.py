import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -------------------------
# 1. Synthetic 8-class data
# -------------------------
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
    n_classes=8, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Add ID column for validation rows
ids = np.arange(len(y_val))

# -------------------------
# 2. Random Forest
# -------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_val)
rf_probs = rf.predict_proba(X_val)

# -------------------------
# 3. XGBoost
# -------------------------
xgb_model = xgb.XGBClassifier(
    objective="multi:softprob", num_class=8,
    n_estimators=200, random_state=42, use_label_encoder=False, eval_metric="mlogloss"
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_val)
xgb_probs = xgb_model.predict_proba(X_val)

# -------------------------
# 4. LightGBM
# -------------------------
lgb_model = lgb.LGBMClassifier(
    objective="multiclass", num_class=8,
    n_estimators=200, random_state=42
)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_val)
lgb_probs = lgb_model.predict_proba(X_val)

# -------------------------
# 5. Neural Network (Keras)
# -------------------------
y_train_nn = to_categorical(y_train, num_classes=8)
y_val_nn = to_categorical(y_val, num_classes=8)

nn = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(8, activation="softmax")
])
nn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
nn.fit(X_train, y_train_nn, epochs=20, batch_size=64, verbose=0)

nn_probs = nn.predict(X_val)
nn_preds = np.argmax(nn_probs, axis=1)

# -------------------------
# 6. Build result tables
# -------------------------
def build_results(model_name, ids, true_labels, pred_labels, probs):
    class_cols = [f"{model_name}_Class_{i}" for i in range(probs.shape[1])]
    df = pd.DataFrame(probs, columns=class_cols)
    df.insert(0, f"{model_name}_Predicted", pred_labels)
    df.insert(0, "True_Label", true_labels)
    df.insert(0, "ID", ids)
    return df

rf_df = build_results("RF", ids, y_val, rf_preds, rf_probs)
xgb_df = build_results("XGB", ids, y_val, xgb_preds, xgb_probs)
lgb_df = build_results("LGB", ids, y_val, lgb_preds, lgb_probs)
nn_df  = build_results("NN", ids, y_val, nn_preds, nn_probs)

# -------------------------
# 7. Merge all results
# -------------------------
# Merge on ID + True_Label (shared across all models)
df_results = rf_df.merge(xgb_df.drop(columns=["True_Label"]), on="ID")
df_results = df_results.merge(lgb_df.drop(columns=["True_Label"]), on="ID")
df_results = df_results.merge(nn_df.drop(columns=["True_Label"]), on="ID")

# -------------------------
# 8. Save & Preview
# -------------------------
print(df_results.head(10))  # preview first 10 rows
df_results.to_csv("multi_model_classification_results.csv", index=False)
Example Output (simplified)
ID	True_Label	RF_Predicted	RF_Class_0	…	XGB_Predicted	XGB_Class_0	…	LGB_Predicted	…	NN_Predicted	NN_Class_0	…
0	3	3	0.01	…	3	0.02	…	3	…	3	0.01	…
1	5	4	0.10	…	4	0.15	…	5	…	5	0.20	…
2	7	7	0.00	…	7	0.05	…	7	…	7	0.70	…
