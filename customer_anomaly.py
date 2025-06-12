import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

What This Does:

    Calculates Cost per Customer and Customer Growth %.

    Uses Isolation Forest to detect anomalies.

    Plots the results and flags outlier weeks in red.

    Helps identify whether operational or financial behavior was abnormal.

Let me know if you'd like to expand this with:

    Multi-feature anomaly detection (e.g., churn, complaints).

    A time-series model (like Prophet).

    Export to Excel with color-coded flags.

# -----------------------------
#  Data: Replace with your own
# -----------------------------
data = {
    "Week": [1, 2, 3, 4, 5],
    "Customers": [1000, 1200, 1440, 1728, 1900],
    "Service_Payments": [10000, 13200, 16500, 20700, 24700]
}
df = pd.DataFrame(data)

# -----------------------------
# Feature Engineering
# -----------------------------
df["Customer_Growth_%"] = df["Customers"].pct_change() * 100
df["Cost_per_Customer"] = df["Service_Payments"] / df["Customers"]

# Drop first row with NaNs
features = df[["Cost_per_Customer", "Customer_Growth_%"]].dropna()

# -----------------------------
# Isolation Forest
# -----------------------------
iso_forest = IsolationForest(contamination=0.2, random_state=42)
outliers = iso_forest.fit_predict(features)

# Map results
df.loc[features.index, "Outlier"] = outliers
df["Outlier_Label"] = df["Outlier"].map({1: "Normal", -1: "Anomaly"})

# -----------------------------
# Plotting
# -----------------------------
colors = df["Outlier"].map({1: "green", -1: "red"})

plt.figure(figsize=(10, 6))
plt.scatter(df["Week"], df["Cost_per_Customer"], c=colors, s=100)
plt.xlabel("Week")
plt.ylabel("Cost per Customer")
plt.title("Outlier Detection using Isolation Forest")
for i, label in enumerate(df["Outlier_Label"]):
    if label == "Anomaly":
        plt.annotate(label, (df["Week"][i], df["Cost_per_Customer"][i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Output
# -----------------------------
print(df[["Week", "Customers", "Cost_per_Customer", "Customer_Growth_%", "Outlier_Label"]])
