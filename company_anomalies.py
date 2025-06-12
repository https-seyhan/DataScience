from sklearn.ensemble import IsolationForest

# Prepare feature set for anomaly detection
features = df[["Customers", "Customer_Growth_%", "Cost_per_Customer"]].dropna()

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.2, random_state=42)
features["Anomaly"] = iso_forest.fit_predict(features)

# Merge back with original DataFrame
df.loc[features.index, "Anomaly"] = features["Anomaly"]

# Mark anomalies
df["Anomaly_Label"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

# Plot results
fig, ax = plt.subplots(figsize=(10, 6))
colors = df["Anomaly_Label"].map({"Normal": "green", "Anomaly": "red"})
ax.scatter(df["Week"], df["Cost_per_Customer"], c=colors, s=100, edgecolor='k')
ax.set_title("Outlier Detection: Cost per Customer Over Time")
ax.set_xlabel("Week")
ax.set_ylabel("Cost per Customer ($)")
for i, row in df.iterrows():
    if row["Anomaly_Label"] == "Anomaly":
        ax.text(row["Week"] + 0.1, row["Cost_per_Customer"], f"Week {int(row['Week'])}", fontsize=9)

plt.grid(True)
plt.tight_layout()
plt.show()

df[["Week", "Customers", "Customer_Growth_%", "Cost_per_Customer", "Anomaly_Label"]].round(2)
