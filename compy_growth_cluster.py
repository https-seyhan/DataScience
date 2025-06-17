Clustering customer-service behavior will help reveal distinct growth-efficiency patterns across weeks — especially useful to identify:

    📈 Efficient growth weeks

    🧨 Inefficient or costly growth

    🚨 Risky fast-scaling outliers (even before they become anomalies)

✅ Step-by-Step Plan for Clustering

We'll cluster weeks based on service efficiency indicators:
📊 Features for Clustering:
Feature	Description
Customer_Growth_%	Weekly customer growth rate
Service_Payment_Growth_%	Weekly service cost growth
Cost_per_Customer	Cost-efficiency metric
Elasticity	Cost responsiveness to customer growth
Anomaly_Score	From Isolation Forest (optional weight)
⚙️ Technique:

We’ll use K-Means clustering with:

    Feature scaling (StandardScaler)

    Elbow Method to choose optimal number of clusters (k)

    Cluster interpretation via summary statistics and plots

▶️ Let's Run the Clustering (next cell will execute it)

Always show details
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Recalculate the derived features
df["Customer_Growth_%"] = df["Customers"].pct_change() * 100
df["Service_Payment_Growth_%"] = df["Service_Payments"].pct_change() * 100
df["Cost_per_Customer"] = df["Service_Payments"] / df["Customers"]
df["Elasticity"] = df["Service_Payment_Growth_%"] / df["Customer_Growth_%"]

# Replace inf and NaN (from division by zero or first row) with 0
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Select features for clustering
features = df[["Customer_Growth_%", "Service_Payment_Growth_%", "Cost_per_Customer", "Elasticity"]]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine optimal number of clusters using elbow method
inertia = []
k_range = range(1, 6)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(6, 4))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method: Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.show()
