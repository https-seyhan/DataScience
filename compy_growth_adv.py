import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Prepare features: Customers and Customer Growth Rate
df_features = df[["Customers", "Customer_Growth_%"]].dropna()

# Polynomial regression to model non-linear cost per customer behavior
X = df_features
y = df.loc[df_features.index, "Cost_per_Customer"]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

reg = LinearRegression().fit(X_poly, y)
df.loc[df_features.index, "Predicted_Cost_per_Customer"] = reg.predict(X_poly)

# KMeans clustering for growth vs cost pattern detection
kmeans = KMeans(n_clusters=2, random_state=42)
df_features["Cluster"] = kmeans.fit_predict(X)

# PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X)

# Plot PCA components with cluster labels
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1],
                     c=df_features["Cluster"], cmap='viridis', s=100, edgecolors='k')
ax.set_title("Customer Growth vs Cost Clustering (PCA Projection)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
plt.grid(True)
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
plt.show()

df[["Week", "Customers", "Customer_Growth_%", "Cost_per_Customer", "Predicted_Cost_per_Customer"]].round(2)
