from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Select features for clustering
features = df_clustering[["Customer_Growth_%", "Service_Payment_Growth_%", "Cost_per_Customer", "Elasticity"]]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans with 2 clusters (small dataset, so we keep it simple)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df_clustering["Cluster"] = kmeans.fit_predict(scaled_features)

# Plot the clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Customer_Growth_%", 
    y="Service_Payment_Growth_%", 
    hue="Cluster", 
    palette="Set1", 
    data=df_clustering,
    s=100
)
plt.title("Customer vs. Payment Growth Clustering")
plt.xlabel("Customer Growth (%)")
plt.ylabel("Service Payment Growth (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

STDOUT/STDERR

/home/sandbox/.local/lib/python3.11/site-packages/seaborn/relational.py:658: UserWarning: You passed a edgecolor/edgecolors ('w') for an unfilled marker ('x').  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.
  points = ax.scatter(*args, **kws)

Output image

Here's your cluster interpretation plot:

    Each point = one week.

    Color = cluster group (based on service growth behavior).

    Axes = weekly % change in customers vs. service payments.

🔍 Interpretation:

    Cluster 0 and Cluster 1 separate weeks with different scaling behavior.

    Weeks in the top-right are fast-growing in both customers and cost — possibly expensive scaling.

    Weeks closer to the diagonal show balanced scaling.

