# Second plot: Elasticity vs. Cost per Customer
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Elasticity", 
    y="Cost_per_Customer", 
    hue="Cluster", 
    palette="Set1", 
    data=df_clustering,
    s=100
)
plt.title("Elasticity vs. Cost per Customer by Cluster")
plt.xlabel("Elasticity")
plt.ylabel("Cost per Customer")
plt.grid(True)
plt.tight_layout()
plt.show()

# Descriptive summary of each cluster
cluster_summary = df_clustering.groupby("Cluster")[["Customer_Growth_%", "Service_Payment_Growth_%", "Elasticity", "Cost_per_Customer"]].mean().round(2)
cluster_summary["Count"] = df_clustering["Cluster"].value_counts()
cluster_summary.reset_index(inplace=True)

cluster_summary
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Elasticity", 
    y="Cost_per_Customer", 
    hue="Cluster", 
    palette="Set1", 
    data=df_clustering,
    s=100
)
plt.title("Elasticity vs. Cost per Customer by Cluster")
plt.xlabel("Elasticity")
plt.ylabel("Cost per Customer")
plt.grid(True)
plt.tight_layout()
plt.show()
# Group by cluster and summarize
cluster_summary = df_clustering.groupby("Cluster")[[
    "Customer_Growth_%", 
    "Service_Payment_Growth_%", 
    "Elasticity", 
    "Cost_per_Customer"
]].mean().round(2)

# Add count of weeks in each cluster
cluster_summary["Count"] = df_clustering["Cluster"].value_counts()
print(cluster_summary.reset_index())
