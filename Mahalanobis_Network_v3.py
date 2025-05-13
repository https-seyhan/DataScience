import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# ----- Step 0: Sample Data -----
df = pd.DataFrame({
    'organization': [f'Org_{i+1}' for i in range(10)],
    'sales_value': np.random.normal(10000, 2000, 10),
    'num_clients': np.random.randint(50, 200, 10),
    'num_employees': np.random.randint(10, 100, 10),
    'item_quantity': np.random.randint(100, 1000, 10),
    'num_services': np.random.randint(1, 15, 10)
})

# ----- Step 1: Normalize -----
features = ['sales_value', 'num_clients', 'num_employees', 'item_quantity', 'num_services']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
df_z = pd.DataFrame(X_scaled, columns=features)
df_z['organization'] = df['organization']

# ----- Step 2: Mahalanobis Distance for Outliers -----
cov_matrix = np.cov(X_scaled, rowvar=False)
inv_cov = np.linalg.inv(cov_matrix)
mean_vec = np.mean(X_scaled, axis=0)
mahal_distances = [distance.mahalanobis(row, mean_vec, inv_cov) for row in X_scaled]
df['mahalanobis'] = mahal_distances
df['is_outlier'] = df['mahalanobis'] > 3.0  # adjustable

# ----- Step 3: Identify Extreme Metrics (Z-score > 2) -----
z_thresh = 2.0
flags = df_z[features].abs() > z_thresh
flags['organization'] = df['organization']
flags = flags.merge(df[['organization', 'is_outlier']], on='organization')
metric_outliers = flags[flags['is_outlier']].drop(columns='is_outlier')

# ----- Step 4: Save Outlier Report to Excel -----
outlier_detail = df[df['is_outlier']].copy()
outlier_detail['triggering_metrics'] = outlier_detail['organization'].apply(
    lambda org: ', '.join([col for col in features if metric_outliers.loc[metric_outliers['organization'] == org, col].any()])
)
outlier_detail.to_excel("outlier_organizations.xlsx", index=False)
print("✅ Outlier details saved to 'outlier_organizations.xlsx'")

# ----- Step 5: Clustering (K-Means) -----
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# ----- Step 6: Similarity Network (Optional Visual) -----
similarity = cosine_similarity(X_scaled)
orgs = df['organization'].tolist()
G = nx.Graph()
G.add_nodes_from(orgs)
for i in range(len(orgs)):
    for j in range(i + 1, len(orgs)):
        sim = similarity[i][j]
        if sim > 0.5:
            G.add_edge(orgs[i], orgs[j], weight=round(sim, 2))

# Visualize Network
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)
node_colors = ['red' if df[df['organization'] == node]['is_outlier'].values[0] else 'skyblue' for node in G.nodes]
nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=1500, width=1.5)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Organization Network (Red = Outlier)")
plt.show()

# ----- Step 7: Heatmap of Z-Scores (Outliers Only) -----
if not metric_outliers.empty:
    heatmap_data = df_z.set_index('organization').loc[metric_outliers['organization'].unique()]
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data[features], annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("📊 Z-score Heatmap of Outlier Organizations")
    plt.tight_layout()
    plt.show()
else:
    print("ℹ️ No outliers to visualize in heatmap.")
