import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import networkx as nx
import matplotlib.pyplot as plt

# ----- Sample Data (you can replace with your own) -----
df = pd.DataFrame({
    'organization': [f'Org_{i+1}' for i in range(10)],
    'sales_value': np.random.normal(10000, 2000, 10),
    'num_clients': np.random.randint(50, 200, 10),
    'num_employees': np.random.randint(10, 100, 10),
    'item_quantity': np.random.randint(100, 1000, 10),
    'num_services': np.random.randint(1, 15, 10)
})

# ----- Step 1: Normalize Features -----
features = ['sales_value', 'num_clients', 'num_employees', 'item_quantity', 'num_services']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
df_zscores = pd.DataFrame(X_scaled, columns=features)
df_zscores['organization'] = df['organization']

# ----- Step 2: Mahalanobis Distance -----
cov_matrix = np.cov(X_scaled, rowvar=False)
inv_cov = np.linalg.inv(cov_matrix)
mean_vec = np.mean(X_scaled, axis=0)
mahal_distances = [distance.mahalanobis(row, mean_vec, inv_cov) for row in X_scaled]
df['mahalanobis'] = mahal_distances
df['is_outlier'] = df['mahalanobis'] > 3.0  # adjustable threshold

# ----- Step 3: Flag Extreme Metrics Using Z-score > 2.0 -----
z_threshold = 2.0
extreme_flags = df_zscores[features].abs() > z_threshold
extreme_flags['organization'] = df['organization']
extreme_flags = extreme_flags.merge(df[['organization', 'is_outlier']], on='organization')
metric_outliers = extreme_flags[extreme_flags['is_outlier']]

# ----- Step 4: Cosine Similarity & Network -----
similarity = cosine_similarity(X_scaled)
orgs = df['organization'].tolist()
G = nx.Graph()
G.add_nodes_from(orgs)

# Add edges if similarity > 0.5
threshold = 0.5
for i in range(len(orgs)):
    for j in range(i + 1, len(orgs)):
        sim = similarity[i][j]
        if sim > threshold:
            G.add_edge(orgs[i], orgs[j], weight=round(sim, 2))

# ----- Step 5: Visualize Network with Outliers -----
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 7))
node_colors = ['red' if df[df['organization'] == node]['is_outlier'].values[0] else 'skyblue' for node in G.nodes]
nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray',
        node_size=1500, width=1.5, font_weight='bold')
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Organization Similarity Network (Red = Outlier)")
plt.show()

# ----- Step 6: Report Which Metrics Triggered Outlier -----
print("\n📌 Outlier Analysis by Metric:")
for _, row in metric_outliers.iterrows():
    org = row['organization']
    flagged_metrics = [metric for metric in features if row[metric]]
    print(f"🔴 {org} is an outlier due to: {', '.join(flagged_metrics) if flagged_metrics else 'Multivariate deviation only'}")
