import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt

# ----- Sample Data -----
df = pd.DataFrame({
    'organization': [f'Org_{i+1}' for i in range(10)],
    'sales_value': np.random.normal(10000, 2000, 10),
    'num_clients': np.random.randint(50, 200, 10),
    'num_employees': np.random.randint(10, 100, 10),
    'item_quantity': np.random.randint(100, 1000, 10),
    'num_services': np.random.randint(1, 15, 10)
})

# ----- Scale the Data -----
features = ['sales_value', 'num_clients', 'num_employees', 'item_quantity', 'num_services']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ----- Mahalanobis Distance -----
cov_matrix = np.cov(X_scaled, rowvar=False)
inv_cov = np.linalg.inv(cov_matrix)
center = np.mean(X_scaled, axis=0)
mahalanobis_scores = [distance.mahalanobis(x, center, inv_cov) for x in X_scaled]
df['mahalanobis'] = mahalanobis_scores

# Mark outliers: threshold ~ 2.5–3.0 (adjustable)
threshold = 3.0
df['is_outlier'] = df['mahalanobis'] > threshold

# ----- Print Outliers -----
print("🔍 Outlier Organizations:")
print(df[df['is_outlier']][['organization', 'mahalanobis']])

# ----- Optional: Build Similarity Network (to spot isolates) -----
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(X_scaled)
G = nx.Graph()
orgs = df['organization'].tolist()
G.add_nodes_from(orgs)

# Add edges if similarity > 0.5
for i in range(len(orgs)):
    for j in range(i + 1, len(orgs)):
        sim = similarity[i][j]
        if sim > 0.5:
            G.add_edge(orgs[i], orgs[j], weight=round(sim, 2))

# ----- Visualize with Outliers Highlighted -----
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 6))

node_colors = ['red' if df[df['organization'] == node]['is_outlier'].values[0] else 'skyblue' for node in G.nodes]
nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray',
        node_size=1500, width=1.5, font_weight='bold')

plt.title("📊 Organization Network with Outliers Highlighted (Red)")
plt.show()
