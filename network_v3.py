import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ----------- 1. Simulated Dataset ------------
np.random.seed(42)
df = pd.DataFrame({
    'organization': [f'Org_{i+1}' for i in range(10)],
    'sales_value': np.random.normal(10000, 1500, 10),
    'num_clients': np.random.randint(50, 200, 10),
    'num_employees': np.random.randint(10, 100, 10),
    'item_quantity': np.random.randint(100, 1000, 10),
    'num_services': np.random.randint(1, 20, 10)
})

# ----------- 2. Scale the Metrics ------------
features = ['sales_value', 'num_clients', 'num_employees', 'item_quantity', 'num_services']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ----------- 3. Compute Cosine Similarity Matrix ------------
similarity_matrix = cosine_similarity(X_scaled)

# ----------- 4. Build Organization Similarity Graph ------------
G = nx.Graph()
orgs = df['organization']

# Add nodes
for org in orgs:
    G.add_node(org)

# Add edges (only for high similarity)
threshold = 0.9  # You can adjust this
for i in range(len(orgs)):
    for j in range(i + 1, len(orgs)):
        sim = similarity_matrix[i][j]
        if sim > threshold:
            G.add_edge(orgs[i], orgs[j], weight=round(sim, 2))

# ----------- 5. Draw Network ------------
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 6))
edge_weights = [d['weight'] * 5 for (_, _, d) in G.edges(data=True)]

nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray',
        width=edge_weights, node_size=1500, font_weight='bold')
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Organization Similarity Network")
plt.show()
