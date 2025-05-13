import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# STEP 1: Simulated Dataset (replace with your real data)
df = pd.DataFrame({
    'organization': [f'Org_{i+1}' for i in range(10)],
    'sales_value': np.random.normal(10000, 1500, 10),
    'num_clients': np.random.randint(50, 200, 10),
    'num_employees': np.random.randint(10, 100, 10),
    'item_quantity': np.random.randint(100, 1000, 10),
    'num_services': np.random.randint(1, 20, 10)
})

# STEP 2: Normalize Metrics
features = ['sales_value', 'num_clients', 'num_employees', 'item_quantity', 'num_services']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# STEP 3: Compute Similarities
cos_sim_matrix = cosine_similarity(X_scaled)

# STEP 4: Create Network of Similar Organizations
G = nx.Graph()
orgs = df['organization']

# Add nodes for each organization
for org in orgs:
    G.add_node(org)

# Add edges where similarity > 0.9
threshold = 0.9
for i in range(len(orgs)):
    for j in range(i + 1, len(orgs)):
        sim = cos_sim_matrix[i][j]
        if sim > threshold:
            G.add_edge(orgs[i], orgs[j], weight=round(sim, 2))

# STEP 5: Draw the Network
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
edges = G.edges(data=True)
weights = [d['weight'] * 5 for (_, _, d) in edges]  # thicker = more similar

nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', width=weights, node_size=1500)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("📊 Similarity Network of Organizations")
plt.show()
