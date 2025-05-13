import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ----------- 1. Sample Dataset ------------
# Simulate 10 organizations with 5 numerical metrics
np.random.seed(42)
df = pd.DataFrame({
    'organization': [f'Org_{i+1}' for i in range(10)],
    'sales_value': np.random.normal(10000, 1500, 10),
    'num_clients': np.random.randint(50, 200, 10),
    'num_employees': np.random.randint(10, 100, 10),
    'item_quantity': np.random.randint(100, 1000, 10),
    'num_services': np.random.randint(1, 20, 10)
})

# ----------- 2. Normalize Values ------------
scaler = MinMaxScaler()
metrics = ['sales_value', 'num_clients', 'num_employees', 'item_quantity', 'num_services']
df_scaled = df.copy()
df_scaled[metrics] = scaler.fit_transform(df[metrics])

# ----------- 3. Build Bipartite Graph ------------
B = nx.Graph()

# Add nodes for organizations (set 0) and metrics (set 1)
orgs = df_scaled['organization']
B.add_nodes_from(orgs, bipartite=0)
B.add_nodes_from(metrics, bipartite=1)

# Add weighted edges (org ↔ metric)
for _, row in df_scaled.iterrows():
    org = row['organization']
    for metric in metrics:
        value = row[metric]
        B.add_edge(org, metric, weight=round(value, 2))

# ----------- 4. Draw the Network ------------
# Split node types for layout
org_nodes = [n for n in B.nodes if n in orgs.values]
metric_nodes = [n for n in B.nodes if n in metrics]

pos = nx.bipartite_layout(B, org_nodes)
plt.figure(figsize=(10, 6))
edge_weights = [d['weight'] * 5 for (_, _, d) in B.edges(data=True)]

# Draw
nx.draw(B, pos, with_labels=True, node_color=['lightblue' if n in org_nodes else 'lightgreen' for n in B.nodes],
        node_size=1500, edge_color='gray', width=edge_weights, font_size=10)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in B.edges(data=True)}
nx.draw_networkx_edge_labels(B, pos, edge_labels=edge_labels, font_size=8)
plt.title("Bipartite Network: Organizations and Their Metrics")
plt.show()
