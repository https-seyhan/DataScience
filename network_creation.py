import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ----------- 1. Sample Data ------------
# Simulating dataset (replace with your real data)
np.random.seed(0)
df = pd.DataFrame({
    'sales_value': np.random.normal(10000, 2000, 50),
    'num_clients': np.random.randint(50, 200, 50),
    'num_employees': np.random.randint(10, 100, 50),
    'item_quantity': np.random.randint(100, 1000, 50),
    'num_services': np.random.randint(1, 10, 50)
})

# ----------- 2. Correlation Matrix ------------
variables = df.columns
G = nx.Graph()

# Add nodes
for var in variables:
    G.add_node(var)

# Add edges based on correlation
for i in range(len(variables)):
    for j in range(i + 1, len(variables)):
        var1, var2 = variables[i], variables[j]
        corr, _ = pearsonr(df[var1], df[var2])
        if abs(corr) > 0.3:  # only include meaningful correlations
            G.add_edge(var1, var2, weight=round(corr, 2))

# ----------- 3. Visualize Network ------------
pos = nx.spring_layout(G, seed=42)
edges = G.edges(data=True)
weights = [abs(d['weight']) * 5 for (_, _, d) in edges]  # Scale weights for visibility

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', width=weights, font_weight='bold')
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Network Analysis of Organizational Metrics")
plt.show()
