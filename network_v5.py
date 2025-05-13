import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ----- Sample Data -----
df = pd.DataFrame({
    'organization': [f'Org_{i+1}' for i in range(10)],
    'sales_value': np.random.normal(10000, 2000, 10),
    'num_clients': np.random.randint(50, 200, 10),
    'num_employees': np.random.randint(10, 100, 10),
    'item_quantity': np.random.randint(100, 1000, 10),
    'num_services': np.random.randint(1, 15, 10)
})

# ----- Normalize -----
features = ['sales_value', 'num_clients', 'num_employees', 'item_quantity', 'num_services']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ----- Similarity -----
similarity = cosine_similarity(X_scaled)
orgs = df['organization'].tolist()

# ----- Check similarities -----
print("Similarity Scores Between Organizations:")
for i in range(len(orgs)):
    for j in range(i + 1, len(orgs)):
        sim = similarity[i][j]
        print(f"{orgs[i]} vs {orgs[j]}: {sim:.2f}")

# ----- Graph -----
G = nx.Graph()
G.add_nodes_from(orgs)

# ----- Add edges if similarity > 0.4 -----
threshold = 0.4
for i in range(len(orgs)):
    for j in range(i + 1, len(orgs)):
        sim = similarity[i][j]
        if sim > threshold:
            G.add_edge(orgs[i], orgs[j], weight=round(sim, 2))

# ----- Visualize -----
if G.number_of_edges() == 0:
    print("⚠️ Still no edges. Try setting threshold = 0.3 or lower.")
else:
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    edge_weights = [d['weight'] * 5 for (_, _, d) in G.edges(data=True)]
    nx.draw(G, pos, with_labels=True, node_color='skyblue',
            edge_color='gray', width=edge_weights, node_size=1500, font_weight='bold')
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Organization Similarity Network (Threshold > 0.4)")
    plt.show()
