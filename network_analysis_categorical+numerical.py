import pandas as pd

# Sample dataset
data = pd.DataFrame({
    'Organization': ['OrgA', 'OrgB', 'OrgC', 'OrgD', 'OrgE'],
    'Product_Type': ['Software', 'Hardware', 'Software', 'Service', 'Hardware'],
    'Sales': [100000, 250000, 120000, 90000, 300000],
    'Num_Employees': [50, 120, 60, 40, 150]
})
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

features = ['Product_Type', 'Sales', 'Num_Employees']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), ['Product_Type']),
    ('num', StandardScaler(), ['Sales', 'Num_Employees'])
])

X = preprocessor.fit_transform(data[features])
from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(X)
import networkx as nx
import matplotlib.pyplot as plt

# Initialize graph
G = nx.Graph()

# Add nodes
for i, org in enumerate(data['Organization']):
    G.add_node(org, label=org)

# Add edges based on similarity threshold
threshold = 0.9  # Adjust as needed
for i in range(len(data)):
    for j in range(i+1, len(data)):
        similarity = similarity_matrix[i, j]
        if similarity > threshold:
            G.add_edge(data['Organization'][i], data['Organization'][j], weight=similarity)

# Draw graph
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', edge_color='gray', font_weight='bold')
plt.title("Organization Similarity Network")
plt.show()
from networkx.algorithms.community import greedy_modularity_communities

communities = list(greedy_modularity_communities(G))
for i, comm in enumerate(communities):
    print(f"Community {i+1}: {comm}")
 Interpretation

    Nodes represent organizations.

    Edges represent high similarity (based on product type, sales, and employees).

    Communities reveal clusters of similar organizations.

    You can add edge labels or weights to visualize how close organizations are.