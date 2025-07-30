import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# -------- 1. Load ASIC/ABN data --------
# Example schema — replace with actual data source (CSV, DB, API, etc.)
data = [
    {"abn": "12345678901", "acn": "1122334455", "entity_name": "Alpha Constructions", 
     "address": "12 Main St, Sydney NSW", "status": "cancelled", "date": "2023-06-01"},
    {"abn": "10987654321", "acn": "9988776655", "entity_name": "Alpha Rebuilds", 
     "address": "12 Main Street, Sydney NSW", "status": "active", "date": "2023-06-10"},
    {"abn": "22233344455", "acn": "5566778899", "entity_name": "Beta Builders", 
     "address": "90 George St, Sydney NSW", "status": "active", "date": "2023-06-15"},
]

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# -------- 2. Separate closed and opened businesses --------
closed = df[df['status'].str.lower().isin(['cancelled', 'deregistered', 'closed'])].reset_index(drop=True)
opened = df[df['status'].str.lower().isin(['active', 'registered'])].reset_index(drop=True)

# -------- 3. Vectorize addresses using TF-IDF --------
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
combined_addresses = closed['address'].tolist() + opened['address'].tolist()
tfidf_matrix = vectorizer.fit_transform(combined_addresses)

closed_vecs = tfidf_matrix[:len(closed)]
opened_vecs = tfidf_matrix[len(closed):]

# -------- 4. Compute cosine similarity --------
similarity_matrix = cosine_similarity(closed_vecs, opened_vecs)

# -------- 5. Threshold + Temporal Logic --------
THRESHOLD = 0.75
G = nx.Graph()

for i, closed_row in closed.iterrows():
    for j, opened_row in opened.iterrows():
        similarity = similarity_matrix[i, j]
        if similarity >= THRESHOLD and opened_row['date'] > closed_row['date']:
            closed_id = f"{closed_row['abn']} ({closed_row['entity_name']})"
            opened_id = f"{opened_row['abn']} ({opened_row['entity_name']})"

            G.add_node(closed_id, label="closed", address=closed_row['address'])
            G.add_node(opened_id, label="opened", address=opened_row['address'])
            G.add_edge(closed_id, opened_id, weight=similarity)

# -------- 6. Print suspected phoenixing links --------
print("🚨 Suspected Phoenixing Cases:")
for u, v, data in G.edges(data=True):
    print(f"{u} ➡ {v} | Similarity: {data['weight']:.2f}")

# -------- 7. Optional: Visualize --------
pos = nx.spring_layout(G, seed=42)
edge_labels = nx.get_edge_attributes(G, 'weight')

plt.figure(figsize=(12, 6))
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2500, font_size=9)
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
plt.title("Suspected Phoenixing: ASIC/ABN Entities at Similar Addresses")
plt.tight_layout()
plt.show()
