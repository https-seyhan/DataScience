import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.fuzz import token_sort_ratio
import matplotlib.pyplot as plt

# -------- 1. Sample Data --------
df_a = pd.DataFrame({
    'abn': ['11111111111', '22222222222'],
    'address': ['12 Main St, Sydney', '90 George Street, Sydney']
})

df_b = pd.DataFrame({
    'address': ['12 Main Street, Sydney', '92 George St, Sydney', '200 King St, Melbourne']
})

# -------- 2. Vectorize All Addresses --------
all_addresses = df_a['address'].tolist() + df_b['address'].tolist()
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
tfidf_matrix = vectorizer.fit_transform(all_addresses)

vec_a = tfidf_matrix[:len(df_a)]
vec_b = tfidf_matrix[len(df_a):]

# -------- 3. Cosine Similarity and Fuzzy Matching --------
cos_sim = cosine_similarity(vec_a, vec_b)
COSINE_THRESHOLD = 0.75
FUZZY_THRESHOLD = 85

# -------- 4. Build Network --------
G = nx.Graph()

for i, row in df_a.iterrows():
    abn_node = f"ABN: {row['abn']}"
    address_a = f"A: {row['address']}"
    
    # Connect ABN to its own address (dataset A)
    G.add_node(abn_node, type='abn')
    G.add_node(address_a, type='address_a')
    G.add_edge(abn_node, address_a, relation="self")

    # Compare with dataset B
    for j, addr_b in df_b['address'].items():
        address_b = f"B: {addr_b}"
        cos_score = cos_sim[i, j]
        fuzz_score = token_sort_ratio(row['address'], addr_b)
        
        if cos_score >= COSINE_THRESHOLD or fuzz_score >= FUZZY_THRESHOLD:
            G.add_node(address_b, type='address_b')
            G.add_edge(abn_node, address_b, relation="match", cosine=round(cos_score, 2), fuzzy=fuzz_score)

# -------- 5. Print Results --------
print("🔗 ABNs linked to their addresses and similar B addresses:")
for u, v, d in G.edges(data=True):
    print(f"{u} ↔ {v} | Relation: {d.get('relation')} | Cosine: {d.get('cosine', '-')}, Fuzzy: {d.get('fuzzy', '-')}")
    
# -------- 6. Visualize --------
pos = nx.spring_layout(G, seed=42)
colors = []
for node in G.nodes:
    t = G.nodes[node]['type']
    colors.append('lightblue' if t == 'abn' else ('lightgreen' if t == 'address_a' else 'orange'))

plt.figure(figsize=(12, 6))
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2500, font_size=9)

# Label only match edges
edge_labels = {(u, v): f"{d.get('cosine','')}" for u, v, d in G.edges(data=True) if d.get("relation") == "match"}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title("ABN to Own & Similar Addresses (Dataset A and B)")
plt.tight_layout()
plt.show()
