import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chisquare

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

# ----- Benford's Law Section -----
def leading_digit_distribution(data):
    data = data[data > 0]  # Only positive values are valid
    leading_digits = data.astype(str).str.replace('.', '').str.lstrip('0').str[0].astype(int)
    digit_counts = leading_digits.value_counts(normalize=True).sort_index()
    digit_counts_abs = leading_digits.value_counts().sort_index()
    return digit_counts, digit_counts_abs, len(leading_digits)

# Benford's expected distribution
benford_dist = pd.Series({
    1: 0.301, 2: 0.176, 3: 0.125,
    4: 0.097, 5: 0.079, 6: 0.067,
    7: 0.058, 8: 0.051, 9: 0.046
})

print("\n🔍 Benford's Law Analysis:")
for col in features:
    print(f"\n📊 Checking '{col}':")
    col_data = df[col]
    actual_rel, actual_abs, valid_count = leading_digit_distribution(col_data)
    actual_rel_full = actual_rel.reindex(benford_dist.index, fill_value=0)
    actual_abs_full = actual_abs.reindex(benford_dist.index, fill_value=0)
    expected_abs = (benford_dist * valid_count).round().astype(int)

    # Plot comparison
    comparison_df = pd.DataFrame({
        'Benford': benford_dist,
        'Actual': actual_rel_full
    })
    comparison_df.plot(kind='bar', figsize=(8, 4), title=f"Benford's Law vs Actual for {col}")
    plt.xlabel("Leading Digit")
    plt.ylabel("Proportion")
    plt.grid(True)
    plt.show()

    # MAD (Mean Absolute Deviation)
    mad = np.mean(np.abs(actual_rel_full - benford_dist))
    print(f"  ➤ Mean Absolute Deviation (MAD): {mad:.4f}")

    # Chi-square test
    if expected_abs.sum() == actual_abs_full.sum():
        chi_stat, p_value = chisquare(actual_abs_full, f_exp=expected_abs)
        print(f"  ➤ Chi-Square Statistic: {chi_stat:.2f}, p-value: {p_value:.4f}")
    else:
        print(f"  ⚠️ Chi-square skipped: count mismatch (expected {expected_abs.sum()}, actual {actual_abs_full.sum()})")

# ----- Similarity -----
similarity = cosine_similarity(X_scaled)
orgs = df['organization'].tolist()

print("\n🤝 Similarity Scores Between Organizations:")
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
    print("\n⚠️ Still no edges. Try setting threshold = 0.3 or lower.")
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
