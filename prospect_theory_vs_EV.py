import numpy as np
import matplotlib.pyplot as plt

# --- Prospect Theory Parameters ---
alpha = 0.88         # Value curve for gains
beta = 0.88          # Value curve for losses
lambda_loss = 2.25   # Loss aversion
gamma = 0.61         # Probability weighting (gain)
delta = 0.69         # Probability weighting (loss)

# --- Value Function (Prospect Theory) ---
def value(x):
    return x**alpha if x >= 0 else -lambda_loss * (abs(x)**beta)

# --- Probability Weighting Function ---
def weight(p, is_gain=True):
    if is_gain:
        return (p**gamma) / ((p**gamma + (1-p)**gamma)**(1/gamma))
    else:
        return (p**delta) / ((p**delta + (1-p)**delta)**(1/delta))

# --- Define Multiple Options (outcomes and probabilities) ---
options = {
    "Option A (Safe Gain)": {
        "outcomes": [20000],
        "probs": [1.0]
    },
    "Option B (Risky Gain)": {
        "outcomes": [50000, 0],
        "probs": [0.5, 0.5]
    },
    "Option C (Safe Loss)": {
        "outcomes": [-20000],
        "probs": [1.0]
    },
    "Option D (Risky Loss)": {
        "outcomes": [-50000, 0],
        "probs": [0.5, 0.5]
    },
    "Option E (Mixed Outcome)": {
        "outcomes": [100000, -90000],
        "probs": [0.4, 0.6]
    }
}

# --- Evaluate Each Option ---
results = []

for name, opt in options.items():
    ev = sum([p * x for p, x in zip(opt["probs"], opt["outcomes"])])
    pt = sum([
        weight(p, is_gain=(x >= 0)) * value(x)
        for p, x in zip(opt["probs"], opt["outcomes"])
    ])
    results.append((name, ev, pt))

# --- Print Results ---
print("📊 EV vs Prospect Theory Value Comparison:\n")
print(f"{'Option':<25} | {'EV ($)':>10} | {'PT Value':>10}")
print("-"*50)
for name, ev, pt in results:
    print(f"{name:<25} | {ev:>10,.2f} | {pt:>10,.2f}")

# --- Bar Chart: EV vs PT Value ---
labels = [r[0] for r in results]
ev_values = [r[1] for r in results]
pt_values = [r[2] for r in results]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12,6))
bars1 = ax.bar(x - width/2, ev_values, width, label='Expected Value', color='steelblue')
bars2 = ax.bar(x + width/2, pt_values, width, label='Prospect Theory Value', color='darkorange')

ax.set_ylabel('Value ($ / Utility)')
ax.set_title('Comparison of Expected Value vs Prospect Theory')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.legend()
ax.axhline(0, color='black', linewidth=0.7)
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()
