# Phoenixing Detection Pipeline with Excel and PDF Export

import pandas as pd
import numpy as np
import networkx as nx
from fuzzywuzzy import fuzz
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os

# Step 1: Load and Clean Data
companies = pd.read_csv("companies.csv")
directors = pd.read_csv("directors.csv")
liquidations = pd.read_csv("liquidations.csv")

# Clean names and convert dates
companies["Name"] = companies["Name"].str.lower().str.strip()
companies["StartDate"] = pd.to_datetime(companies["StartDate"])
companies["EndDate"] = pd.to_datetime(companies["EndDate"])
liquidations["LiquidationDate"] = pd.to_datetime(liquidations["LiquidationDate"])

# Step 2: Detect Potential Phoenix Pairs
potential_phoenix = []

def is_similar(name1, name2, threshold=85):
    return fuzz.token_set_ratio(name1, name2) >= threshold

for _, old_row in liquidations.iterrows():
    director_id = old_row['DirectorID']
    old_name = old_row['CompanyName']
    end_date = old_row['LiquidationDate']

    new_cos = companies[(companies['DirectorID'] == director_id) &
                        (companies['StartDate'] > end_date)]

    for _, new_row in new_cos.iterrows():
        if is_similar(old_name, new_row['Name']):
            potential_phoenix.append({
                "DirectorID": director_id,
                "OldCompany": old_name,
                "NewCompany": new_row['Name'],
                "DateDiff": (new_row['StartDate'] - end_date).days,
                "AddressMatch": old_row.get('Address') == new_row.get('Address')
            })

phoenix_df = pd.DataFrame(potential_phoenix)

# Step 3: Build Network Graph
G = nx.Graph()
for _, row in companies.iterrows():
    G.add_edge(f"Director_{row['DirectorID']}", f"Company_{row['ACN']}")

# Optionally, extract connected components
components = list(nx.connected_components(G))

# Step 4: Anomaly Detection with Isolation Forest
companies['LifespanDays'] = (companies['EndDate'] - companies['StartDate']).dt.days
# Dummy data for missing columns in this example
dummy_cols = ['TotalDebt', 'Assets', 'NumEmployees', 'PastLiquidations']
for col in dummy_cols:
    if col not in companies.columns:
        companies[col] = np.random.randint(1, 1000, size=len(companies))

features = companies[['LifespanDays', 'TotalDebt', 'Assets', 'NumEmployees', 'PastLiquidations']].dropna()
model = IsolationForest(contamination=0.05, random_state=42)
companies['PhoenixRiskScore'] = model.fit_predict(features)

# Step 5: Visualization
plt.figure(figsize=(10, 6))
sns.histplot(companies['PhoenixRiskScore'], bins=3, kde=False)
plt.title("Distribution of Phoenix Risk Scores")
plt.xlabel("Risk Score (-1=High Risk, 1=Low Risk)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("risk_score_distribution.png")
plt.close()

# Step 6: Export to Excel
phoenix_df.to_excel("phoenix_report.xlsx", index=False)

# Step 7: Export to PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Phoenixing Detection Report", ln=True, align='C')

# Add summary statistics
pdf.set_font("Arial", size=10)
pdf.ln(10)
pdf.cell(200, 10, txt=f"Total companies analyzed: {len(companies)}", ln=True)
pdf.cell(200, 10, txt=f"Potential phoenix cases detected: {len(phoenix_df)}", ln=True)

# Add image
if os.path.exists("risk_score_distribution.png"):
    pdf.image("risk_score_distribution.png", x=10, y=None, w=180)

# Add table of top phoenix cases
pdf.add_page()
pdf.set_font("Arial", size=10)
pdf.cell(200, 10, txt="Top Potential Phoenix Cases", ln=True, align='L')
pdf.ln(5)

for i, row in phoenix_df.head(10).iterrows():
    line = f"Director {row['DirectorID']}: {row['OldCompany']} -> {row['NewCompany']}, Days Between: {row['DateDiff']}, AddressMatch: {row['AddressMatch']}"
    pdf.multi_cell(0, 10, line)

pdf.output("phoenix_report.pdf")

# Alert for High-Risk Directors
high_risk = companies[companies['PhoenixRiskScore'] == -1]

print("\n⚠️ High-Risk Directors")
for _, row in high_risk.iterrows():
    print(f"Director {row['DirectorID']} linked to high-risk company: {row['Name']}")
