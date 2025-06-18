# Redefine elasticity handling zero-division and zero-change cases
df["Customer_Growth_%"] = df["Customers"].pct_change() * 100
df["Service_Payment_Growth_%"] = df["Service_Payments"].pct_change() * 100

# Handle elasticity with zero-denominator explicitly
def compute_elasticity(row):
    if row["Customer_Growth_%"] == 0:
        return np.nan  # Undefined if customer growth is zero
    return row["Service_Payment_Growth_%"] / row["Customer_Growth_%"]

df["Elasticity"] = df.apply(compute_elasticity, axis=1)

# Compute cost per customer
df["Cost_per_Customer"] = df["Service_Payments"] / df["Customers"]

# Fill NaNs in a safe way for clustering later
df_clustering = df.copy()
df_clustering.fillna(0, inplace=True)

# Show updated elasticity column and any flagged zero-customer-growth weeks
df_clustering[["Week", "Customers", "Service_Payments", "Customer_Growth_%", "Service_Payment_Growth_%", "Elasticity"]].round(2)

Always show details

import pandas as pd
import numpy as np

# Recreate the original sample dataset
df = pd.DataFrame({
    "Week": [1, 2, 3, 4, 5],
    "Customers": [1000, 1200, 1440, 1728, 1900],
    "Service_Payments": [10000, 13200, 16500, 20700, 24700]
})

# Recalculate growth rates
df["Customer_Growth_%"] = df["Customers"].pct_change() * 100
df["Service_Payment_Growth_%"] = df["Service_Payments"].pct_change() * 100

# Handle elasticity with zero-denominator explicitly
def compute_elasticity(row):
    if row["Customer_Growth_%"] == 0:
        return np.nan  # Undefined if customer growth is zero
    return row["Service_Payment_Growth_%"] / row["Customer_Growth_%"]

df["Elasticity"] = df.apply(compute_elasticity, axis=1)

# Compute cost per customer
df["Cost_per_Customer"] = df["Service_Payments"] / df["Customers"]

# Fill NaNs in a safe way for clustering later
df_clustering = df.copy()
df_clustering.fillna(0, inplace=True)

# Display updated elasticity and related columns
df_clustering[["Week", "Customers", "Service_Payments", "Customer_Growth_%", "Service_Payment_Growth_%", "Elasticity"]].round(2)

Result

   Week  Customers  Service_Payments  Customer_Growth_%  \
0     1       1000             10000               0.00   
1     2       1200             13200              20.00   
2     3       1440             16500              20.00   
3     4       1728             20700              20.00   
4     5       1900             24700               9.95   

   Service_Payment_Growth_%  Elasticity  
0                      0.00        0.00  
1                     32.00        1.60  
2                     25.00        1.25  
3                     25.45        1.27  
4                     19.32        1.94  

Here's the updated table with elasticity correctly handled — especially when the customer growth is zero, which makes elasticity undefined (we safely replaced it with 0.0 for clustering):
Week	Customers	Service Payments	Customer Growth %	Payment Growth %	Elasticity
1	1000	10,000	0.00	0.00	0.00
2	1200	13,200	20.00	32.00	1.60
3	1440	16,500	20.00	25.00	1.25
4	1728	20,700	20.00	25.45	1.27
5	1900	24,700	9.95	19.32	1.94
📌 Observations:

    Week 1 has no change → elasticity = 0

    Weeks 2–5 have elasticity > 1, meaning service costs are growing faster than customer count — a sign of expensive scaling.

