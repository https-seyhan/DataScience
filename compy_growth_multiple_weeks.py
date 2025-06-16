import pandas as pd
import matplotlib.pyplot as plt

# Sample data with duplicated weeks (modify as needed)
data = {
    "Week": [1, 2, 2, 3, 4, 5, 5],
    "Customers": [1000, 1200, 1300, 1440, 1728, 1900, 1950],
    "Service_Payments": [10000, 13200, 14000, 16500, 20700, 24700, 25000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Handle duplicate weeks by aggregating (sum or average as appropriate)
aggregated_df = df.groupby("Week", as_index=False).agg({
    "Customers": "sum",              # or "mean", depending on your context
    "Service_Payments": "sum"        # usually summing payments makes sense
})

# Sort by Week in case grouping changes order
aggregated_df = aggregated_df.sort_values("Week")

# Calculate Customer Growth %
aggregated_df["Customer_Growth_%"] = aggregated_df["Customers"].pct_change() * 100

# Calculate Cost per Customer
aggregated_df["Cost_per_Customer"] = aggregated_df["Service_Payments"] / aggregated_df["Customers"]

# Calculate % change in Service Payments
aggregated_df["Service_Payment_Growth_%"] = aggregated_df["Service_Payments"].pct_change() * 100

# Calculate Elasticity
aggregated_df["Elasticity"] = aggregated_df["Service_Payment_Growth_%"] / aggregated_df["Customer_Growth_%"]

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Customer and Service Payment Trends
axs[0].plot(aggregated_df["Week"], aggregated_df["Customers"], label="Customers", marker='o')
axs[0].plot(aggregated_df["Week"], aggregated_df["Service_Payments"], label="Service Payments ($)", marker='s')
axs[0].set_ylabel("Count / Payments ($)")
axs[0].set_title("Customer Growth vs. Service Payments")
axs[0].legend()
axs[0].grid(True)

# Cost per Customer and Elasticity
axs[1].plot(aggregated_df["Week"], aggregated_df["Cost_per_Customer"], label="Cost per Customer ($)", marker='o')
axs[1].plot(aggregated_df["Week"], aggregated_df["Elasticity"], label="Elasticity", marker='s')
axs[1].set_ylabel("Cost / Elasticity")
axs[1].set_xlabel("Week")
axs[1].set_title("Service Cost Efficiency Metrics")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Display rounded table
aggregated_df.round(2)
