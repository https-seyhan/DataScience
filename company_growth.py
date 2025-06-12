import pandas as pd
import matplotlib.pyplot as plt

# Sample data: Replace with real data as needed
data = {
    "Week": [1, 2, 3, 4, 5],
    "Customers": [1000, 1200, 1440, 1728, 1900],
    "Service_Payments": [10000, 13200, 16500, 20700, 24700]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate Customer Growth %
df["Customer_Growth_%"] = df["Customers"].pct_change() * 100

# Calculate Cost per Customer
df["Cost_per_Customer"] = df["Service_Payments"] / df["Customers"]

# Calculate % change in Service Payments
df["Service_Payment_Growth_%"] = df["Service_Payments"].pct_change() * 100

# Calculate Elasticity
df["Elasticity"] = df["Service_Payment_Growth_%"] / df["Customer_Growth_%"]

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Customer and Service Payment Trends
axs[0].plot(df["Week"], df["Customers"], label="Customers", marker='o')
axs[0].plot(df["Week"], df["Service_Payments"], label="Service Payments ($)", marker='s')
axs[0].set_ylabel("Count / Payments ($)")
axs[0].set_title("Customer Growth vs. Service Payments")
axs[0].legend()
axs[0].grid(True)

# Cost per Customer and Elasticity
axs[1].plot(df["Week"], df["Cost_per_Customer"], label="Cost per Customer ($)", marker='o')
axs[1].plot(df["Week"], df["Elasticity"], label="Elasticity", marker='s')
axs[1].set_ylabel("Cost / Elasticity")
axs[1].set_xlabel("Week")
axs[1].set_title("Service Cost Efficiency Metrics")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

df.round(2)
