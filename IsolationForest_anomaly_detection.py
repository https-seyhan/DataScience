from sklearn.ensemble import IsolationForest

# ---------- Forecasting for Next 5 Weeks Using Polynomial Regression ----------
# Extend customer growth (assuming ~10% weekly growth as a base)
future_customers = [int(df["Customers"].iloc[-1] * (1.1 ** i)) for i in range(1, 6)]
future_weeks = list(range(df["Week"].iloc[-1] + 1, df["Week"].iloc[-1] + 6))
future_customers_array = np.array(future_customers).reshape(-1, 1)
future_customers_poly = poly.transform(future_customers_array)
future_payments_pred = poly_model.predict(future_customers_poly)

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    "Week": future_weeks,
    "Customers": future_customers,
    "Forecasted_Payments": future_payments_pred
})

# ---------- Combine with original data for visualization ----------
combined_df = pd.concat([
    df[["Week", "Customers", "Service_Payments"]].rename(columns={"Service_Payments": "Payments"}),
    forecast_df.rename(columns={"Forecasted_Payments": "Payments"})
])

# ---------- Anomaly Detection using Isolation Forest ----------
# Use only original data for anomaly detection
features = df[["Customers", "Service_Payments"]]
iso_forest = IsolationForest(contamination=0.2, random_state=42)
df["Anomaly_Score"] = iso_forest.fit_predict(features)

# ---------- Visualization ----------
plt.figure(figsize=(10, 6))

# Original and forecast points
plt.plot(combined_df["Week"], combined_df["Payments"], label="Payments (Actual + Forecast)", color="blue", marker='o')
plt.axvline(df["Week"].max(), color='gray', linestyle='--', label="Forecast Start")

# Highlight anomalies
anomalies = df[df["Anomaly_Score"] == -1]
plt.scatter(anomalies["Week"], anomalies["Service_Payments"], color='red', label="Anomaly", zorder=5)

plt.title("Customer Service Payments: Actual, Forecasted & Anomalies")
plt.xlabel("Week")
plt.ylabel("Payments ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

forecast_df.round(2)

STDOUT/STDERR

/home/sandbox/.local/lib/python3.11/site-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but IsolationForest was fitted with feature names
  warnings.warn(

Result

   Week  Customers  Forecasted_Payments
0     6       2090             28136.73
1     7       2299             32557.73
2     8       2528             37765.36
3     9       2781             43960.78
4    10       3059             51303.48

Output image
🔮 Forecast + Anomaly Detection Summary
✅ 5-Week Forecast (Based on Polynomial Growth)
Week	Estimated Customers	Forecasted Service Payments ($)
6	2,090	28,136.73
7	2,299	32,557.73
8	2,528	37,765.36
9	2,781	43,960.78
10	3,059	51,303.48

Insight: If current growth trends continue, service costs could double within 5 weeks, indicating urgent need for scale management.
🚨 Anomaly Detection (Isolation Forest)

Anomalies in the historical data were identified based on deviations in the relationship between customer counts and payments. These points could signal:

    Unusual service issues

    Unexpected cost spikes (e.g., system failures, manual support overload)

    Incorrect data entries
