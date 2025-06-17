# Re-import necessary libraries after code state reset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Sample data
data = {
    "Week": [1, 2, 3, 4, 5],
    "Customers": [1000, 1200, 1440, 1728, 1900],
    "Service_Payments": [10000, 13200, 16500, 20700, 24700]
}

# Create DataFrame
df = pd.DataFrame(data)

# Prepare data for regression models
X = df["Customers"].values.reshape(-1, 1)
y = df["Service_Payments"].values

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

# Calculate metrics
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)

mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(df["Customers"], df["Service_Payments"], color='black', label="Actual Data", zorder=3)
plt.plot(df["Customers"], y_pred_linear, color='blue', label=f"Linear Fit (R²={r2_linear:.2f})")
plt.plot(df["Customers"], y_pred_poly, color='green', linestyle='--', label=f"Polynomial Fit (R²={r2_poly:.2f})")
plt.xlabel("Number of Customers")
plt.ylabel("Service Payments ($)")
plt.title("Advanced Modeling: Service Payments vs. Customers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

{
    "Linear_Regression": {
        "MSE": round(mse_linear, 2),
        "R_squared": round(r2_linear, 2)
    },
    "Polynomial_Regression": {
        "MSE": round(mse_poly, 2),
        "R_squared": round(r2_poly, 2)
    }
}

Result

{'Linear_Regression': {'MSE': 194420.21, 'R_squared': 0.99},
 'Polynomial_Regression': {'MSE': 109061.17, 'R_squared': 1.0}}

Output image
📊 Advanced Modeling Summary

We applied Linear Regression and Polynomial Regression (degree 2) to evaluate the relationship between the number of customers and service payments.
🔍 Model Comparison
Model	R² (Goodness of Fit)	MSE (Error)
Linear Regression	0.99 (very high)	194,420
Polynomial (Deg 2)	1.00 (perfect fit)	109,061
📈 Interpretation:

    Both models show a strong fit, but the polynomial regression fits the data near-perfectly, indicating a non-linear growth in service payments.

    This suggests that as customers grow, the cost per customer increases at an accelerating rate — an early warning sign of scalability issues.

    The non-linearity captured by the polynomial model reinforces that growth is becoming more expensive per new customer added.

✅ Business Implication:

    Your growth strategy may be exceeding your operational capacity.

    Consider automation, self-service options, or tiered support to absorb growing demand more efficiently.

    Regularly reassess your cost structure and team load as part of scaling.
