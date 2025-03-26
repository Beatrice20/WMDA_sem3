## Exercise 2 (10 minutes): Polynomial Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic non-linear dataset
np.random.seed(42)
num_samples = 30

# Single feature for clarity (e.g., 'sqft' or just X)
X = np.linspace(0, 10, num_samples).reshape(-1, 1)

# True relationship: y = 2 * X^2 - 3 * X + noise
y_true = 2 * (X**2) - 3 * X
noise = np.random.normal(0, 3, size=num_samples)
y = y_true.flatten() + noise

# Convert to DataFrame
df = pd.DataFrame({"Feature": X.flatten(), "Target": y})

# 2. Separate features and target
X = df[["Feature"]]
y = df["Target"]

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Transform features to polynomial (degree=2 or 3 for illustration)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 5. Create and train a Linear Regression model on the polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly) # Use the model to predict on the test set

# 6. Evaluate the model
r2_polynomial = r2_score(y_test, y_pred)
mse_polynomial = mean_squared_error(y_test, y_pred)
mae_polynomial = mean_absolute_error(y_test, y_pred)
print("R² Score:", r2_polynomial)
print("MSE:", mse_polynomial)
print("MAE:", mae_polynomial)

  # Fit a linear model to these high-degree polynomial features
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)

# Evaluate the liniar model
r2_linear = r2_score(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print("R² Score:", r2_linear)
print("MSE:", mse_linear)
print("MAE:", mae_linear)

# 7. Optional: Plot to visualize the fit
#    Generate a smooth curve for plotting
plt.scatter(X, y, label="Noisy Data Points")
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot_pred = model.predict(X_plot_poly)
plt.plot(X_plot, y_plot_pred, label="Polynomial Fit (Degree=2)", color='red')
plt.plot(X_plot, 2 * X_plot**2 - 3 * X_plot, label="True Underlying Trend (Quadratic)", color='blue')

plt.title("Polynomial Regression vs. True Quadratic Trend")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.show()
