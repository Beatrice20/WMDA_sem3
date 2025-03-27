## Exercise 6 (10 minutes): kNN for Regression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic dataset
np.random.seed(42)
num_samples = 30

# Let's generate two features (e.g., Feature1, Feature2) and a target
X = np.random.rand(num_samples, 2) * 10
# Define a "true" relationship for the target: y = 3*X1 + 2*X2 + noise
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 5, size=num_samples)

# Convert to a DataFrame for clarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Target"] = y

# 2. Separate features and target
X = df[["Feature1", "Feature2"]]
y = df["Target"]

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Feature scaling (recommended for distance-based methods like kNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Create and train the kNN Regressor
#    We'll start with n_neighbors=3 (can try different values)
knn3_reg = KNeighborsRegressor(n_neighbors=3)
knn3_reg.fit(X_train_scaled, y_train)
y_pred_3 = knn3_reg.predict(X_test_scaled)

# 6. Evaluate on the test set
print("For k = 3")
print("R²:", r2_score(y_test, y_pred_3))
print("MSE:", mean_squared_error(y_test, y_pred_3))
print("MAE:", mean_absolute_error(y_test, y_pred_3))
print()

# 7. (Optional) Explore the effect of different k values
#    You can loop over various values of k and compare performance.
k_values = [3, 5, 7, 9]
print("kNN Regression Performance for different k values:\n")

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"k={k} -> R²: {r2:.3f}, MSE: {mse:.3f}, MAE: {mae:.3f}")