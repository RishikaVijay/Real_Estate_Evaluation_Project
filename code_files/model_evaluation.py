import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "../dataset/Real_Estate_Evaluation_Data.csv")

df = pd.read_csv(file_path)
x = df.drop("Y house price of unit area", axis=1)
y = df["Y house price of unit area"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model_path = os.path.join(base_path, "trained_model.pkl")
model = joblib.load(model_path)
y_pred = model.predict(x_test)
print("Predictions on test data:\n", y_pred)
comparison_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print(comparison_df.head(10))  # Show first 10 comparisons


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("Evaluation Results:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared Score (R²):", r2)
