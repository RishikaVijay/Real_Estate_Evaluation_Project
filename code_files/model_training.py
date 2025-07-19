import pandas as pd
import os
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "../dataset/Real_Estate_Evaluation_Data.csv")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(file_path)
x = df.drop("Y house price of unit area", axis=1)
y = df["Y house price of unit area"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

import joblib
joblib.dump(model, 'trained_model.pkl')