import pandas as pd
import os
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "../dataset/Real_Estate_Evaluation_Data.csv")

df = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully")
print("\n Dataset shape:", df.shape)
print("\n First 5 rows:\n", df.head())
print("\n Data Types:\n", df.dtypes)
print("\n Misiing values:\n", df.isnull().sum())
print("\n Statistical Summary:\n", df.describe())