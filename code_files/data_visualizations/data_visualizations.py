import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "../../dataset/Real_Estate_Evaluation_Data.csv")
df = pd.read_csv(data_path)


# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "correlation_heatmap.png"))

# Price vs House Age
plt.figure(figsize=(8, 5))
sns.scatterplot(x="X2 house age", y="Y house price of unit area", data=df)
plt.title("House Price vs Age of House")
plt.savefig(os.path.join(base_path, "price_vs_age.png"))

# Price vs Distance to MRT station
plt.figure(figsize=(8, 5))
sns.scatterplot(x="X3 distance to the nearest MRT station", y="Y house price of unit area", data=df)
plt.title("House Price vs Distance to MRT Station")
plt.savefig(os.path.join(base_path, "price_vs_mrt.png"))

plt.show()