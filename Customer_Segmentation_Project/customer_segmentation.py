import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load Dataset
df = pd.read_csv("data/ifood_df.csv")
print("Dataset Loaded Successfully")
print(df.head())

# 2. Data Cleaning
df = df.dropna()

# 3. Feature Engineering
df['TotalSpend'] = (
    df['MntWines'] +
    df['MntFruits'] +
    df['MntMeatProducts'] +
    df['MntFishProducts'] +
    df['MntSweetProducts'] +
    df['MntGoldProds']
)

features = df[['Income', 'TotalSpend', 'Recency']]

# 4. Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 5. Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# 6. Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled_features)

# 7. Visualization
sns.scatterplot(
    x=df['Income'],
    y=df['TotalSpend'],
    hue=df['Segment'],
    palette='viridis'
)
plt.title("Customer Segmentation")
plt.show()

# 8. Segment Analysis
print("\nCustomer Segment Analysis:")
print(df.groupby('Segment')[['Income', 'TotalSpend', 'Recency']].mean())
