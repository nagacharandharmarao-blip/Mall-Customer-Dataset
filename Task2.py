import kagglehub
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# =========================
# DOWNLOAD DATASET
# =========================
path = kagglehub.dataset_download(
    "vjchoudhary7/customer-segmentation-tutorial-in-python"
)

print("Dataset downloaded at:")
print(path)

# =========================
# LOAD CSV FILE
# =========================
csv_path = os.path.join(path, "Mall_Customers.csv")
data = pd.read_csv(csv_path)

print("\nDataset Preview:")
print(data.head())

# =========================
# FEATURE SELECTION
# =========================
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# ELBOW METHOD
# =========================
wcss = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

# =========================
# TRAIN FINAL MODEL
# =========================
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels
data["Cluster"] = clusters

# =========================
# PRINT RESULTS
# =========================
print("\nClustered Data (sample):")
print(data[["CustomerID", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head())

print("\nCluster distribution:")
print(data["Cluster"].value_counts())

# =========================
# SAVE OUTPUT IN CURRENT DIRECTORY
# =========================
output_file = "clustered_output.csv"
data.to_csv(output_file, index=False)

print("\nClustered file saved successfully at:")
print(os.path.abspath(output_file))

# =========================
# PLOTS (DISPLAY LAST)
# =========================
plt.figure()
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters,
    cmap="viridis"
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    s=200,
    label="Centroids"
)

plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()
