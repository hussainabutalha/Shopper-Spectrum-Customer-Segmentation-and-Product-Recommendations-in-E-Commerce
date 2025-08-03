from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

# --- Scale the RFM data ---
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)

# --- Find optimal k using Elbow Method ---
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

# plt.figure(figsize=(8, 6))
# plt.plot(K, inertia, 'bo-')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method For Optimal k')
# plt.show() # You would typically view this plot to choose k, e.g., 4

# --- Fit K-Means with the chosen k (e.g., k=4) ---
optimal_k = 4
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm_data['Cluster'] = kmeans_model.fit_predict(rfm_scaled)

# --- Save the model and scaler for Streamlit ---
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Customer clusters:")
print(rfm_data.groupby('Cluster').agg({'Recency':'mean', 'Frequency':'mean', 'Monetary':'mean'}))