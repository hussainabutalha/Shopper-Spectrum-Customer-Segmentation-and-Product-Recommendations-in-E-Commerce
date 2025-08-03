import pandas as pd
import numpy as np

# Load the dataset
# Assuming the file is named 'data.csv' and is in the same directory
df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')

# --- Data Cleaning ---

# Remove rows with missing CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# Remove cancelled invoices (InvoiceNo starts with 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove returns (negative quantity) and zero prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Convert CustomerID to integer
df['CustomerID'] = df['CustomerID'].astype(int)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("Data preprocessed successfully!")
print(df.head())

#STEP 2 CAL RFM
# Create 'TotalPrice' column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Set a snapshot date (latest transaction date + 1 day) for Recency calculation
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Calculate RFM values
rfm_data = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

# Rename columns
rfm_data.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalPrice': 'Monetary'}, inplace=True)

print("RFM features calculated:")
print(rfm_data.head())

#STEP 3 K-MEANS
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