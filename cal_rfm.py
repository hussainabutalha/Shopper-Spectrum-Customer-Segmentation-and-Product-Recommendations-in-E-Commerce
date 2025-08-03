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