import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load saved models and data ---
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('item_similarity_df.pkl', 'rb') as f:
    item_similarity_df = pickle.load(f)
with open('product_descriptions.pkl', 'rb') as f:
    product_descriptions = pickle.load(f)

# --- Helper Functions ---
def get_recommendations(product_name, similarity_df, descriptions):
    try:
        stock_code = descriptions[descriptions['Description'] == product_name].index[0]
        similar_scores = similarity_df[stock_code].sort_values(ascending=False)
        top_5 = similar_scores.iloc[1:6]
        recommended_products = descriptions.loc[top_5.index]['Description'].tolist()
        return recommended_products
    except (IndexError, KeyError):
        return "Product not found. Please select one from the list."

def predict_cluster(recency, frequency, monetary):
    # Create a dataframe for the input
    input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    # Scale the input
    input_scaled = scaler.transform(input_data)
    # Predict the cluster
    prediction = kmeans_model.predict(input_scaled)
    # Map cluster number to a label
    cluster_labels = {0: 'At-Risk', 1: 'High-Value', 2: 'Regular', 3: 'Occasional'} # Adjust based on your cluster analysis
    return cluster_labels.get(prediction[0], 'Unknown Cluster')

# --- Streamlit App UI ---
st.title('🛒 Shopper Spectrum: Segmentation & Recommendations')

# --- Product Recommendation Module ---
st.header('🎯 Product Recommendation')
product_list = product_descriptions['Description'].unique().tolist()
selected_product = st.selectbox('Select a Product Name:', product_list)

if st.button('Get Recommendations'):
    recommendations = get_recommendations(selected_product, item_similarity_df, product_descriptions)
    if isinstance(recommendations, list):
        st.subheader('Top 5 Recommended Products:')
        for i, product in enumerate(recommendations):
            st.write(f"{i+1}. {product}")
    else:
        st.error(recommendations)

# --- Customer Segmentation Module ---
st.header('🔍 Customer Segmentation')
recency = st.number_input('Recency (days)', min_value=0, value=50)
frequency = st.number_input('Frequency (number of purchases)', min_value=1, value=10)
monetary = st.number_input('Monetary (total spend)', min_value=0.0, value=500.0, format="%.2f")

if st.button('Predict Cluster'):
    cluster = predict_cluster(recency, frequency, monetary)
    st.success(f'The customer belongs to the: **{cluster}** segment.')