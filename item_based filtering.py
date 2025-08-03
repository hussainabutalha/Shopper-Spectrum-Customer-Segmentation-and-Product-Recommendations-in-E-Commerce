from sklearn.metrics.pairwise import cosine_similarity

# --- Create the customer-item matrix ---
# We use a subset of data for performance
df_rec = df.sample(n=20000, random_state=42)
customer_item_matrix = df_rec.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum')
customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)

# --- Create the item-item similarity matrix ---
item_similarity = cosine_similarity(customer_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=customer_item_matrix.columns, columns=customer_item_matrix.columns)

# --- Save the similarity matrix and product descriptions for Streamlit ---
# Create a mapping from StockCode to Description
product_descriptions = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')

with open('item_similarity_df.pkl', 'wb') as f:
    pickle.dump(item_similarity_df, f)
with open('product_descriptions.pkl', 'wb') as f:
    pickle.dump(product_descriptions, f)


# --- Recommendation Function ---
def get_recommendations(product_name, similarity_df, descriptions):
    try:
        # Find stock code for the product name
        stock_code = descriptions[descriptions['Description'] == product_name].index[0]

        # Get similarity scores for the product
        similar_scores = similarity_df[stock_code].sort_values(ascending=False)

        # Get top 5 similar products (excluding the product itself)
        top_5 = similar_scores.iloc[1:6]

        # Get their descriptions
        recommended_products = descriptions.loc[top_5.index]['Description'].tolist()
        return recommended_products
    except IndexError:
        return "Product not found. Please try another one."

# Example Usage
# print(get_recommendations('WHITE HANGING HEART T-LIGHT HOLDER', item_similarity_df, product_descriptions))