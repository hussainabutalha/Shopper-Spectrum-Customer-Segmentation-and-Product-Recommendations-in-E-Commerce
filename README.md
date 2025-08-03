🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations
This project analyzes e-commerce transaction data to segment customers using RFM analysis and K-Means clustering, and builds a product recommendation system using item-based collaborative filtering. The insights are delivered through an interactive Streamlit web application.

🌟 Features
The project is delivered as an interactive web application with two main features:

1. 🎯 Product Recommendation Module

Objective: Recommend similar products to a user based on collaborative filtering.

Functionality:

A dropdown menu to select a product from the dataset.

A "Get Recommendations" button that displays the top 5 most similar products.

2. 🔍 Customer Segmentation Module

Objective: Predict the segment of a customer based on their purchasing behavior.

Functionality:

Number inputs for Recency (days since last purchase), Frequency (number of purchases), and Monetary (total spend).

A "Predict Cluster" button that displays the predicted customer segment (e.g., High-Value, At-Risk, etc.).

🛠️ Tech Stack & Libraries
Python 3.8+

Pandas & NumPy: For data manipulation and numerical operations.

Scikit-learn: For K-Means clustering, data scaling, and cosine similarity.

Streamlit: For building the interactive web application.

Matplotlib: For data visualization in the analysis phase.

Jupyter Notebook: For the data analysis and model building workflow.

⚙️ Setup and Installation
Follow these steps to set up the project environment and run the application locally.

1. Prerequisites
Ensure you have Python 3.8 or a newer version installed.

2. Clone the Repository
Clone this repository to your local machine using git:

git clone [https://github.com/YOUR_USERNAME/shopper-spectrum.git]((https://github.com/hussainabutalha/Shopper-Spectrum-Customer-Segmentation-and-Product-Recommendations-in-E-Commerce/tree/main))
cd shopper-spectrum

(Replace YOUR_USERNAME with your actual GitHub username)

3. Create a Virtual Environment (Recommended)
It's a good practice to create a virtual environment to manage project-specific dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

4. Install Dependencies
Install all the required libraries using the requirements.txt file.

pip install -r requirements.txt

🚀 Running the Application
Once the setup is complete, you can run the Streamlit application.

Navigate to the project directory in your terminal.

Run the following command:

streamlit run app.py

The application will open automatically in your default web browser.

📂 Project Structure
.
├── 📄 app.py                    # The main Streamlit application script
├── 📓 Shopper_Spectrum_Analysis.ipynb # Jupyter Notebook with data cleaning, EDA, and model building
├── 📄 requirements.txt          # List of Python dependencies for the project
├── 📦 models/                    # Directory for saved models and data
│   ├── kmeans_model.pkl          # Saved K-Means clustering model
│   ├── scaler.pkl                # Saved StandardScaler object
│   ├── item_similarity_df.pkl    # Saved item-item similarity matrix
│   └── product_descriptions.pkl  # Saved product descriptions
└── 📄 README.md                 # This file

📊 Dataset
This project uses the Online Retail II UCI dataset, which contains transactional data for a UK-based online retail company from 2009 to 2011.
