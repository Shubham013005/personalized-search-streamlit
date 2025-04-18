# personalized_search_app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dummy product catalog
products = [
    {"title": "Wireless Headphones", "description": "Bluetooth, noise-cancelling over-ear headphones"},
    {"title": "Gaming Mouse", "description": "RGB wired gaming mouse with customizable buttons"},
    {"title": "Yoga Mat", "description": "Eco-friendly non-slip yoga mat for workouts"},
    {"title": "Smartwatch", "description": "Fitness tracking smartwatch with heart rate monitor"},
    {"title": "Electric Kettle", "description": "1.5L stainless steel fast-boil electric kettle"},
    {"title": "Running Shoes", "description": "Breathable lightweight shoes for running and walking"},
]

# User profile (interest tags from previous behavior)
user_profile = "fitness, workout, health, smart devices, yoga"

# Prepare product data
df = pd.DataFrame(products)

# Combine product title + description
df["text"] = df["title"] + " " + df["description"]

# UI
st.title("🛒 Personalized E-commerce Search")
query = st.text_input("Search for a product:")

if query:
    # Combine user profile + query
    combined_input = [user_profile, query] + df["text"].tolist()

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(combined_input)

    # Calculate cosine similarity between query+profile and products
    profile_query_vector = vectors[0] + vectors[1]
    product_vectors = vectors[2:]
    scores = cosine_similarity(profile_query_vector, product_vectors).flatten()

    # Sort products by similarity score
    df["score"] = scores
    results = df.sort_values(by="score", ascending=False)

    st.subheader("🔍 Personalized Results")
    for idx, row in results.iterrows():
        st.write(f"**{row['title']}**")
        st.write(f"{row['description']}")
        st.write(f"Relevance Score: `{row['score']:.2f}`")
        st.markdown("---")
