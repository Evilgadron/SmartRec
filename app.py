import os
import sys
import streamlit as st

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src.data_loader import load_ratings, load_movies
from src.preprocessing import train_test_split_by_user, normalize_ratings, create_user_item_matrix
from src.collaborative_filtering import (
    compute_user_similarity,
    compute_item_similarity
)
from src.recommender import (
    recommend_items,
    recommend_items_item_based
)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="SmartRec", layout="centered")

st.title("🎬 SmartRec – Personalized Recommendation System")
st.write("Get movie recommendations using collaborative filtering.")

# -----------------------------
# Load & Prepare Data (Cached)
# -----------------------------
@st.cache_data
def load_and_prepare():
    ratings = load_ratings()
    movies = load_movies()

    train_df, _ = train_test_split_by_user(ratings)
    train_df, _ = normalize_ratings(train_df)

    user_item_matrix = create_user_item_matrix(train_df, rating_col="rating")

    user_similarity = compute_user_similarity(user_item_matrix)
    item_similarity = compute_item_similarity(user_item_matrix)

    return user_item_matrix, user_similarity, item_similarity, movies

user_item_matrix, user_similarity, item_similarity, movies = load_and_prepare()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Settings")

user_id = st.sidebar.selectbox(
    "Select User ID",
    user_item_matrix.index.tolist()
)

model_type = st.sidebar.radio(
    "Recommendation Method",
    ["User-Based Collaborative Filtering", "Item-Based Collaborative Filtering"]
)

top_n = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

# -----------------------------
# Generate Recommendations
# -----------------------------
if st.sidebar.button("Get Recommendations"):
    st.subheader(f"🎥 Recommendations for User {user_id}")

    if model_type == "User-Based Collaborative Filtering":
        recs = recommend_items(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            user_similarity=user_similarity,
            movies_df=movies,
            n_recommendations=top_n,
            k=5
        )
    else:
        recs = recommend_items_item_based(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            item_similarity=item_similarity,
            movies_df=movies,
            n_recommendations=top_n,
            k=5
        )

    st.dataframe(
        recs[["movie_title", "predicted_rating"]],
        use_container_width=True
    )
