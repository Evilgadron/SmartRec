import os
import sys

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data_loader import load_ratings, load_movies
from src.preprocessing import (
    train_test_split_by_user,
    normalize_ratings,
    create_user_item_matrix
)
from src.collaborative_filtering import compute_user_similarity
from src.recommender import recommend_items
from src.evaluation import precision_at_k
from src.collaborative_filtering import compute_item_similarity
from src.recommender import recommend_items_item_based


def main():
    print("Starting SmartRec pipeline...")

    # Load data
    ratings = load_ratings()
    movies = load_movies()

    # Preprocessing
    train_df, test_df = train_test_split_by_user(ratings)
    train_df, _ = normalize_ratings(train_df)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train_ratings.csv", index=False)
    test_df.to_csv("data/processed/test_ratings.csv", index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # Build user-item matrix
    user_item_matrix = create_user_item_matrix(train_df, rating_col="rating")

    # Compute similarity
    user_similarity = compute_user_similarity(user_item_matrix)

    # Generate recommendations for a sample user
    user_id = user_item_matrix.index[0]

    recommendations = recommend_items(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
        movies_df=movies,
        n_recommendations=5,
        k=5
    )

    print(f"\n🎬 Recommendations for User {user_id}:")
    print(recommendations)

    # Evaluation
    precision = precision_at_k(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        test_df=test_df,
        user_similarity=user_similarity,
        recommend_fn=recommend_items,
        movies_df=movies,
        k=5
    )

    print(f"\n📊 Precision@5 for User {user_id}: {precision}")

    # Item-based similarity
    item_similarity = compute_item_similarity(user_item_matrix)

    item_recs = recommend_items_item_based(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
        movies_df=movies,
        n_recommendations=5,
        k=5
    )

    print(f"\n🎬 Item-Based Recommendations for User {user_id}:")
    print(item_recs)

if __name__ == "__main__":
    main()
