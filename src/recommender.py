import pandas as pd
from src.collaborative_filtering import (
    predict_rating_user_based,
    predict_rating_item_based
)

# =========================
# USER-BASED RECOMMENDER
# =========================

def recommend_items(
    user_id,
    user_item_matrix,
    user_similarity,
    movies_df,
    n_recommendations=5,
    k=5
):
    """
    Generate top-N recommendations using user-based collaborative filtering.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError("User ID not found")

    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index

    predictions = []

    for item_id in unrated_items:
        pred = predict_rating_user_based(
            user_id,
            item_id,
            user_item_matrix,
            user_similarity,
            k
        )
        if pred is not None:
            predictions.append((item_id, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_items = predictions[:n_recommendations]

    recommendations = pd.DataFrame(
        top_items, columns=["item_id", "predicted_rating"]
    ).merge(movies_df, on="item_id")

    return recommendations


# =========================
# ITEM-BASED RECOMMENDER
# =========================

def recommend_items_item_based(
    user_id,
    user_item_matrix,
    item_similarity,
    movies_df,
    n_recommendations=5,
    k=5
):
    """
    Generate top-N recommendations using item-based collaborative filtering.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError("User ID not found")

    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index

    predictions = []

    for item_id in unrated_items:
        pred = predict_rating_item_based(
            user_id,
            item_id,
            user_item_matrix,
            item_similarity,
            k
        )
        if pred is not None:
            predictions.append((item_id, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_items = predictions[:n_recommendations]

    recommendations = pd.DataFrame(
        top_items, columns=["item_id", "predicted_rating"]
    ).merge(movies_df, on="item_id")

    return recommendations
