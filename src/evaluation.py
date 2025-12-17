"""evaluation.py

Placeholder module for evaluation metrics and helpers.
"""

import numpy as np


def precision_at_k(
    user_id,
    user_item_matrix,
    test_df,
    user_similarity,
    recommend_fn,
    movies_df,
    k=5
):
    """
    Computes Precision@K for a single user.
    """

    # Items the user actually interacted with in test set
    relevant_items = set(
        test_df[test_df["user_id"] == user_id]["item_id"]
    )

    if not relevant_items:
        return None

    # Get recommendations
    recommendations = recommend_fn(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
        movies_df=movies_df,
        n_recommendations=k
    )

    recommended_items = set(recommendations["item_id"])

    # Precision@K
    true_positives = len(recommended_items & relevant_items)
    precision = true_positives / k

    return precision
