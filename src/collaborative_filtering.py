"""collaborative_filtering.py

This file handles similarity computation and rating prediction."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# USER-BASED COLLABORATIVE FILTERING
# =========================

def compute_user_similarity(user_item_matrix):
    """
    Computes cosine similarity between users.
    """
    matrix_filled = user_item_matrix.fillna(0)

    similarity = cosine_similarity(matrix_filled)

    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    return similarity_df


def predict_rating_user_based(user_id, item_id, user_item_matrix, user_similarity, k=5):
    """
    Predict rating using user-based collaborative filtering.
    """
    if item_id not in user_item_matrix.columns:
        return None

    item_ratings = user_item_matrix[item_id].dropna()
    if item_ratings.empty:
        return None

    sim_scores = user_similarity.loc[user_id, item_ratings.index]
    top_k_users = sim_scores.sort_values(ascending=False).head(k)

    ratings = item_ratings.loc[top_k_users.index]

    predicted_rating = np.dot(top_k_users, ratings) / np.sum(top_k_users)

    return predicted_rating


# =========================
# ITEM-BASED COLLABORATIVE FILTERING
# =========================

def compute_item_similarity(user_item_matrix):
    """
    Computes cosine similarity between items.
    """
    item_user_matrix = user_item_matrix.T.fillna(0)

    similarity = cosine_similarity(item_user_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=item_user_matrix.index,
        columns=item_user_matrix.index
    )

    return similarity_df


def predict_rating_item_based(user_id, item_id, user_item_matrix, item_similarity, k=5):
    """
    Predict rating using item-based collaborative filtering.
    """
    if item_id not in item_similarity.columns:
        return None

    user_ratings = user_item_matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return None

    sim_scores = item_similarity.loc[item_id, user_ratings.index]
    top_k_items = sim_scores.sort_values(ascending=False).head(k)

    ratings = user_ratings.loc[top_k_items.index]

    predicted_rating = np.dot(top_k_items, ratings) / np.sum(top_k_items)

    return predicted_rating
