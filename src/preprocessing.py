"""
preprocessing.py

This module prepares data for collaborative filtering:
- User-aware train-test split
- Rating normalization (mean-centering)
- User-item matrix construction
"""

import pandas as pd
import numpy as np


def train_test_split_by_user(ratings, test_size=0.2, random_state=42):
    """
    Splits ratings into train and test sets on a per-user basis.

    This ensures that each user appears in both training and testing
    data, preventing cold-start leakage.

    Parameters:
        ratings (pd.DataFrame): Ratings data
        test_size (float): Fraction of data per user for testing
        random_state (int): Random seed

    Returns:
        train (pd.DataFrame)
        test (pd.DataFrame)
    """
    np.random.seed(random_state)

    train_list = []
    test_list = []

    for user_id, user_data in ratings.groupby("user_id"):
        if len(user_data) < 5:
            train_list.append(user_data)
            continue

        test_count = int(len(user_data) * test_size)
        test_indices = np.random.choice(
            user_data.index,
            size=test_count,
            replace=False
        )

        test_data = user_data.loc[test_indices]
        train_data = user_data.drop(test_indices)

        train_list.append(train_data)
        test_list.append(test_data)

    train = pd.concat(train_list)
    test = pd.concat(test_list)

    return train, test


def normalize_ratings(train_df):
    """
    Normalizes ratings by subtracting each user's mean rating.

    This removes user-specific rating bias.

    Parameters:
        train_df (pd.DataFrame): Training ratings data

    Returns:
        normalized_df (pd.DataFrame)
        user_means (pd.Series)
    """
    user_means = train_df.groupby("user_id")["rating"].mean()

    normalized_df = train_df.copy()
    normalized_df["rating_normalized"] = normalized_df.apply(
        lambda x: x["rating"] - user_means[x["user_id"]],
        axis=1
    )

    return normalized_df, user_means


def create_user_item_matrix(df, rating_col="rating"):
    """
    Creates a user-item interaction matrix.

    Parameters:
        df (pd.DataFrame): Ratings dataframe
        rating_col (str): Column to use as values

    Returns:
        pd.DataFrame: User-item matrix
    """
    matrix = df.pivot(
        index="user_id",
        columns="item_id",
        values=rating_col
    )
    return matrix
