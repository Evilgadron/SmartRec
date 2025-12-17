"""data_loader.py

Responsible for loading raw data in a clean, reusable way.
"""
import os
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_ratings():
    path = os.path.join(PROJECT_ROOT, "data", "raw", "u.data")

    ratings = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    return ratings


def load_movies():
    path = os.path.join(PROJECT_ROOT, "data", "raw", "u.item")

    movies = pd.read_csv(
        path,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "movie_title"]
    )
    return movies
