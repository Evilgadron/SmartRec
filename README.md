# SmartRec – Personalized Recommendation Engine

## Overview
SmartRec is a collaborative filtering–based recommendation system that suggests personalized items to users using past interaction data.

## Techniques Used
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Matrix Factorization (SVD)

## Dataset
MovieLens 100K Dataset

## Tech Stack
- Python
- Scikit-learn
- Pandas
- Streamlit

## Results
- User-based collaborative filtering
- Precision@5: 0.4
- Evaluated on held-out test data


## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
