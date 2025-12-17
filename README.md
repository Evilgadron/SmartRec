```
# 🎬 SmartRec — Intelligent Movie Recommendation System

🔗 **Live Application**  
👉 https://smartrec-nweylm9mbh6ig4esszcxjq.streamlit.app/

SmartRec is a **production-ready recommendation system** that generates personalized movie suggestions using **collaborative filtering** based on historical user behavior.  
The system learns implicit user preferences from interaction data and delivers **ranked, top-N recommendations** through an interactive web interface.

---

## 🚀 How to Use SmartRec

1. Open the live application using the link above.
2. In the sidebar:
   - Select a **User ID**.
   - Choose a recommendation strategy:
     - **User-Based Collaborative Filtering**
     - **Item-Based Collaborative Filtering**
   - Set the number of recommendations (**Top-N**).
3. Click **“Get Recommendations”**.
4. Instantly view a ranked list of recommended movies with predicted relevance scores.

The recommendations adapt dynamically based on the selected user and algorithm.

---

## 🧠 System Overview

SmartRec follows a clean, modular pipeline:

1. **Data Ingestion**
   - Loads historical user–movie interactions (ratings).
2. **Preprocessing**
   - User-aware train/test split to prevent data leakage.
   - Rating normalization to reduce user bias.
3. **Modeling**
   - User-Based Collaborative Filtering (k-NN + cosine similarity).
   - Item-Based Collaborative Filtering for improved stability.
4. **Recommendation Generation**
   - Predicts unseen item relevance.
   - Produces ranked top-N movie suggestions.
5. **Evaluation**
   - Measures recommendation quality using **Precision@K**.
6. **Deployment**
   - Interactive Streamlit web application.

---

## 🛠️ Tech Stack

### Language
- **Python**

### Libraries & Frameworks
- **Pandas** — Data manipulation and preprocessing  
- **NumPy** — Numerical computations  
- **Scikit-learn** — Similarity computation and ML utilities  
- **Streamlit** — Interactive web deployment  

### Machine Learning Techniques
- Collaborative Filtering
  - User-Based CF
  - Item-Based CF
- k-Nearest Neighbors (k-NN)
- Cosine Similarity
- Precision@K evaluation metric

### Dataset
- **MovieLens 100K Dataset**  
  https://grouplens.org/datasets/movielens/100k/

---

## 📊 Model Performance

- **Evaluation Metric:** Precision@K  
- **Baseline Result:**  
  - Precision@5 ≈ **0.40**

This indicates that, on average, **40% of the top-5 recommended movies are relevant** to the user based on unseen test interactions.

---

## ⚠️ Known Limitations

- **Cold-Start Problem**  
  New users or new movies without interaction history cannot be recommended effectively.

- **Data Sparsity**  
  User–item interaction matrices are highly sparse, impacting similarity reliability.

- **Scalability**  
  In-memory similarity computations may not scale efficiently to very large datasets.

- **Purely Behavior-Based**  
  No content features (genres, tags, descriptions) are currently used.

---

## 🔮 Future Enhancements

- Implement **Matrix Factorization (SVD)** for improved performance on sparse data.
- Build a **Hybrid Recommendation System** (Collaborative + Content-Based).
- Introduce **cold-start handling** using metadata or embeddings.
- Add recommendation **explanations** (“Why this movie?”).
- Improve UI with posters, genres, and filters.
- Optimize for large-scale datasets using approximate nearest neighbors.
- Extend recommendations beyond movies (music, products).

---

## 📁 Project Structure

```

SmartRec-Recommendation-System/
│
├── data/
│   ├── raw/                # MovieLens dataset
│   └── processed/          # Train/Test splits
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── collaborative_filtering.py
│   ├── recommender.py
│   ├── evaluation.py
│   └── run.py
│
├── app.py                  # Streamlit application
├── requirements.txt
└── README.md

```

---

## 👨‍💻 Author

Developed as a professional Machine Learning portfolio project to demonstrate end-to-end design, evaluation, and deployment of recommendation systems using collaborative filtering.
```

