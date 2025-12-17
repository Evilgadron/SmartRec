# 🎬 SmartRec — Intelligent Movie Recommendation System
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Recommendation_System-green)
![Collaborative Filtering](https://img.shields.io/badge/Collaborative_Filtering-kNN-orange)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

🔗 **Live Demo**  
👉 https://smartrec-nweylm9mbh6ig4esszcxjq.streamlit.app/

SmartRec is a **production-ready machine learning recommendation system** that delivers **personalized movie recommendations** using **collaborative filtering techniques**.  
It learns user preferences from historical interaction data and generates **ranked Top-N movie suggestions** via an interactive web interface.

> 🔍 **Keywords:** Movie Recommendation System, Collaborative Filtering, k-NN, Cosine Similarity, Streamlit ML App, Precision@K, MovieLens Dataset

---

## 🚀 Features

- Personalized movie recommendations
- User-Based & Item-Based Collaborative Filtering
- Real-time interactive UI with Streamlit
- Ranked Top-N recommendations with relevance scores
- Offline model evaluation using Precision@K
- Clean, modular, production-style codebase

---

## 🧠 System Architecture

SmartRec follows a **modular end-to-end recommendation pipeline**:

### 🔹 Architecture Flow

User Interaction (Streamlit UI)
↓
User Selection (User ID + Algorithm + Top-N)
↓
Preprocessed Ratings Matrix
↓
Similarity Computation (Cosine Similarity)
↓
k-NN Collaborative Filtering Model
↓
Prediction of Unseen Movies
↓
Top-N Ranked Recommendations

### 🔹 Component Breakdown

1. **Data Ingestion**
   - Loads MovieLens 100K user–movie ratings.

2. **Preprocessing Layer**
   - User-aware train/test split to avoid data leakage.
   - Rating normalization to reduce individual rating bias.

3. **Modeling Layer**
   - User-Based CF using k-NN + cosine similarity.
   - Item-Based CF for improved stability and sparsity handling.

4. **Recommendation Engine**
   - Predicts relevance scores for unseen items.
   - Generates ranked Top-N movie recommendations.

5. **Evaluation Layer**
   - Uses Precision@K to measure recommendation quality.

6. **Presentation Layer**
   - Streamlit-based interactive web interface.

---

## 🛠️ Tech Stack

### Language
- **Python**

### Libraries & Tools
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Similarity computation & ML utilities
- **Streamlit** — Web application deployment

### ML Techniques
- Collaborative Filtering (User-Based & Item-Based)
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
  - **Precision@5 ≈ 0.40**

📌 On average, **40% of the Top-5 recommended movies are relevant**, based on unseen test interactions.

---

## ⚠️ Limitations

- **Cold-Start Problem**
  - New users or movies without history cannot be recommended accurately.

- **Data Sparsity**
  - Sparse interaction matrices affect similarity quality.

- **Scalability**
  - In-memory similarity computation may not scale to very large datasets.

- **Behavior-Only Model**
  - Content features (genres, tags, descriptions) are not yet used.

---

## 🔮 Future Improvements

- Implement **Matrix Factorization (SVD)** for sparse data handling.
- Build a **Hybrid Recommendation System** (Collaborative + Content-Based).
- Add **cold-start solutions** using metadata or embeddings.
- Provide **recommendation explanations** (“Why this movie?”).
- Enhance UI with posters, genres, and filters.
- Optimize scalability using approximate nearest neighbors.
- Extend system to recommend music, products, or courses.

---

## 📁 Project Structure

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


## 👨‍💻 Author

Developed as a **machine learning portfolio project** demonstrating:
- Recommendation system design
- Collaborative filtering algorithms
- Model evaluation techniques
- End-to-end ML deployment using Streamlit



