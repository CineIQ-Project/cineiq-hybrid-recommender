# CineIQ-by-Coding-Club-
<br>
<br>
<br>
PROBLEM STATEMENT
<br>
Content discovery on modern streaming platforms is opaque, biased toward promoted titles, and traps users in recommendation loops. There is a need for an open, explainable movie recommendation engine that combines multiple ML strategies to deliver personalized, interpretable suggestions that evolve with user taste over time.
<br>
DELIVERABLES
<br>
• Hybrid Recommendation Engine: Combines collaborative filtering, content- based filtering (TF-IDF + cosine similarity), and SVD-based matrix factorization via a weighted ensemble
<br>
• Sentiment-Aware Re-Ranker: Uses VADER/DistilBERT on user reviews to re-rank recommendations based on real audience reception signals
<br>
• User Taste Dashboard: Streamlit interface visualizing genre radar charts, decade preferences, and director/actor affinities from rating history
<br>
• Explainability Layer: Every recommendation surfaces a human-readable reason using LIME or rule-based templates
<br>
DATASETS:
<br>
• MovieLens 25M-grouplens.org/datasets/movielens/25m
<br>
• TMDB Metadata (Kaggle) — cast, genres, keywords for 45K movies
<br>
• IMDB 50K Reviews (Kaggle) — for sentiment model training
<br>
TECH STACK:
<br>
• ML: Python, scikit-learn, Surprise (SVD), Pandas, NumPy
<br>
• NLP: VADER / HuggingFace DistilBERT
<br>
• Serving: FastAPI (/recommend and /similar endpoints) Dashboard: Streamlit + Plotly
<br>
• Tracking: MLflow for experiment logging
