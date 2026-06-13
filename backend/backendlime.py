import os
import joblib
import json
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

app = FastAPI(title="CineIQ Hybrid Engine")         

# 1. Dynamically locating the artifacts folder from inside the backend/ folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR) #  main CineIQ folder
ARTIFACTS_DIR = os.path.join(ROOT_DIR, 'artifacts')          

print("Loading...")

# 2. Loading Collaborative Model
svd_path = os.path.join(ARTIFACTS_DIR, 'svd_model.pkl')
svd_model = joblib.load(svd_path)

# 3. Loading Content Models
tfidf_matrix = joblib.load(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'))
movies_df = joblib.load(os.path.join(ARTIFACTS_DIR, 'movies_content_model.pkl'))
movie_indices = joblib.load(os.path.join(ARTIFACTS_DIR, 'movie_indices.pkl'))

# Creating a case-insensitive dictionary for search
lower_movie_indices = {str(title).lower(): idx for title, idx in movie_indices.items()}

# 4. Loading Sentiment Engine
print("Loading Sentiment Engine...")
sentiment_engine = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
    device=0 # Forces GPU execution!
)

# 5. Loading ID Translator
mappings_path = os.path.join(ARTIFACTS_DIR, 'mappings.json')
with open(mappings_path, 'r') as f:
    id_translator = json.load(f)

print("All models successfully")

#ENDPOINTS

@app.get("/")
def home():
    return {"status": "CineIQ Backend is Alive and Ready!"}

@app.get("/recommend")
def get_hybrid_recommendations(user_id: int, movie_title: str):
    search_title = movie_title.lower()

    if search_title not in lower_movie_indices:
        return {"error": f"Movie '{movie_title}' not found in the CineIQ database."}
    
    idx = lower_movie_indices[search_title]
    if hasattr(idx, 'iloc'):
        idx = idx.iloc[0]
    idx = int(idx)

    # Content Filtering (getting top 30 similar)
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_movie_indices = sim_scores.argsort()[-31:-1][::-1]

    # Optimization: Extracting data and batch process sentiment outside the loop
    overviews = [str(movies_df.iloc[i]['overview']) for i in similar_movie_indices]
    
    # Batch inference is significantly faster than a sequential for-loop
    sentiment_results = sentiment_engine(overviews)

    recommendations = []
    
    for dynamic_idx, i in enumerate(similar_movie_indices):
        sim_score = sim_scores[i]
        movie_id = movies_df.iloc[i]['id']
        title = movies_df.iloc[i]['title']

        # Collaborative Score Calculation
        translated_id = id_translator.get(str(movie_id), movie_id)
        try:
            translated_id = int(translated_id)
        except ValueError:
            pass

        svd_pred = svd_model.predict(uid=user_id, iid=translated_id).est
        normalized_svd = svd_pred / 5.0

        # Sentiment Analysis from Batch Results
        hf_result = sentiment_results[dynamic_idx]
        sentiment_math = 1.0 if hf_result['label'] == 'POSITIVE' else -1.0
        sentiment_multiplier = sentiment_math * hf_result['score']

       # Isolated Feature Weights for White-Box LIME
        weight_content = float(sim_score * 0.4)
        weight_collab = float(normalized_svd * 0.4)
        weight_sentiment = float(sentiment_multiplier * 0.2)
        
        # Hybrid Mixer Math
        hybrid_score = weight_content + weight_collab + weight_sentiment

        recommendations.append({
            "title": title,
            "content_similarity": round(float(sim_score), 3),
            "predicted_user_rating": round(float(svd_pred), 2),
            "sentiment_label": hf_result['label'],
            "final_hybrid_score": round(float(hybrid_score), 3),
            
            # THE NEW EXPLAINABILITY DATA LAYER
            "feature_weights": [
                {"name": "Content Similarity", "weight": round(weight_content, 3)},
                {"name": "Collaborative (Taste)", "weight": round(weight_collab, 3)},
                {"name": f"Sentiment ({hf_result['label']})", "weight": round(weight_sentiment, 3)}
            ]
        })

    # Sorting and slicing top 5
    recommendations.sort(key=lambda x: x["final_hybrid_score"], reverse=True)

    return {
        "user_id": user_id,
        "target_movie": movie_title,
        "recommendations": recommendations[:5]
    }