import os
import joblib
from fastapi import FastAPI
from surprise import dump

# Initialize the Server
app = FastAPI(title="CineIQ Hybrid Engine")

# 1. Dynamically locate the artifacts folder
CURRENT_DIR = os.getcwd()
ARTIFACTS_DIR = os.path.join(CURRENT_DIR, 'artifacts')

print("Waking up CineIQ models... This might take a few seconds.")

# 2. Waking up the Collaborative Model
svd_path = os.path.join(ARTIFACTS_DIR, 'svd_model.pkl')
svd_model = joblib.load(svd_path)

# 3. Waking up the Content Models
tfidf_matrix = joblib.load(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'))
movies_df = joblib.load(os.path.join(ARTIFACTS_DIR, 'movies_content_model.pkl'))
movie_indices = joblib.load(os.path.join(ARTIFACTS_DIR, 'movie_indices.pkl'))

# (Sentiment Analysis will be imported here later!)

print("All models successfully loaded into RAM!")

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "CineIQ Backend is Alive and Ready!"}

@app.get("/recommend")
def test_hybrid_route(user_id: int, movie_title: str):
    """
    This is a test endpoint. Tomorrow we will put the actual Hybrid Math here!
    """
    return {
        "message": f"Request received for User {user_id} and Movie: {movie_title}",
        "collaborative_engine_status": "Online",
        "content_engine_status": "Online"
    }