# CineIQ
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3b8eaefe-42ba-4b1d-8ded-3d217d7c11ee" />

A hybrid movie recommendation system that integrates content-based filtering and collaborative filtering to generate personalized film suggestions. The system leverages metadata features (genres, cast, crew, keywords) alongside user-rating interaction patterns to produce recommendations that balance item similarity with latent preference modeling.

## Project Overview
### Project Demo
**[Click here to watch the CineIQ feature demonstration](https://drive.google.com/file/d/1LM5s3FCkCQjnhCLARdv6TkJUFGIzuCzE/view?usp=sharing)** 


CineIQ implements a two-pronged recommendation strategy. The content-based filtering pipeline constructs item feature vectors from TMDB movie metadata and computes pairwise similarity scores to surface thematically related titles, effectively solving the "Cold Start" problem. The collaborative filtering pipeline operates on the MovieLens ratings corpus, utilizing Singular Value Decomposition (SVD) to identify latent factors that capture implicit preference structures. 

This hybrid architecture mitigates the limitations inherent to purely collaborative approaches while retaining the serendipity and personalization that content-only methods lack.

## Repository Architecture

Our pipeline is organized into a modular, production-ready microservice structure:

```text
CineIQ/
├── data/                  # REQUIRED: Place your raw CSV datasets here (git-ignored)
├── artifacts/             # Serialized .pkl models and matrices (git-ignored)
├── models/                # ML Pipeline & Model Training
│   ├── src/               # Custom Python modules (data_loader, config, logger, etc.)
│   ├── collaborative_filtering.ipynb  # SVD Matrix Factorization
│   └── content_filtering.ipynb            # NLP & Cosine Similarity
├── backend/               # Core recommendation API layer (FastAPI)
├── frontend/              # User-facing application (Streamlit)
├── .gitignore             # Strict ignore rules for large datasets and MLflow logs
└── requirements.txt       # Python dependency manifest
```

## Datasets Required

To run this pipeline, you must acquire the following datasets from TMDB and MovieLens. 

**Important:** You must place these exactly inside the `CineIQ/data/` directory. They are explicitly ignored by Git due to size constraints.

1. `movies_metadata.csv` (TMDB)
2. `credits.csv` (TMDB)
3. `keywords.csv` (TMDB)
4. `ratings.csv` (MovieLens)
5. `movies.csv` (MovieLens)
6. `links.csv` (MovieLens)

## Experiment Tracking

Model training and evaluation experiments are tracked via a custom `ExperimentLogger` utilizing **MLflow**. The system tracks hyperparameter configurations per run and evaluation metrics (RMSE, MAE, Precision@K). Note: To prevent repository bloat, heavy model artifacts are serialized directly to `/artifacts`, while lightweight metrics are logged locally.

## Local Setup and Installation

**Prerequisites:** Python 3.8+

1. **Clone the repository:**
```bash
git clone https://github.com/CineIQ-Project/cineiq-hybrid-recommender.git
cd cineiq-hybrid-recommender
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Prepare Data:** Download the 6 required CSV files listed above and place them into the `CineIQ/data/` folder.

## Usage & Execution

### 1. Train the Models
To generate the `.pkl` files needed for the recommendation engine, navigate to the models directory and execute the notebooks:
```bash
cd models
jupyter notebook content_filtering.ipynb
jupyter notebook collaborative_filtering.ipynb
```
*(Verify that 5 `.pkl` files are successfully generated inside the `/artifacts` folder before proceeding.)*

### 2. Run the Backend API Service
Boot up the core recommendation logic layer:
```bash
cd backend
uvicorn backendlime:app --reload
```
*(The API will be accessible at http://localhost:8000)*

### 3. Run the Frontend Interface
In a separate terminal, launch the user interface:
```bash
cd frontend
streamlit run frontendlime.py
```

