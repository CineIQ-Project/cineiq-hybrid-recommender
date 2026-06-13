# CineIQ

A hybrid movie recommendation system that integrates content-based filtering and collaborative filtering to generate personalized film suggestions. The system leverages metadata features (genres, cast, crew, keywords) alongside user-rating interaction patterns to produce recommendations that balance item similarity with latent preference modeling.

## Project Overview

CineIQ implements a two-pronged recommendation strategy. The content-based filtering pipeline constructs item feature vectors from TMDB movie metadata -- including genre taxonomies, cast and crew attributions, and keyword descriptors -- and computes pairwise similarity scores to surface thematically related titles. The collaborative filtering pipeline operates on the MovieLens ratings corpus, decomposing the sparse user-item interaction matrix to identify latent factors that capture implicit preference structures across the user population. The hybrid architecture enables the system to mitigate the cold-start limitations inherent to purely collaborative approaches while retaining the serendipity and personalization that content-only methods lack.

## Repository Architecture

```
.
├── app.py                                  # Streamlit/Gradio application entry point
├── backend.py                              # Core recommendation logic and API layer
├── collaborative_filtering(updated).ipynb  # Collaborative filtering model development
├── content_filtering_tmdb.ipynb            # Content-based filtering model development
├── mlflow.db                               # MLflow experiment tracking database (SQLite)
├── requirements.txt                        # Python dependency manifest
└── Datasets/
    ├── credits.csv
    ├── genome-scores.csv
    ├── genome-tags.csv
    ├── IMDB Dataset.csv
    ├── keywords.csv
    ├── links.csv
    ├── movies.csv
    ├── movies_metadata.csv
    ├── ratings.csv
    ├── tags.csv
    ├── tmdb_5000_credits.csv
    └── tmdb_5000_movies.csv
```

| File | Role |
|------|------|
| `app.py` | Front-end application runner. Serves the user-facing recommendation interface. |
| `backend.py` | Backend API and orchestration layer. Handles data loading, model inference, and response formatting. |
| `collaborative_filtering(updated).ipynb` | Notebook for training and evaluating the collaborative filtering model. Contains matrix factorization experiments, hyperparameter sweeps, and evaluation metrics. |
| `content_filtering_tmdb.ipynb` | Notebook for building the content-based filtering pipeline. Covers feature extraction from TMDB metadata, similarity computation, and qualitative evaluation. |
| `mlflow.db` | Persistent SQLite store for MLflow experiment runs, parameters, metrics, and artifact references. |
| `requirements.txt` | Pinned dependencies for reproducible environment setup. |

## Datasets Used

The system draws from two primary data sources:

- **TMDB (The Movie Database):** `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` provide structured metadata including genres, keywords, cast, crew, budget, revenue, and production details. Supplementary files `movies_metadata.csv`, `credits.csv`, and `keywords.csv` extend coverage across the broader TMDB catalog.

- **MovieLens:** `ratings.csv`, `tags.csv`, `genome-scores.csv`, `genome-tags.csv`, `movies.csv`, and `links.csv` constitute the MovieLens dataset, providing explicit user ratings, free-text tags, and tag-genome relevance scores. The `links.csv` file maps MovieLens identifiers to TMDB and IMDB identifiers, enabling cross-dataset entity resolution.

- **IMDB:** `IMDB Dataset.csv` supplements the above with additional film metadata from the IMDB catalog.

## Experiment Tracking

Model training and evaluation experiments are tracked via [MLflow](https://mlflow.org/). The `mlflow.db` SQLite database persists all experiment metadata, including:

- Hyperparameter configurations per run
- Evaluation metrics (RMSE, precision, recall, etc.)
- Model versioning and artifact lineage

To launch the MLflow tracking UI against the local database:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The UI will be accessible at `http://127.0.0.1:5000` by default.

## Local Setup and Installation

**Prerequisites:** Python 3.8 or higher.

1. Clone the repository:

```bash
git clone https://github.com/<username>/CineIQ.git
cd CineIQ
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

**Run the application front-end:**

```bash
python app.py
```

**Run the backend service independently:**

```bash
python backend.py
```

**Reproduce model training:**

Open the Jupyter notebooks for interactive execution:

```bash
jupyter notebook collaborative_filtering(updated).ipynb
jupyter notebook content_filtering_tmdb.ipynb
```

## License

This project is provided as-is for educational and research purposes.
