# CineIQ-by-Coding-Club-
## Local Setup & Installation

Because the training datasets and machine learning models are too large for GitHub, you will need to generate them locally before booting the server.

**1. Clone the repository**
`git clone https://github.com/yourusername/cineiq-hybrid-recommender.git`
`cd cineiq-hybrid-recommender`

**2. Download the Data**
* Download the [MovieLens Dataset](link_to_kaggle) and the [TMDB Dataset](link_to_kaggle).
* Extract the CSV files and place them directly into the `data/` folder.

**3. Train the Models**
Open the `models/` folder and run the Jupyter Notebooks (`collaborative_filtering.ipynb` and `content_filtering.ipynb`). This will read the CSVs from the data folder, train the AI, and automatically save the `.pkl` models into the `artifacts/` folder.

**4. Boot the Microservices**
Open two terminals.
Terminal 1 (The Brain): `cd backend && uvicorn backend_new:app --reload`
Terminal 2 (The Face): `cd frontend && streamlit run frontend_new.py`
