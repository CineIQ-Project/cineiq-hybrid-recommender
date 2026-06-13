"""
CineIQ — User Rating Profile Dashboard
Streamlit app that visualises a specific user's rating history
from the Hybrid Movie Recommender System.

Data pipeline:
  ratings.csv  ──► user's highly-rated movieIds
  links.csv    ──► movieId → tmdbId
  movies_content_model.pkl ──► tmdbId → title, genres, year, cast, director
"""

import os
import json
import pickle
import warnings
import wordninja
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys
import time
import time
import random

current_directory = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(current_directory) # Step up to main folder

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import load_ratings, load_movies
from src.config import Config

def generate_lime_html(features: list, max_weight: float = 0.4) -> str:
    """Generates a lightweight HTML/CSS visual breakdown of feature weights."""
    bars_html = ""
    for f in features:
        weight = f["weight"]
        # Calculating visual percentage relative to max possible weight
        width_pct = min(abs(weight) / max_weight * 100, 100)
        
        # Red for negative (toxic sentiment), Green/Blue for positive
        bar_color = "#ef4444" if weight < 0 else "linear-gradient(to right, #0ba360, #3cba92)"
        sign = "+" if weight >= 0 else ""
        
        bars_html += f"""
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 140px; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">{f['name']}</div>
            <div style="flex-grow: 1; background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px; margin: 0 10px; overflow: hidden;">
                <div style="width: {width_pct}%; background: {bar_color}; height: 100%; border-radius: 3px;"></div>
            </div>
            <div style="min-width: 60px; white-space: nowrap; text-align: right; color: #cbd5e1; font-size: 0.85rem; font-family: monospace;">{sign}{weight:.3f}</div>
        </div>
        """
    return f'<div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">{bars_html}</div>'

def get_dynamic_theme(user_identifier):
    """Generates a consistent, glowing theme based on the user ID or Mock string."""
    random.seed(str(user_identifier))
    themes = [
        # Cyber Pink
        {"bg_glow": "rgba(236,72,153,0.15)", "card_border": "rgba(236,72,153,0.4)", 
         "text_gradient": "linear-gradient(to right, #ff0844, #ffb199)", 
         "primary": "#ff0844", "secondary": "#ffb199",
         "subtitle_color": "#fbcfe8",
         "sidebar_bg": "linear-gradient(180deg, #12050e 0%, #3a0b24 100%)"}, 
         
        # Matrix Green
        {"bg_glow": "rgba(16,185,129,0.15)", "card_border": "rgba(16,185,129,0.4)", 
         "text_gradient": "linear-gradient(to right, #0ba360, #3cba92)", 
         "primary": "#0ba360", "secondary": "#3cba92",
         "subtitle_color": "lightgreen",
         "sidebar_bg": "linear-gradient(180deg, #04120a 0%, #0a3822 100%)"}, 
         
        # Deep Space Violet 
        {"bg_glow": "rgba(139,92,246,0.15)", "card_border": "rgba(139,92,246,0.4)", 
         "text_gradient": "linear-gradient(to right, #8e2de2, #4a00e0)", 
         "primary": "#8e2de2", "secondary": "#4a00e0",
         "subtitle_color": "#d8b4fe",
         "sidebar_bg": "linear-gradient(180deg, #0f0c29 0%, #302b63 100%)"}, 
         
        # Neon Ocean Blue
        {"bg_glow": "rgba(6,182,212,0.15)", "card_border": "rgba(6,182,212,0.4)", 
         "text_gradient": "linear-gradient(to right, #00c6ff, #0072ff)", 
         "primary": "#00c6ff", "secondary": "#0072ff",
         "subtitle_color": "#bae6fd",
         "sidebar_bg": "linear-gradient(180deg, #05101a 0%, #0b344a 100%)"}, 
    ]
    return random.choice(themes)

#1 PAGE CONFIG
st.set_page_config(
    page_title="CineIQ · User Profile Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_glassmorphism(theme):
    """Injects Apple-style frosted glass and Three.js-style bloom effects."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
        background-color: #030305;
        background-image: radial-gradient(circle at 50% -20%, {theme['bg_glow']}, transparent 60%);
        background-attachment: fixed;
    }}
    
    /* ── Color-Matched Sidebar ── */
    [data-testid="stSidebar"] {{
        background: {theme['sidebar_bg']} !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 5px 0 15px rgba(0,0,0,0.5);
    }}
    
    /* ── Floating Glowing Metric Cards ── */
    [data-testid="stMetric"] {{
        background: rgba(20, 20, 25, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.05);
        border-top: 1px solid {theme['card_border']};
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px {theme['bg_glow']};
        border-color: {theme['card_border']};
    }}

    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        background: {theme['text_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        filter: drop-shadow(0px 0px 12px {theme['bg_glow']});
    }}
    
    /* ── Hero Banner (Permanent Glow Upgrade) ── */
    .hero {{
        background: rgba(30, 30, 35, 0.4);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-top: 1px solid {theme['card_border']};
        border-radius: 24px;
        padding: 40px 50px;
        margin-bottom: 2.5rem;
        /* 1. Permanent ambient glow for the container box */
        box-shadow: 0 0 30px {theme['bg_glow']}, inset 0 0 0 1px rgba(255,255,255,0.02);
        transition: all 0.4s ease;
    }}
    .hero h1 {{
        margin: 0 0 10px 0;
        font-size: 3rem;
        background: {theme['text_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        /* 2. Stacked drop-shadows for a massive, permanent text glow */
        filter: drop-shadow(0px 0px 8px rgba(255,255,255,0.2)) drop-shadow(0px 0px 20px {theme['primary']});
    }}

    .hero h1 {{
        margin: 0 0 10px 0;
        font-size: 3rem;
        background: {theme['text_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        filter: drop-shadow(0px 0px 15px {theme['bg_glow']});
    }}

    .hero p {{
        color: {theme['subtitle_color']};
        font-size: 1.15rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-shadow: 0px 0px 8px {theme['bg_glow']};
        margin: 0;
    }}

    /* ── Plotly Chart Outer Glow ── */
    .stPlotlyChart {{
        /* Drops a glowing shadow precisely around the painted SVG elements */
        filter: drop-shadow(0px 0px 15px {theme['bg_glow']});
    }}

    /* ── Full Recommendation Cards ── */
    .rec-card {{
        background: rgba(20, 20, 25, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {theme['card_border']};
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 25px;
        /* The aggressive ambient glow */
        box-shadow: 0 0 25px {theme['bg_glow']}, inset 0 0 15px rgba(0,0,0,0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}
    .rec-card:hover {{
        transform: translateY(-5px);
        /* Glow intensifies on hover */
        box-shadow: 0 0 40px {theme['bg_glow']}, inset 0 0 15px rgba(0,0,0,0.6);
        border-color: {theme['primary']};
    }}
    .rec-content {{
        flex: 3;
        padding-right: 20px;
    }}
    .rec-stats {{
        flex: 1;
        text-align: right;
        border-left: 1px solid rgba(255,255,255,0.1);
        padding-left: 30px;
    }}
    .rec-title {{
        background: {theme['text_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin: 0 0 12px 0;
        font-size: 2rem;
        filter: drop-shadow(0px 0px 10px {theme['bg_glow']});
    }}
    .rec-explanation {{
        color: #cbd5e1;
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.5;
    }}
    .rec-score-label {{
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .rec-score-value {{
        background: {theme['text_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.5rem;
        line-height: 1.1;
        margin-bottom: 12px;
        filter: drop-shadow(0px 0px 15px {theme['bg_glow']});
    }}
    .rec-mini-stat {{
        color: #e2e8f0;
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 500;
    }}


    </style>
    """, unsafe_allow_html=True)
   

# 2.  PATHS
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# Pointing to the data/ folder
RATINGS_PATH = os.path.join(ROOT_DIR, "data", "ratings.csv")
LINKS_PATH = os.path.join(ROOT_DIR, "data", "links.csv")


# 3.  CACHED DATA LOADERS

@st.cache_resource(show_spinner="Loading content model …")
def load_content_model() -> pd.DataFrame:
    """Load the pickled TMDB content DataFrame."""
    path = os.path.join(ARTIFACTS_DIR, "movies_content_model.pkl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


@st.cache_resource(show_spinner="Loading ID mappings …")
def load_mappings() -> dict:
    """Load the user/item SVD mapping file."""
    path = os.path.join(ARTIFACTS_DIR, "mappings.json")
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading links (MovieLens → TMDB) …")
def load_links() -> pd.DataFrame:
    """Load links.csv that maps MovieLens movieId → tmdbId."""
    df = pd.read_csv(
        LINKS_PATH,
        dtype={"movieId": "int32", "imdbId": "str", "tmdbId": "float32"},
    )
    df["tmdbId"] = df["tmdbId"].dropna().astype(int)
    return df


@st.cache_data(show_spinner="Loading ratings (this may take a moment) …")
def load_ratings() -> pd.DataFrame:
    """Load ratings.csv with memory-efficient dtypes."""
    return pd.read_csv(
        RATINGS_PATH,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int32",
        },
    )


@st.cache_data(show_spinner="Building user index …")
def build_user_index(_ratings: pd.DataFrame) -> set:
    """Return the set of all unique user IDs."""
    return set(_ratings["userId"].unique().tolist())



# 4.  MOCK PROFILES (for instant demo)

MOCK_PROFILES = {
    "🎥 Sci-Fi Buff (Mock)": {
        "titles": [
            "Inception", "Interstellar", "The Matrix", "Blade Runner 2049",
            "Ex Machina", "Arrival", "Gravity", "The Martian",
            "Edge of Tomorrow", "Minority Report", "Looper",
            "District 9", "Moon", "Annihilation",
        ],
        "rating": 4.5,
    },
    "🎭 Drama Lover (Mock)": {
        "titles": [
            "The Shawshank Redemption", "Schindler's List", "Forrest Gump",
            "The Godfather", "Fight Club", "Pulp Fiction", "Whiplash",
            "Good Will Hunting", "A Beautiful Mind", "The Pianist",
            "12 Years a Slave", "The Pursuit of Happyness",
        ],
        "rating": 4.8,
    },
    "🦸 Superhero Fan (Mock)": {
        "titles": [
            "The Dark Knight", "Iron Man", "The Avengers",
            "Spider-Man: Into the Spider-Verse", "Logan",
            "Guardians of the Galaxy", "Black Panther", "Thor: Ragnarok",
            "Captain America: The Winter Soldier", "Wonder Woman",
            "Deadpool", "Doctor Strange",
        ],
        "rating": 4.2,
    },
}



# 5.  HELPER FUNCTIONS

def _safe_split(value: str, sep: str = " ") -> list:
    """Split a space-or-comma separated string, filtering blanks."""
    if pd.isna(value) or not str(value).strip():
        return []
    return [tok.strip() for tok in str(value).split(sep) if tok.strip()]


def _decade(year) -> str:
    """Convert a year to its decade label (e.g. 1990 → '1990s')."""
    try:
        y = int(year)
        return f"{(y // 10) * 10}s"
    except (ValueError, TypeError):
        return "Unknown"


def resolve_user_history(
    user_id: int,
    ratings_df: pd.DataFrame,
    links_df: pd.DataFrame,
    content_df: pd.DataFrame,
    min_rating: float = 3.5,
) -> pd.DataFrame:
    """
    For a given userId, get all movies they rated ≥ min_rating,
    then join with the TMDB content model for metadata.
    """
    user_ratings = ratings_df[
        (ratings_df["userId"] == user_id) & (ratings_df["rating"] >= min_rating)
    ].copy()

    if user_ratings.empty:
        return pd.DataFrame()

    # Merge with links to get tmdbId
    merged = user_ratings.merge(links_df[["movieId", "tmdbId"]], on="movieId", how="left")
    merged = merged.dropna(subset=["tmdbId"])
    merged["tmdbId"] = merged["tmdbId"].astype(int)

    # Merge with content model
    result = merged.merge(
        content_df, left_on="tmdbId", right_on="id", how="inner"
    )
    return result


def resolve_mock_history(
    profile: dict, content_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a synthetic history DataFrame from a mock profile dict.
    """
    titles_lower = {t.lower() for t in profile["titles"]}
    mask = content_df["title_lower"].isin(titles_lower)
    result = content_df[mask].copy()
    result["rating"] = profile["rating"]
    return result



# 6.  CHART BUILDERS

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit, sans-serif", color="#cbd5e1"),
    margin=dict(l=20, r=20, t=40, b=20),
    hoverlabel=dict(
        bgcolor="rgba(15, 12, 41, 0.9)", 
        font_size=14, 
        font_family="Outfit"
    )
)


def genre_radar_chart(history: pd.DataFrame, theme: dict) -> go.Figure:
    """Radar chart with simulated WebGL bloom effect."""

    all_genres = []
    for g in history["genres_str"].dropna():
        all_genres.extend(_safe_split(g))

    if not all_genres:
        return _empty_figure("No genre data available")

    genre_counts = pd.Series(all_genres).value_counts()

    labels = [g.capitalize() for g in genre_counts.index]
    values = genre_counts.values.tolist()

    # Close radar loop
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    # ── Glow Layer (render first, behind main shape) ──
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            mode="lines",
            line=dict(
                color=theme["primary"],
                width=12,
            ),
            opacity=0.15,
            hoverinfo="skip",
            showlegend=False,
            name="Glow",
        )
    )

    # ── Main Radar Layer ──
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=theme["bg_glow"],
            line=dict(
                color=theme["primary"],
                width=3,
            ),
            marker=dict(
                size=8,
                color="#ffffff",
                line=dict(
                    color=theme["primary"],
                    width=2,
                ),
            ),
            name="Genre Count",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Outfit",
            color="#94a3b8",
        ),

        polar=dict(
            # Tinted radar interior
            bgcolor=theme["bg_glow"],

            radialaxis=dict(
                visible=True,
                gridcolor="rgba(255,255,255,0.10)",
                color="#64748b",
            ),

            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                color="#e2e8f0",
                tickfont=dict(
                    size=12,
                ),
                linecolor=theme["primary"],  # colored outer ring
                linewidth=2,
            ),
        ),

        showlegend=False,
        margin=dict(
            l=40,
            r=40,
            t=40,
            b=40,
        ),
        height=460,
    )

    return fig


def decade_bar_chart(history: pd.DataFrame, theme: dict) -> go.Figure:
    """Bar chart of movies per decade matching dynamic theme."""
    history = history.copy()
    history["decade"] = history["year"].apply(_decade)
    decade_counts = (
        history.groupby("decade")
        .size()
        .reset_index(name="count")
        .sort_values("decade")
    )
    decade_counts = decade_counts[decade_counts["decade"] != "Unknown"]

    if decade_counts.empty:
        return _empty_figure("No decade data available")

    fig = px.bar(
        decade_counts,
        x="decade",
        y="count",
        color="count",
        # Swap hardcoded colors for a smooth theme gradient
        color_continuous_scale=[theme['secondary'], theme['primary']], 
        labels={"decade": "Decade", "count": "Movies"},
    )
    fig.update_traces(
        marker_line_width=0,
        marker_cornerradius=6,
        hovertemplate="<b>%{x}</b><br>%{y} movies<extra></extra>",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        coloraxis_showscale=False,
        title=dict(text="Movies by Decade", font=dict(size=16, color="#e0e7ff")),
        xaxis=dict(title="", tickfont=dict(size=12)),
        yaxis=dict(title="", tickfont=dict(size=11), gridcolor="rgba(255,255,255,0.05)"),
        height=380,
    )
    return fig


def director_bar_chart(history: pd.DataFrame, theme: dict, top_n: int = 8) -> go.Figure:
    """Horizontal bar chart of top directors matching dynamic theme."""
    all_dirs = []
    for d in history["director_str"].dropna():
        cleaned = _safe_split(d)
        all_dirs.extend(cleaned)

    if not all_dirs:
        return _empty_figure("No director data available")

    counts = pd.Series(all_dirs).value_counts().head(top_n).iloc[::-1]
    labels = [_format_name(d) for d in counts.index]

    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=labels,
            orientation="h",
            marker=dict(
                color=counts.values,
                # Theme gradient for the bars
                colorscale=[[0, theme['secondary']], [1, theme['primary']]],
                cornerradius=4,
                line_width=0,
            ),
            hovertemplate="<b>%{y}</b><br>%{x} movies<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Top Directors", font=dict(size=16, color="#e0e7ff")),
        xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)", dtick=1),
        yaxis=dict(title="", tickfont=dict(size=12)),
        height=380,
    )
    return fig

def actor_bar_chart(history: pd.DataFrame, theme: dict, top_n: int = 8) -> go.Figure:
    """Horizontal bar chart of top actors matching dynamic theme."""
    all_actors = []
    for c in history["cast_str"].dropna():
        cleaned = _safe_split(c)
        all_actors.extend(cleaned)

    if not all_actors:
        return _empty_figure("No cast data available")

    counts = pd.Series(all_actors).value_counts().head(top_n).iloc[::-1]
    labels = [_format_name(a) for a in counts.index]

    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=labels,
            orientation="h",
            marker=dict(
                color=counts.values,
                # Theme gradient for the bars
                colorscale=[[0, theme['secondary']], [1, theme['primary']]],
                cornerradius=4,
                line_width=0,
            ),
            hovertemplate="<b>%{y}</b><br>%{x} movies<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Top Actors", font=dict(size=16, color="#e0e7ff")),
        xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)", dtick=1),
        yaxis=dict(title="", tickfont=dict(size=12)),
        height=380,
    )
    return fig


def rating_distribution_chart(history: pd.DataFrame, theme: dict) -> go.Figure:
    """Histogram of user's rating distribution matching dynamic theme."""
    if "rating" not in history.columns:
        return _empty_figure("No rating data")

    ratings = history["rating"].dropna()
    fig = px.histogram(
        ratings,
        nbins=10,
        # Force the histogram to use the theme's primary color
        color_discrete_sequence=[theme['primary']], 
        labels={"value": "Rating", "count": "Movies"},
    )
    fig.update_traces(
        marker_line_width=0,
        marker_cornerradius=4,
        hovertemplate="Rating: %{x}<br>Count: %{y}<extra></extra>",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Rating Distribution", font=dict(size=16, color="#e0e7ff")),
        xaxis=dict(title="Rating", tickfont=dict(size=12), dtick=0.5),
        yaxis=dict(title="Count", tickfont=dict(size=11), gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False,
        height=340,
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    """Return a placeholder figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="#94a3b8"),
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    return fig


def _format_name(raw: str) -> str:
    """
    Uses NLP to split concatenated tokens like 'stevenspielberg' 
    back into 'Steven Spielberg'.
    """
    if pd.isna(raw) or not str(raw).strip():
        return "Unknown"
    
    # Clean the raw string just in case
    clean_raw = str(raw).replace("-", "").lower()
    
    # WordNinja magically splits 'tomhanks' -> ['tom', 'hanks']
    split_words = wordninja.split(clean_raw)
    
    # Join them with a space and capitalize properly
    return " ".join(split_words).title()



# 7.  SIDEBAR

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:20px 0;">
            <span style="font-size:3rem; text-shadow: 0 0 20px rgba(236,72,153,0.5);">🎬</span>
            <h2 style="margin:10px 0 0 0; font-weight:800; font-size: 2.2rem;
                        background:linear-gradient(to right, #00f2fe, #4facfe);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        text-shadow: 0px 4px 20px rgba(0, 242, 254, 0.3);">
                CineIQ
            </h2>
            <p style="font-size:0.9rem; color:#94a3b8; font-weight: 500; letter-spacing: 1px; margin-top: 5px;">
                ENGINE INTELLIGENCE
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    mode = st.radio(
        "Profile source",
        ["🔢 Enter User ID", "🧪 Mock Profile"],
        index=1,
        help="Choose a real user from the dataset or a pre-built mock profile.",
    )

    user_id = None
    mock_key = None
    user_identifier = "default"

    if mode == "🔢 Enter User ID":
        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=300000,
            value=22,
        )

        min_rating = st.slider(
            "Minimum rating threshold",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.5,
        )

        user_identifier = str(user_id)

    else:
        mock_key = st.selectbox(
            "Select mock profile",
            list(MOCK_PROFILES.keys()),
            help="Pre-built taste profiles for instant demo.",
        )

        min_rating = 3.5
        user_identifier = mock_key

    # INJECTING DYNAMIC STYLING HERE
    current_theme = get_dynamic_theme(user_identifier)
    inject_glassmorphism(current_theme)

    st.divider()
    st.caption("Built with Streamlit · Plotly · Pandas")



# 8.  MAIN CONTENT


# Hero banner
st.markdown(
    """
    <div class="hero">
        <h1>🎬 User Profile Analytics</h1>
        <p>Explore taste DNA — genres, decades, directors & actors that define a viewer.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data
content_df = load_content_model()

if mode == "🔢 Enter User ID":
    # Need ratings & links for real users
    ratings_df = load_ratings()
    links_df = load_links()
    user_set = build_user_index(ratings_df)

    if user_id not in user_set:
        st.error(
            f"⚠️ User **{user_id}** not found in the dataset. "
            f"Try a value between 1 and {max(user_set):,}."
        )
        st.stop()

    history = resolve_user_history(user_id, ratings_df, links_df, content_df, min_rating)
    profile_label = f"User #{user_id}"
else:
    profile = MOCK_PROFILES[mock_key]
    history = resolve_mock_history(profile, content_df)
    profile_label = mock_key

# Guard: empty history
if history.empty:
    st.warning(
        "😕 No matching movies found for this profile. "
        "Try lowering the minimum rating or switching profiles."
    )
    st.stop()

# ── KPI row ──
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Movies", f"{len(history):,}")
with k2:
    avg_vote = history["vote_average"].mean()
    st.metric("Avg TMDB Score", f"{avg_vote:.1f}" if pd.notna(avg_vote) else "—")
with k3:
    avg_year = history["year"].dropna().mean()
    st.metric("Avg Release Year", f"{int(avg_year)}" if pd.notna(avg_year) else "—")
with k4:
    unique_genres = set()
    for g in history["genres_str"].dropna():
        unique_genres.update(_safe_split(g))
    st.metric("Genres Explored", f"{len(unique_genres)}")

st.markdown("")  # spacer

# ── Row 1:  Genre Radar  +  Decade Bar ──
st.markdown(
    '<div class="section-header"><span class="icon">🧬</span> Taste Profile</div>',
    unsafe_allow_html=True,
)

col_radar, col_decade = st.columns([1, 1], gap="large")

with col_radar:
    st.plotly_chart(genre_radar_chart(history, current_theme), use_container_width=True)

with col_decade:
   st.plotly_chart(decade_bar_chart(history, current_theme), use_container_width=True)
# ── Row 2:  Director + Actor affinities ──
st.markdown(
    '<div class="section-header"><span class="icon">🌟</span> Director & Actor Affinities</div>',
    unsafe_allow_html=True,
)

col_dir, col_act = st.columns(2, gap="large")

with col_dir:
    st.plotly_chart(director_bar_chart(history, current_theme, top_n=8), use_container_width=True)

with col_act:
    st.plotly_chart(actor_bar_chart(history, current_theme, top_n=8), use_container_width=True)

# ── Row 3:  Rating histogram (real users only) ──
if "rating" in history.columns:
    st.markdown(
        '<div class="section-header"><span class="icon">📊</span> Rating Behaviour</div>',
        unsafe_allow_html=True,
    )
    col_hist, col_space = st.columns([1, 1])
    with col_hist:
        st.plotly_chart(rating_distribution_chart(history, current_theme), use_container_width=True)

# ── Expander:  Raw movie list ──
with st.expander(f"📋 Full Movie List — {profile_label}", expanded=False):
    display_cols = ["title", "year", "genres_str", "director_str", "cast_str", "vote_average", "popularity"]
    available_cols = [c for c in display_cols if c in history.columns]
    if "rating" in history.columns:
        available_cols = ["rating"] + available_cols

    pretty = history[available_cols].copy()
    rename_map = {
        "title": "Title",
        "year": "Year",
        "genres_str": "Genres",
        "director_str": "Director",
        "cast_str": "Cast",
        "vote_average": "TMDB Score",
        "popularity": "Popularity",
        "rating": "Your Rating",
    }
    pretty = pretty.rename(columns={k: v for k, v in rename_map.items() if k in pretty.columns})
    st.dataframe(
        pretty.sort_values(
            by="TMDB Score" if "TMDB Score" in pretty.columns else pretty.columns[0],
            ascending=False,
        ),
        use_container_width=True,
        height=420,
    )



import requests
import streamlit as st

# --- RECOMMENDATIONS PAGE ---
st.header("🎬 Hybrid Recommendations")

# Safely grab the user_id from the sidebar. If Mock Profile is active, use a dummy ID (like 0) so the backend SVD model doesn't crash on 'None'.
api_user_id = user_id if user_id is not None else 0

movie_input = st.text_input("Enter a movie you like:", placeholder="e.g. Inception")

if st.button("Get Recommendations") and movie_input:
    with st.spinner("CineIQ is thinking..."):
        response = requests.get(
            "http://localhost:8000/recommend",
            # FIX 1: Now securely passing api_user_id to the backend
            params={"user_id": api_user_id, "movie_title": movie_input} 
        )
    
    if response.status_code == 200:
        data = response.json()
        
        if "error" in data:
            st.error(data["error"])
        else:
            # FIX 2: Now displaying the correct ID in the UI
            st.subheader(f"Top picks for User #{api_user_id} based on *{data['target_movie']}*")
            
            for i, rec in enumerate(data["recommendations"], 1):
                sentiment_emoji = "😊" if rec["sentiment_label"] == "POSITIVE" else "😐"
                
                # Generate the visual math breakdown instead of text
                lime_visualization = generate_lime_html(rec["feature_weights"])
                
                # Building the pure HTML glowing card
                card_html = f"""
                <div class="rec-card">
                    <div class="rec-content">
                        <h3 class="rec-title">{i}. {rec['title']}</h3>
                        <div style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 10px;">Model Attribution Breakdown:</div>
                        {lime_visualization}
                    </div>
                    <div class="rec-stats">
                        <div class="rec-score-label">Hybrid Score</div>
                        <div class="rec-score-value">{rec['final_hybrid_score']}</div>
                        <div class="rec-mini-stat">{sentiment_emoji} Audience: {rec['sentiment_label']}</div>
                        <div class="rec-mini-stat">⭐ Predicted Rating: {rec['predicted_user_rating']}/5</div>
                    </div>
                </div>
                """
                # Injecting the card into Streamlit
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.error("Backend not reachable. Make sure uvicorn is running.")