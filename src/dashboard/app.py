"""
src/dashboard/app.py

Interactive Streamlit Data App & Dashboard for the Movie Recommendation System.

Features:
  1. Live Recommendation Engine — test user IDs, view scores, titles, genres.
  2. Continuous Data Replay Control — click "Release Next Batch" to stream data.
  3. Model Evaluation & Drift Metrics — view RMSE, NDCG@10, Precision@10 trends.
  4. Champion Model Information — inspect active production model metadata.

Run locally:
    streamlit run src/dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Movie Recommender MLOps Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8F00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Supporting Data ──────────────────────────────────────────────────────

@st.cache_data
def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_movies():
    path = Path("data/raw/ml-1m/movies.csv")
    if not path.exists():
        path = Path("data/raw/ml-latest-small/movies.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["movieId", "title", "genres"])


cfg = load_config()
movies_df = load_movies()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/000000/clapperboard.png", width=64)
st.sidebar.title("MLOps RecSys")
st.sidebar.markdown("**Dataset**: MovieLens `ml-1m` (1M ratings)")

page = st.sidebar.radio(
    "Navigation",
    ["🎬 Live Recommendations", "🔄 Data Replay Controller", "📊 Model Performance & Drift", "⚙️ Champion Metadata"],
)


# ── Page 1: Live Recommendations ─────────────────────────────────────────────

if page == "🎬 Live Recommendations":
    st.markdown('<div class="main-header">Live Movie Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Query the active Champion model for personalized movie recommendations.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Query Parameters")
        user_id = st.number_input("User ID", min_value=1, max_value=6040, value=1, step=1)
        top_k = st.slider("Top K Recommendations", min_value=1, max_value=20, value=10)
        fetch_button = st.button("Generate Recommendations", type="primary", use_container_width=True)

    with col2:
        st.subheader(f"Top-{top_k} Recommendations for User {user_id}")

        if fetch_button or "recommendations" in st.session_state:
            try:
                from src.inference.recommender import Recommender
                rec = Recommender("configs/config.yaml").load()
                results = rec.recommend(user_id=user_id, n=top_k)

                if results:
                    st.success(f"Model Type: **{rec.model_type.upper()}** | Served: {len(results)} movies")
                    recs_df = pd.DataFrame(results)

                    # Display formatted table
                    cols_to_show = ["movieId", "title", "genres"]
                    if "score" in recs_df.columns:
                        cols_to_show.append("score")
                    elif "predicted_rating" in recs_df.columns:
                        cols_to_show.append("predicted_rating")

                    st.dataframe(
                        recs_df[cols_to_show],
                        column_config={
                            "movieId": "Movie ID",
                            "title": "Movie Title",
                            "genres": "Genres",
                            "score": st.column_config.NumberColumn("Score / Rating", format="%.3f"),
                            "predicted_rating": st.column_config.NumberColumn("Predicted Rating", format="%.3f"),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No recommendations returned for this user.")
            except Exception as e:
                st.error(f"Error loading champion model: {e}")


# ── Page 2: Data Replay Controller ───────────────────────────────────────────

elif page == "🔄 Data Replay Controller":
    st.markdown('<div class="main-header">Continuous Data Replay Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulates periodic data streaming from the immutable master dataset.</div>', unsafe_allow_html=True)

    from src.replay.replay_state import load_state

    state = load_state()

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Batch", f"#{state.last_released_batch}")
    col2.metric("Total Ratings Released", f"{state.total_ratings_released:,}")
    col3.metric("Dataset Version", state.dataset_version)

    st.markdown("---")

    if st.button("🚀 Release Next Chronological Batch (N-Days)", type="primary"):
        with st.spinner("Slicing next chronological batch from master dataset..."):
            from src.replay.replay_controller import get_next_batch
            path = get_next_batch("configs/config.yaml")
            if path:
                st.success(f"✓ Released batch saved to: `{path}`")
                new_state = load_state()
                st.balloons()
            else:
                st.warning("All master dataset data has already been replayed!")


# ── Page 3: Model Performance & Drift ────────────────────────────────────────

elif page == "📊 Model Performance & Drift":
    st.markdown('<div class="main-header">Model Performance & Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Track evaluation metrics and drift detection logs across retraining cycles.</div>', unsafe_allow_html=True)

    from src.monitoring.performance import load_performance_history

    history = load_performance_history()

    if history:
        st.subheader("Historical Retraining Cycles")
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No historical performance cycles logged yet. Run a retraining cycle to record metrics.")


# ── Page 4: Champion Metadata ─────────────────────────────────────────────────

elif page == "⚙️ Champion Metadata":
    st.markdown('<div class="main-header">Active Champion Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection of current production champion metadata.</div>', unsafe_allow_html=True)

    meta_path = Path("models/champion_meta.yaml")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        st.json(meta)
    else:
        st.warning("No champion metadata found at `models/champion_meta.yaml`.")
