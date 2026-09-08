"""
src/dashboard/app.py

Movie Mind Reader — Powered by BERT4Rec 4-Layer Transformer.

Flow:
  Step 1: User picks 3–5 movies they recently loved (their "watch sequence").
  Step 2: AI generates 4 candidate movies using BERT4Rec sequential inference.
  Step 3: User secretly picks ONE of the 4 movies in their mind (without telling AI).
  Step 4: User clicks Reveal → AI predicts which movie was chosen using real BERT4Rec scores.

Run:
    streamlit run src/dashboard/app.py
"""

import os
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import yaml

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Movie Mind Reader",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B0C10;
}

.hero {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E50914 0%, #F56565 50%, #ED8936 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1.05rem;
    color: #9A9EA7;
    font-weight: 400;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.5;
}
.step-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #E50914;
    margin-bottom: 0.2rem;
}
.step-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F7FAFC;
    margin-bottom: 0.8rem;
    letter-spacing: -0.01em;
}
.movie-card {
    background: #14161F;
    border: 1px solid #232734;
    border-radius: 12px;
    padding: 1.2rem 1.1rem;
    margin-bottom: 0.8rem;
    transition: all 0.2s ease;
    height: 125px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.movie-card:hover {
    border-color: #E50914;
    background: #191C27;
}
.movie-card .option-num {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #E50914;
    text-transform: uppercase;
}
.movie-card .movie-title {
    font-size: 1rem;
    font-weight: 600;
    color: #FFFFFF;
    margin: 0.2rem 0;
    line-height: 1.35;
}
.movie-card .movie-genres {
    font-size: 0.78rem;
    color: #717A8A;
}
.winner-card {
    background: #151823;
    border: 1.5px solid #E50914;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.divider {
    border: none;
    border-top: 1px solid #1C1F2B;
    margin: 1.8rem 0;
}
.hint-box {
    background: #131620;
    border-left: 3px solid #E50914;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.8rem 0;
    color: #A0AEC0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────

GENRE_TO_ML1M = {
    "Action":    [2571, 260, 1196, 2028, 1527],
    "Comedy":    [1, 3114, 2355, 3751, 8641],
    "Drama":     [318, 527, 1197, 1198, 1221],
    "Romance":   [356, 586, 1580, 3948, 4973],
    "Thriller":  [593, 589, 1240, 1617, 2918],
    "Horror":    [2762, 2918, 3555, 1291, 588],
    "Sci-Fi":    [2571, 1196, 260, 2028, 3114],
    "Crime":     [858, 296, 593, 1213, 2959],
    "War":       [527, 110, 2628, 1704, 1945],
    "Adventure": [260, 1196, 3578, 2028, 1527],
    "Sport":     [110, 527, 1968, 1097, 1617],
    "Biography": [527, 110, 912, 1704, 1617],
    "Music":     [2858, 2355, 1, 3114, 4306],
}

@st.cache_data
def load_movies():
    p_32m = Path("data/raw/ml-32m/movies_vocab.csv")
    p_latest = Path("data/raw/ml-latest-small/movies.csv")

    if p_32m.exists():
        df = pd.read_csv(p_32m)
        df["source"] = "Hollywood"
    elif p_latest.exists():
        df = pd.read_csv(p_latest)
        df["source"] = "Hollywood"
        df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype(float)
        df = df[df["year"].isna() | (df["year"] >= 1970)].drop(columns="year")
    else:
        df = pd.DataFrame([
            {"movieId": 1,    "title": "Toy Story (1995)",    "genres": "Animation|Comedy"},
            {"movieId": 2571, "title": "Matrix, The (1999)",  "genres": "Action|Sci-Fi"},
        ])
        df["source"] = "Hollywood"

    # Append Bollywood / Indian movies
    p_bolly = Path("data/raw/bollywood/movies.csv")
    if p_bolly.exists():
        df_bolly = pd.read_csv(p_bolly)
        df_bolly["source"] = "Indian Cinema"
        df = pd.concat([df, df_bolly], ignore_index=True).drop_duplicates(subset="title")

    return df.reset_index(drop=True)


def genre_to_bert4rec_ids(genres_str: str, n: int = 5) -> list:
    """Map a genre string to known movie IDs."""
    genres = genres_str.replace("|", " ").split()
    candidates = []
    for g in genres:
        candidates.extend(GENRE_TO_ML1M.get(g, []))
    seen = set()
    result = []
    for mid in candidates:
        if mid not in seen:
            seen.add(mid)
            result.append(mid)
    return result[:n] if result else [318, 260, 2571, 593, 356]


@st.cache_resource
def load_bert4rec():
    from src.inference.bert4rec_inference import BERT4RecInference
    return BERT4RecInference().load()


movies_df   = load_movies()
title_to_id = dict(zip(movies_df["title"], movies_df["movieId"]))
id_to_row   = movies_df.set_index("movieId").to_dict(orient="index")
engine      = load_bert4rec()

if engine:
    ALL_TITLES = sorted(
        t for t, mid in title_to_id.items()
        if int(mid) in engine.movie_to_idx
    )
else:
    ALL_TITLES = sorted(movies_df["title"].tolist())


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-title">Movie Mind Reader</div>
    <div class="hero-sub">Can our recommendation model guess what movie you'll pick next?</div>
</div>

<div class="hint-box" style="border-left-color: #E50914; background: #131622; max-width: 720px; margin: 1.2rem auto; text-align: left; padding: 1rem 1.4rem;">
    <strong style="color: #F7FAFC; font-size: 0.95rem;">How it works:</strong>
    <ol style="margin: 0.4rem 0 0 1.2rem; padding: 0; color: #A0AEC0; font-size: 0.88rem; line-height: 1.6;">
        <li><b>Step 1:</b> Add 1 to 5 movies you recently enjoyed.</li>
        <li><b>Step 2:</b> Pick <b>ONE</b> of the 4 generated movies silently in your mind.</li>
        <li><b>Step 3:</b> Select what you picked and click <i>Reveal Prediction</i> to see if the model guessed right!</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 1: Select Favorites ──────────────────────────────────────────────────

st.markdown('<div class="step-label">Step 1 of 3</div>', unsafe_allow_html=True)
st.markdown('<div class="step-title">Add movies you recently enjoyed</div>', unsafe_allow_html=True)

# Initialise selected list in session state
if "selected_titles" not in st.session_state:
    st.session_state["selected_titles"] = [
        t for t in ["Matrix, The (1999)", "Star Wars: Episode IV - A New Hope (1977)", "Terminator 2: Judgment Day (1991)"]
        if t in title_to_id
    ]

ALL_TITLES_LOWER = {t: t.lower() for t in ALL_TITLES}

def add_movie_to_sequence(movie_title: str):
    if movie_title not in st.session_state["selected_titles"] and len(st.session_state["selected_titles"]) < 5:
        st.session_state["selected_titles"].append(movie_title)

# Reset picker state if reset_picker flag is set
if st.session_state.get("reset_picker"):
    st.session_state["search_q_input"] = ""
    st.session_state["reset_picker"]   = False

# ── Side-by-Side Search & Top 7 Matches Dropdown (0ms Lag, All 54k Movies) ────
col_search, col_dropdown = st.columns([1, 1])

with col_search:
    search_q = st.text_input(
        "Search titles:",
        placeholder="Type e.g. Godfather, Inception, Shutter Island, RRR...",
        key="search_q_input",
    )

query = search_q.strip().lower()
if query and len(query) >= 2:
    prefix  = [t for t in ALL_TITLES if ALL_TITLES_LOWER[t].startswith(query)]
    sub     = [t for t in ALL_TITLES if query in ALL_TITLES_LOWER[t] and not ALL_TITLES_LOWER[t].startswith(query)]
    matches = (prefix + sub)[:7]  # Top 7 matches max -> ZERO browser lag!
else:
    # Top default suggestions when search is empty
    matches = [t for t in [
        "Godfather, The (1972)",
        "Inception (2010)",
        "Matrix, The (1999)",
        "Pulp Fiction (1994)",
        "Shawshank Redemption, The (1994)",
        "Dark Knight, The (2008)",
        "Interstellar (2014)",
    ] if t in title_to_id][:7]

with col_dropdown:
    lbl = f"Matches for \"{search_q.strip()}\":" if query else "Select from suggestions:"
    chosen_movie = st.selectbox(
        lbl,
        options=["— Choose a movie to add —"] + matches,
        key="top_matches_dropdown",
    )
    if chosen_movie and chosen_movie != "— Choose a movie to add —":
        add_movie_to_sequence(chosen_movie)
        st.session_state["reset_picker"] = True
        st.rerun()

# ── Display Watch Sequence ────────────────────────────────────────────────────
selected_titles = st.session_state["selected_titles"]

if selected_titles:
    st.markdown(f"**Your Selection** ({len(selected_titles)}/5 movies in watch order):")
    for i, t in enumerate(selected_titles):
        c1, c2 = st.columns([6, 1])
        c1.markdown(f"`{i+1}.` **{t}**")
        if c2.button("Remove", key=f"remove_{i}", help="Remove movie"):
            st.session_state["selected_titles"].pop(i)
            st.rerun()
else:
    st.markdown("""
    <div class="hint-box">
        Search and add at least one movie above to proceed.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if len(selected_titles) < 1:
    st.stop()

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 2: Options Generation ────────────────────────────────────────────────

st.markdown('<div class="step-label">Step 2 of 3</div>', unsafe_allow_html=True)
st.markdown('<div class="step-title">Pick one movie in your mind</div>', unsafe_allow_html=True)

sequence_changed = (st.session_state.get("last_selected_titles") != selected_titles)

generate_clicked = st.button("Refresh 4 Options", type="secondary", key="gen_btn")
if generate_clicked or sequence_changed or "game_options" not in st.session_state or "ai_top_id" not in st.session_state:
    st.session_state["last_selected_titles"] = list(selected_titles)
    selected_ids = [int(title_to_id[t]) for t in selected_titles if t in title_to_id]
    seed_ids     = [mid for mid in selected_ids if engine and mid in engine.movie_to_idx]

    top_recs = engine.top_n(seed_ids, n=50, exclude_ids=selected_ids) if (engine and seed_ids) else []

    scored_options = []
    for rec_mid, score in top_recs:
        if rec_mid in id_to_row:
            row = id_to_row[rec_mid]
            scored_options.append({
                "movieId": rec_mid,
                "score":   float(score),
                "title":   str(row["title"]),
                "genres":  str(row.get("genres", "")),
                "source":  str(row.get("source", "")),
            })

    if len(scored_options) >= 4:
        top_pick    = scored_options[0]
        pool        = scored_options[1:min(30, len(scored_options))]
        distractors = random.sample(pool, min(3, len(pool)))
        all_4       = [top_pick] + distractors
    else:
        avail     = movies_df[~movies_df["movieId"].isin(selected_ids)]
        sample_df = avail.sample(n=min(4, len(avail)), random_state=random.randint(1, 99999))
        fallback_ids = [int(r["movieId"]) for _, r in sample_df.iterrows()]
        fb_scores    = engine.score_movies(seed_ids, fallback_ids) if engine else {}
        all_4 = [
            {
                "movieId": int(r["movieId"]),
                "score":   fb_scores.get(int(r["movieId"]), 0.0001),
                "title":   str(r["title"]),
                "genres":  str(r.get("genres", "")),
                "source":  str(r.get("source", "")),
            }
            for _, r in sample_df.iterrows()
        ]

    random.shuffle(all_4)
    st.session_state["game_options"] = all_4
    st.session_state["ai_top_id"]    = max(all_4, key=lambda x: x["score"])["movieId"]
    st.session_state["revealed"]     = False


options_4 = st.session_state.get("game_options", [])
ai_top_id = st.session_state.get("ai_top_id", options_4[0]["movieId"] if options_4 else 0)

st.markdown("""
<div class="hint-box">
    Look at the 4 movies below. Pick <b>ONE</b> silently in your mind — keep your choice secret until Step 3!
</div>
""", unsafe_allow_html=True)

# Display 4 movie cards in 2x2 grid
col_a, col_b = st.columns(2)
for i, opt in enumerate(options_4):
    col = col_a if i % 2 == 0 else col_b
    src = opt.get("source", "")
    src_badge = f'<span style="font-size:0.7rem;background:#1e2d3d;color:#5bc4f5;padding:1px 6px;border-radius:3px;margin-left:4px">{src}</span>' if src else ""
    with col:
        st.markdown(f"""
        <div class="movie-card">
            <div class="option-num">Option {i+1} {src_badge}</div>
            <div class="movie-title">{opt['title']}</div>
            <div class="movie-genres">{opt['genres']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 3: Reveal Prediction ─────────────────────────────────────────────────

st.markdown('<div class="step-label">Step 3 of 3</div>', unsafe_allow_html=True)
st.markdown('<div class="step-title">See if the model guessed your pick</div>', unsafe_allow_html=True)

user_choice_label = st.radio(
    "Select the option you chose in your mind:",
    options=[f"Option {i+1}: {opt['title']}" for i, opt in enumerate(options_4)],
    horizontal=True,
    index=None,
)

reveal_clicked = st.button("Reveal Prediction", type="primary",
                            use_container_width=True, disabled=(user_choice_label is None))

if reveal_clicked and user_choice_label:
    user_chosen_idx = int(user_choice_label.split(":")[0].replace("Option ", "").strip()) - 1
    user_chosen_opt = options_4[user_chosen_idx]
    user_chosen_id  = user_chosen_opt["movieId"]

    ai_top_opt  = next((o for o in options_4 if o["movieId"] == ai_top_id), options_4[0])
    sum_scores  = sum(o["score"] for o in options_4) or 1.0
    ai_confidence = round((ai_top_opt["score"] / sum_scores) * 100, 1)

    # Score bars for all 4 options
    st.markdown("### Model Prediction Confidence")
    for i, opt in enumerate(options_4):
        pct = round((opt["score"] / sum_scores) * 100, 1)
        is_ai   = opt["movieId"] == ai_top_id
        is_user = opt["movieId"] == user_chosen_id
        label_badge = ""
        if is_ai and is_user:
            label_badge = " (Predicted & Selected)"
        elif is_ai:
            label_badge = " (Model Prediction)"
        elif is_user:
            label_badge = " (Your Selection)"
        st.markdown(f"**Option {i+1}**: {opt['title']}{label_badge}")
        st.progress(pct / 100.0)
        st.caption(f"Relative confidence: {pct}%")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Result reveal
    if user_chosen_id == ai_top_id:
        st.markdown(f"""
        <div class="winner-card">
            <div style="font-size: 1.5rem; font-weight: 800; color: #E50914; margin-bottom: 0.5rem">Match Found</div>
            <div style="font-size: 1.05rem; color: #DDD; margin-bottom: 0.8rem">
                The model correctly predicted <b>{ai_top_opt['title']}</b>.
            </div>
            <div style="font-size: 0.88rem; color: #888">
                Based on your recent selection, this movie received the highest relative affinity score ({ai_confidence}% confidence).
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#151823; border: 1px solid #282E3E; border-radius: 14px; padding: 1.5rem; text-align:center;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #E2E8F0; margin-bottom:0.5rem">
                Different Choice
            </div>
            <div style="font-size: 0.95rem; color: #CCC; margin-bottom: 0.8rem">
                Model predicted: <b style="color:#E50914">{ai_top_opt['title']}</b>
                &nbsp;|&nbsp;
                You selected: <b style="color:#4299E1">{user_chosen_opt['title']}</b>
            </div>
            <div style="font-size: 0.85rem; color:#717A8A">
                The model assigned higher relative affinity to <i>{ai_top_opt['title']}</i> given your history, but noted your preference for <i>{user_chosen_opt['title']}</i>.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar: Model & Pipeline Info ────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Model Status")
    st.markdown("---")

    meta_path = Path("models/champion_meta.yaml")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        st.markdown("**Active Champion Model**")
        st.json(meta)
    else:
        st.info("No active champion metadata.")

    st.markdown("---")
    if engine:
        st.success("BERT4Rec Model Active")
        st.caption("4-Layer Transformer Encoder")
        st.caption(f"Vocabulary: 54,717 movies")
    else:
        st.error("Model unavailable")

    st.markdown("---")
    if st.button("Trigger Data Stream", help="Stream next batch into the dataset"):
        try:
            from src.replay.replay_controller import get_next_batch
            p = get_next_batch("configs/config.yaml")
            if p:
                st.success(f"Processed batch: {p}")
            else:
                st.warning("No further data.")
        except Exception as e:
            st.error(f"Stream error: {e}")

    st.markdown("---")
    st.caption("BERT4Rec 32M Dataset")
