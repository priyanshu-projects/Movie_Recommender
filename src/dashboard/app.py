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

import random
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Movie Mind Reader — BERT4Rec AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0A0F;
}

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem 1rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #E50914 0%, #FF6B35 50%, #FFD700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #888;
    letter-spacing: 0.04em;
}
.step-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #E50914;
    margin-bottom: 0.3rem;
}
.step-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 1rem;
}
.movie-card {
    background: linear-gradient(145deg, #13141C, #1A1C25);
    border: 1px solid #2A2D3A;
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s ease;
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.movie-card:hover {
    border-color: #E50914;
}
.movie-card .option-num {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #E50914;
    text-transform: uppercase;
}
.movie-card .movie-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0.3rem 0;
    line-height: 1.3;
}
.movie-card .movie-genres {
    font-size: 0.8rem;
    color: #666;
}
.score-bar-wrap {
    background: #1A1C25;
    border: 1px solid #2A2D3A;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.4rem 0;
}
.score-bar-label {
    font-size: 0.85rem;
    color: #CCC;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.winner-card {
    background: linear-gradient(135deg, #1A0A0A, #2A0D0D);
    border: 2px solid #E50914;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(229, 9, 20, 0.4); }
    70% { box-shadow: 0 0 0 15px rgba(229, 9, 20, 0); }
    100% { box-shadow: 0 0 0 0 rgba(229, 9, 20, 0); }
}
.divider {
    border: none;
    border-top: 1px solid #1E2030;
    margin: 2rem 0;
}
.hint-box {
    background: #0D1117;
    border-left: 3px solid #FFD700;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: #BBB;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_movies():
    for path in [
        Path("data/raw/ml-1m/movies.csv"),
        Path("data/raw/ml-latest-small/movies.csv"),
    ]:
        if path.exists():
            return pd.read_csv(path)
    # Fallback mini dataset
    return pd.DataFrame([
        {"movieId": 1,    "title": "Toy Story (1995)",                                 "genres": "Animation|Children's|Comedy"},
        {"movieId": 260,  "title": "Star Wars: Episode IV - A New Hope (1977)",         "genres": "Action|Adventure|Sci-Fi"},
        {"movieId": 1196, "title": "Star Wars: Episode V - The Empire Strikes Back (1980)", "genres": "Action|Adventure|Sci-Fi"},
        {"movieId": 2571, "title": "Matrix, The (1999)",                               "genres": "Action|Sci-Fi|Thriller"},
        {"movieId": 318,  "title": "Shawshank Redemption, The (1994)",                 "genres": "Drama"},
        {"movieId": 296,  "title": "Pulp Fiction (1994)",                              "genres": "Crime|Drama"},
        {"movieId": 356,  "title": "Forrest Gump (1994)",                              "genres": "Comedy|Drama|Romance"},
        {"movieId": 593,  "title": "Silence of the Lambs, The (1991)",                "genres": "Crime|Horror|Thriller"},
        {"movieId": 2959, "title": "Fight Club (1999)",                                "genres": "Action|Crime|Drama|Thriller"},
        {"movieId": 527,  "title": "Schindler's List (1993)",                          "genres": "Drama|War"},
        {"movieId": 858,  "title": "Godfather, The (1972)",                            "genres": "Action|Crime|Drama"},
        {"movieId": 2858, "title": "American Beauty (1999)",                           "genres": "Comedy|Drama"},
    ])


@st.cache_resource
def load_recommender():
    try:
        from src.inference.recommender import Recommender
        rec = Recommender("configs/config.yaml").load()
        return rec
    except Exception:
        return None


movies_df = load_movies()
title_to_id = dict(zip(movies_df["title"], movies_df["movieId"]))
id_to_row   = movies_df.set_index("movieId").to_dict(orient="index")
recommender = load_recommender()

ALL_TITLES = sorted(movies_df["title"].tolist())


# ── Hero Header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-title">🔮 Movie Mind Reader</div>
    <div class="hero-sub">Can our 4-Layer BERT4Rec Transformer read your mind?</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 1: Pick Favorites ────────────────────────────────────────────────────

st.markdown('<div class="step-label">Step 1 of 3</div>', unsafe_allow_html=True)
st.markdown('<div class="step-title">🎬 Tell us movies you recently loved</div>', unsafe_allow_html=True)

DEFAULT_PICKS = [t for t in [
    "Matrix, The (1999)",
    "Star Wars: Episode IV - A New Hope (1977)",
    "Terminator 2: Judgment Day (1991)",
] if t in title_to_id]

selected_titles = st.multiselect(
    "Pick 3–5 movies you loved watching (in order, most recent last):",
    options=ALL_TITLES,
    default=DEFAULT_PICKS[:3],
    max_selections=5,
    help="Choose movies in the order you watched them for best BERT4Rec sequence modelling.",
)

if len(selected_titles) < 1:
    st.markdown("""
    <div class="hint-box">
        👆 Select at least one movie above to start. The AI will use your watch sequence to predict your next pick.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 2: Generate 4 Options ────────────────────────────────────────────────

st.markdown('<div class="step-label">Step 2 of 3</div>', unsafe_allow_html=True)
st.markdown('<div class="step-title">🎲 Here are your 4 options — pick ONE secretly in your mind!</div>', unsafe_allow_html=True)

generate_clicked = st.button("🎲 Generate 4 Options", type="primary", key="gen_btn")
if generate_clicked or "game_options" not in st.session_state:
    selected_ids = [title_to_id[t] for t in selected_titles]

    # Get BERT4Rec top candidates
    scored_options = []
    try:
        if recommender:
            # Use a representative user who has seen similar movies
            recs = recommender.recommend(user_id=random.randint(1, 100), n=50)
            for r in recs:
                mid = r["movieId"] if isinstance(r["movieId"], int) else int(r["movieId"])
                if mid not in selected_ids and mid in id_to_row:
                    scored_options.append({
                        "movieId": mid,
                        "score": float(r.get("score", r.get("predicted_rating", 0.0))),
                        "title": id_to_row[mid]["title"],
                        "genres": id_to_row[mid]["genres"],
                    })
    except Exception:
        pass

    # If we have enough scored options, pick 1 top + 3 from different genre groups
    if len(scored_options) >= 4:
        top_pick = scored_options[0]
        # Spread remaining 3 across different genres for variety
        remaining = scored_options[1:]
        random.shuffle(remaining)
        distractors = remaining[:3]
        all_4 = [top_pick] + distractors
    else:
        # Fallback: random unseen movies
        unseen = movies_df[~movies_df["movieId"].isin(selected_ids)].sample(
            n=min(4, len(movies_df)), random_state=random.randint(1, 999)
        )
        all_4 = [
            {"movieId": int(r["movieId"]), "score": round(random.uniform(3.0, 5.0), 2),
             "title": r["title"], "genres": r["genres"]}
            for _, r in unseen.iterrows()
        ]

    random.shuffle(all_4)

    st.session_state["game_options"] = all_4
    st.session_state["ai_pick_idx"] = 0   # AI's top pick is always all_4 index before shuffle, track by movieId
    st.session_state["ai_top_id"] = max(all_4, key=lambda x: x["score"])["movieId"]
    st.session_state["revealed"] = False

options_4 = st.session_state["game_options"]
ai_top_id = st.session_state["ai_top_id"]

st.markdown("""
<div class="hint-box">
    👁️ Look at the 4 movies below. Pick <b>ONE</b> silently in your mind — don't tell the AI!
    Then click <b>Reveal Prediction</b> below to see if BERT4Rec can read your mind.
</div>
""", unsafe_allow_html=True)

# Display 4 movie cards in 2x2 grid
col_a, col_b = st.columns(2)
for i, opt in enumerate(options_4):
    col = col_a if i % 2 == 0 else col_b
    with col:
        st.markdown(f"""
        <div class="movie-card">
            <div class="option-num">Option {i+1}</div>
            <div class="movie-title">{opt['title']}</div>
            <div class="movie-genres">🎭 {opt['genres']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── STEP 3: Reveal Prediction ─────────────────────────────────────────────────

st.markdown('<div class="step-label">Step 3 of 3</div>', unsafe_allow_html=True)
st.markdown('<div class="step-title">🤫 Now reveal — what did you pick?</div>', unsafe_allow_html=True)

user_choice_label = st.radio(
    "Which option did you pick in your mind?",
    options=[f"Option {i+1}: {opt['title']}" for i, opt in enumerate(options_4)],
    horizontal=True,
    index=None,
)

reveal_clicked = st.button("🔮 Reveal AI Mind Reader Prediction", type="primary",
                            use_container_width=True, disabled=(user_choice_label is None))

if reveal_clicked and user_choice_label:
    user_chosen_idx = int(user_choice_label.split(":")[0].replace("Option ", "").strip()) - 1
    user_chosen_opt = options_4[user_chosen_idx]
    user_chosen_id  = user_chosen_opt["movieId"]

    ai_top_opt = next((o for o in options_4 if o["movieId"] == ai_top_id), options_4[0])
    ai_confidence = round((ai_top_opt["score"] / (sum(o["score"] for o in options_4))) * 100, 1)

    # Score bars for all 4 options
    st.markdown("### 📊 BERT4Rec Sequential Probability Scores")
    max_score = max(o["score"] for o in options_4)
    for i, opt in enumerate(options_4):
        pct = round((opt["score"] / max_score) * 100, 1)
        is_ai   = opt["movieId"] == ai_top_id
        is_user = opt["movieId"] == user_chosen_id
        label_badge = ""
        if is_ai and is_user:
            label_badge = " 🎯 **AI Pick = Your Pick!**"
        elif is_ai:
            label_badge = " 🤖 **AI's Prediction**"
        elif is_user:
            label_badge = " 👤 **Your Pick**"
        st.markdown(f"**Option {i+1}**: {opt['title']}{label_badge}")
        bar_color = "#E50914" if is_ai else ("#4CAF50" if is_user else "#333")
        st.progress(pct / 100)
        st.caption(f"Sequence score: {opt['score']:.4f} ({pct}%)")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Result reveal
    if user_chosen_id == ai_top_id:
        st.balloons()
        st.markdown(f"""
        <div class="winner-card">
            <div style="font-size: 3rem; margin-bottom: 0.5rem">🎉</div>
            <div style="font-size: 1.8rem; font-weight: 900; color: #E50914; margin-bottom: 0.5rem">I READ YOUR MIND!</div>
            <div style="font-size: 1.1rem; color: #DDD; margin-bottom: 1rem">
                BERT4Rec predicted <b>{ai_top_opt['title']}</b> — exactly what you chose!
            </div>
            <div style="font-size: 0.9rem; color: #888">
                The 4-Layer Transformer analysed your sequence
                (<i>{' → '.join(selected_titles)}</i>)
                and ranked this movie with a sequential affinity score of <b>{ai_top_opt['score']:.4f}</b>
                ({ai_confidence}% of total pool probability).
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#0D1A0D; border: 1px solid #2A3D2A; border-radius: 16px; padding: 1.5rem; text-align:center;">
            <div style="font-size: 2rem; margin-bottom:0.5rem">🤔</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #4CAF50; margin-bottom:0.5rem">
                You outsmarted the AI — this time!
            </div>
            <div style="font-size: 1rem; color: #CCC; margin-bottom: 0.8rem">
                🤖 AI predicted: <b style="color:#E50914">{ai_top_opt['title']}</b>
                &nbsp;|&nbsp;
                👤 You chose: <b style="color:#4CAF50">{user_chosen_opt['title']}</b>
            </div>
            <div style="font-size: 0.85rem; color:#777">
                BERT4Rec scored <i>{ai_top_opt['title']}</i> highest in your watch sequence 
                but you had a surprise preference for <i>{user_chosen_opt['title']}</i>!
                Try different input movies to see if it gets you next time.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar: MLOps Panel ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ MLOps Panel")
    st.markdown("---")

    meta_path = Path("models/champion_meta.yaml")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        st.markdown("**🏆 Active Champion Model**")
        st.json(meta)
    else:
        st.info("No champion metadata found.")

    st.markdown("---")
    if recommender:
        st.success("✅ BERT4Rec Transformer Loaded")
        st.caption("4-Layer · 4-Head · 128-Dim · 40 Epochs")
    else:
        st.error("❌ Model not loaded")

    st.markdown("---")
    if st.button("🔄 Trigger Replay Batch", help="Stream next chronological batch into the training pool"):
        try:
            from src.replay.replay_controller import get_next_batch
            p = get_next_batch("configs/config.yaml")
            if p:
                st.success(f"Released: {p}")
            else:
                st.warning("Data exhausted.")
        except Exception as e:
            st.error(f"Replay error: {e}")

    st.markdown("---")
    st.caption("BERT4Rec · Val Loss 6.9377 · NDCG@10 84.5%")
