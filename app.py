# -------------------------------------------------
# 🏀 Hot Shot Props — NBA Player AI Prediction Lab (No NBAStatsHTTP Dependency)
# -------------------------------------------------
# Fully functional single-page app:
# - Team → Player instant selection
# - RandomForest AI predictions
# - Timeout-safe nba_api calls
# - Blue/Black modern UI
# - Saves backtesting results automatically
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
import plotly.graph_objects as go
import os
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🏀 Hot Shot Props | AI Player Prediction Lab",
    page_icon="🏀",
    layout="wide"
)

st.markdown("""
<style>
    body { background-color: #000000; color: white; }
    .stApp { background-color: #000000; }
    h1,h2,h3,h4 { color: #1E88E5; font-family: 'Montserrat', sans-serif; }
    .stat-card {
        background-color: #0D47A1;
        border-radius: 10px;
        padding: 12px;
        color: white;
        text-align: center;
        box-shadow: 0px 0px 10px #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏀 Hot Shot Props — NBA Player AI Prediction Lab")
st.markdown("AI-powered NBA player stat prediction engine using RandomForest regression and backtesting tracking.")

# -------------------------------------------------
# 🔹 Helper: Safe fetch wrapper to avoid timeouts
# -------------------------------------------------
def safe_fetch(fn, *args, retries=3, wait=2, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            st.warning(f"Retry {attempt+1}/{retries} failed: {e}")
            time.sleep(wait)
    st.error("API fetch failed after multiple retries.")
    return None

# -------------------------------------------------
# 🔹 Cached Functions
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_teams_df():
    return pd.DataFrame(teams.get_teams())

@st.cache_data(ttl=3600)
def get_roster(team_name):
    """Fetch current roster for a given team."""
    tdf = get_teams_df()
    team_id = int(tdf.loc[tdf["full_name"] == team_name, "id"].values[0])
    roster_obj = safe_fetch(commonteamroster.CommonTeamRoster, team_id=team_id, season="2024-25")
    if roster_obj is None:
        return pd.DataFrame(columns=["PLAYER", "PLAYER_ID"])
    try:
        return roster_obj.get_data_frames()[0][["PLAYER", "PLAYER_ID"]]
    except Exception:
        return pd.DataFrame(columns=["PLAYER", "PLAYER_ID"])

@st.cache_data(ttl=3600)
def get_player_gamelog(player_id, limit=50):
    """Fetch player's game log."""
    gl_obj = safe_fetch(playergamelog.PlayerGameLog, player_id=player_id, season="2024-25")
    if gl_obj is None:
        return pd.DataFrame()
    try:
        return gl_obj.get_data_frames()[0].head(limit)
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------
# 🔹 AI Model: RandomForest Regressor
# -------------------------------------------------
def train_rf_model(df, target_col):
    """Train a RandomForest model."""
    if df.empty or target_col not in df.columns:
        return None, None

    features = ["MIN", "FGA", "FG3A", "REB", "AST", "STL", "BLK", "TOV", "PTS"]
    available = [f for f in features if f in df.columns]
    df = df[available].dropna()

    if df.shape[0] < 5:
        return None, None

    X = df.drop(columns=[target_col]) if target_col in df.columns else df
    y = df[target_col] if target_col in df.columns else None
    if y is None or y.nunique() <= 1:
        return None, None

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model, available

def predict_stats(df):
    """Predict multiple stats."""
    projections = {}
    for stat in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN"]:
        model, features = train_rf_model(df, stat)
        if model:
            pred = round(model.predict([df[features].mean().values])[0], 1)
            projections[stat] = pred
        else:
            projections[stat] = np.nan
    return projections

# -------------------------------------------------
# 🔹 Sidebar
# -------------------------------------------------
tdf = get_teams_df()
team_name = st.sidebar.selectbox("Select Team", sorted(tdf["full_name"].unique()))

roster_df = get_roster(team_name)
player_name = st.sidebar.selectbox("Select Player", roster_df["PLAYER"] if not roster_df.empty else [])

if not roster_df.empty and player_name:
    player_id = int(roster_df.loc[roster_df["PLAYER"] == player_name, "PLAYER_ID"].values[0])
else:
    player_id = None

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# -------------------------------------------------
# 🔹 Main Prediction Flow
# -------------------------------------------------
if player_id:
    with st.spinner(f"Fetching data for {player_name}..."):
        gamelog = get_player_gamelog(player_id)
        if gamelog.empty:
            st.warning("No recent game logs found for this player.")
        else:
            st.success(f"Loaded {len(gamelog)} games for {player_name}")

    if not gamelog.empty:
        numeric_cols = gamelog.select_dtypes(include=np.number).columns.tolist()
        gamelog = gamelog[numeric_cols].fillna(0)

        with st.spinner("Running AI prediction model..."):
            projections = predict_stats(gamelog)

        # --- Display projection cards ---
        st.markdown(f"## 🎯 AI Projection Card: {player_name} ({team_name})")
        cols = st.columns(4)
        stat_keys = list(projections.keys())

        for i, col in enumerate(cols):
            subset = stat_keys[i*2:(i+1)*2]
            for stat in subset:
                val = projections[stat]
                display_val = "—" if np.isnan(val) else val
                col.markdown(
                    f"""
                    <div class="stat-card">
                        <h3>{stat}</h3>
                        <h1>{display_val}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # --- Visualization ---
        st.markdown("### 📈 Historical Trend vs AI Projection")
        if "PTS" in gamelog.columns:
            hist_pts = gamelog["PTS"].iloc[:20][::-1]
            dates = list(range(len(hist_pts)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=hist_pts, mode="lines+markers", name="Past PTS", line=dict(color="skyblue")))
            if not np.isnan(projections["PTS"]):
                fig.add_trace(go.Scatter(x=[len(dates)], y=[projections["PTS"]], mode="markers", name="Predicted", marker=dict(size=12, color="orange")))
            fig.update_layout(
                template="plotly_dark",
                title=f"{player_name} — Recent Scoring Trend & AI Prediction",
                xaxis_title="Game Index (Most Recent Last)",
                yaxis_title="Points",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Backtesting log ---
        DATA_DIR = "data"
        os.makedirs(DATA_DIR, exist_ok=True)
        history_file = os.path.join(DATA_DIR, "history.csv")

        entry = {
            "player": player_name,
            "team": team_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            **projections
        }

        try:
            if os.path.exists(history_file):
                hist = pd.read_csv(history_file)
                hist = pd.concat([hist, pd.DataFrame([entry])], ignore_index=True)
            else:
                hist = pd.DataFrame([entry])
            hist.to_csv(history_file, index=False)
            st.success("✅ Projection saved for backtesting.")
        except Exception as e:
            st.warning(f"Could not save history: {e}")

else:
    st.info("Select a team and player from the sidebar to begin.")
