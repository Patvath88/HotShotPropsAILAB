# -------------------------------------------------
# 🏀 Hot Shot Props — NBA Player AI Prediction Lab
# -------------------------------------------------
# Features:
# - Team → Player search
# - RandomForest-based stat projections
# - Visualization + historical backtesting
# - Instant loading via Streamlit caching
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog, commonteamroster
import plotly.graph_objects as go
import os

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="🏀 Hot Shot Props | AI Player Prediction Lab",
    page_icon="🏀",
    layout="wide"
)

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

st.title("🏀 Hot Shot Props — NBA Player AI Prediction Lab")
st.markdown("AI-powered NBA player stat prediction engine with RandomForest regression and backtesting tracking.")

# --------------------- CACHING FUNCTIONS ---------------------
@st.cache_data(ttl=3600)
def get_teams_df():
    """Fetch all NBA teams."""
    return pd.DataFrame(teams.get_teams())

@st.cache_data(ttl=3600)
def get_roster(team_name):
    """Fetch current roster for a given team."""
    tdf = get_teams_df()
    team_id = tdf.loc[tdf["full_name"] == team_name, "id"].values[0]
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season="2024-25").get_data_frames()[0]
    return roster[["PLAYER", "PLAYER_ID"]]

@st.cache_data(ttl=3600)
def get_player_gamelog(player_id, limit=50):
    """Fetch player game log for last 50 games (extending to previous season if needed)."""
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season="2024-25").get_data_frames()[0]
        return gl.head(limit)
    except Exception:
        return pd.DataFrame()

# --------------------- AI MODEL ---------------------
def train_rf_model(df, target_col):
    """Train RandomForest model using historical game stats."""
    if df.empty or target_col not in df.columns:
        return None, None

    features = ["MIN", "FGA", "FG3A", "REB", "AST", "STL", "BLK", "TOV", "PTS"]
    df = df[features].dropna()
    if df.shape[0] < 5:
        return None, None

    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model, features

def predict_stats(df):
    """Generate predictions for key stats."""
    projections = {}
    for stat in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN"]:
        model, features = train_rf_model(df, stat)
        if model is not None:
            latest = df[features].mean().values.reshape(1, -1)
            pred = round(model.predict(latest)[0], 1)
            projections[stat] = pred
        else:
            projections[stat] = np.nan
    return projections

# --------------------- SIDEBAR CONTROLS ---------------------
tdf = get_teams_df()
team_name = st.sidebar.selectbox("Select Team", sorted(tdf["full_name"].unique()))

roster_df = get_roster(team_name)
player_name = st.sidebar.selectbox("Select Player", roster_df["PLAYER"])

player_id = roster_df.loc[roster_df["PLAYER"] == player_name, "PLAYER_ID"].values[0]

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Player Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# --------------------- FETCH & PREDICT ---------------------
with st.spinner("Fetching player data..."):
    gamelog = get_player_gamelog(player_id)
    if gamelog.empty:
        st.warning("No recent game logs found for this player.")
    else:
        st.success(f"Loaded {len(gamelog)} games for {player_name}")

if not gamelog.empty:
    # Prepare data
    numeric_cols = gamelog.select_dtypes(include=np.number).columns.tolist()
    gamelog = gamelog[numeric_cols].fillna(0)

    # Run projections
    with st.spinner("Running AI prediction model..."):
        projections = predict_stats(gamelog)

    # --------------------- DISPLAY PROJECTION CARD ---------------------
    st.markdown(f"## 🎯 AI Projection Card: {player_name} ({team_name})")

    col1, col2, col3, col4 = st.columns(4)
    stat_keys = list(projections.keys())

    for i, col in enumerate([col1, col2, col3, col4]):
        subset = stat_keys[i*2:(i+1)*2]
        for stat in subset:
            col.markdown(
                f"""
                <div class="stat-card">
                    <h3>{stat}</h3>
                    <h1>{projections[stat]}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------- VISUAL CHART ---------------------
    st.markdown("### 📈 Historical Trend vs AI Projection")
    hist_pts = gamelog["PTS"].iloc[:20][::-1]
    dates = list(range(len(hist_pts)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=hist_pts, mode="lines+markers", name="Past PTS", line=dict(color="skyblue")))
    fig.add_trace(go.Scatter(x=[len(dates)], y=[projections["PTS"]], mode="markers", name="Predicted", marker=dict(size=12, color="orange")))
    fig.update_layout(
        template="plotly_dark",
        title=f"{player_name} — Recent Scoring Trend & AI Prediction",
        xaxis_title="Game Index (Most Recent Last)",
        yaxis_title="Points",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------- BACKTESTING LOG ---------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
history_file = os.path.join(DATA_DIR, "history.csv")

if not gamelog.empty:
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
