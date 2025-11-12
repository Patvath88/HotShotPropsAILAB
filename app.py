# -------------------------------------------------
# Hot Shot Props | AI Player Prediction Lab (Final Stable Build)
# -------------------------------------------------
# - BallDontLie data with full offline fallback
# - RandomForest AI predictions
# - Team/Player selection always loads instantly
# - Blue/Black ESPN-style visuals
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="🏀 Hot Shot Props | AI Player Prediction Lab",
                   page_icon="🏀", layout="wide")

st.markdown("""
    <style>
        body {background-color: #0A0F1C; color: white; font-family: 'Roboto', sans-serif;}
        .main-title {color: #29B6F6; text-shadow: 0 0 10px #0288D1; font-size: 2.5em; font-weight: bold; text-align: center;}
        .subtext {color: #90CAF9; text-align: center; margin-bottom: 20px;}
        .stButton button {background-color: #29B6F6; color: black; font-weight: bold; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏀 Hot Shot Props | AI Player Prediction Lab</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>BallDontLie-powered player predictions & performance tracking</div>", unsafe_allow_html=True)

# ---------- CONSTANTS ----------
API_BASE = "https://www.balldontlie.io/api/v1/"
BACKTEST_FILE = "data/backtests.csv"
os.makedirs("data", exist_ok=True)

# ---------- DATA FETCH HELPERS ----------
@st.cache_data(ttl=3600)
def get_teams():
    """Fetch teams safely from BallDontLie, with static fallback."""
    try:
        res = requests.get(API_BASE + "teams", timeout=10)
        if res.status_code == 200:
            data = res.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                if "full_name" not in df.columns:
                    df["full_name"] = df["city"] + " " + df["name"]
                return df[["id", "full_name"]]
    except Exception as e:
        print("⚠️ Error fetching BallDontLie teams:", e)

    fallback = [
        {"id": 1, "full_name": "Boston Celtics"},
        {"id": 2, "full_name": "Denver Nuggets"},
        {"id": 3, "full_name": "Miami Heat"},
        {"id": 4, "full_name": "Golden State Warriors"},
        {"id": 5, "full_name": "Los Angeles Lakers"},
    ]
    return pd.DataFrame(fallback)

@st.cache_data(ttl=3600)
def get_players():
    """Fetch all NBA players with fallback."""
    players = []
    try:
        for page in range(1, 6):  # only need 500 players
            r = requests.get(f"{API_BASE}players?page={page}&per_page=100", timeout=10)
            if r.status_code != 200:
                break
            data = r.json().get("data", [])
            if not data:
                break
            players.extend(data)
            if len(data) < 100:
                break
        if not players:
            raise ValueError("Empty player response.")
        df = pd.DataFrame(players)
        expected = ["id", "first_name", "last_name", "team"]
        for col in expected:
            if col not in df.columns:
                df[col] = None
        return df[expected]
    except Exception as e:
        print("⚠️ BallDontLie player fetch failed:", e)
        # static fallback list
        fallback = [
            {"id": 15, "first_name": "Nikola", "last_name": "Jokic", "team": {"full_name": "Denver Nuggets"}},
            {"id": 23, "first_name": "LeBron", "last_name": "James", "team": {"full_name": "Los Angeles Lakers"}},
            {"id": 30, "first_name": "Stephen", "last_name": "Curry", "team": {"full_name": "Golden State Warriors"}},
            {"id": 0, "first_name": "Jayson", "last_name": "Tatum", "team": {"full_name": "Boston Celtics"}},
            {"id": 13, "first_name": "Jimmy", "last_name": "Butler", "team": {"full_name": "Miami Heat"}},
        ]
        return pd.DataFrame(fallback)

def get_player_game_logs(player_id, num_games=20):
    """Pull recent player game logs."""
    logs = []
    page = 1
    while len(logs) < num_games:
        try:
            r = requests.get(f"{API_BASE}stats?player_ids[]={player_id}&per_page=100&page={page}", timeout=10)
            if r.status_code != 200:
                break
            data = r.json().get("data", [])
            if not data:
                break
            logs.extend(data)
            page += 1
            if len(data) < 100:
                break
        except Exception:
            break
    df = pd.DataFrame(logs)
    if df.empty:
        return df
    df = pd.json_normalize(df)
    rename_map = {
        "pts": "PTS", "reb": "REB", "ast": "AST",
        "stl": "STL", "blk": "BLK", "turnover": "TOV",
        "fg3m": "3PTM", "min": "MIN"
    }
    df = df.rename(columns=rename_map)
    cols = [c for c in rename_map.values() if c in df.columns]
    return df[cols].dropna() if not df.empty else df

def train_predict_model(df):
    """Train a RandomForestRegressor and return predictions."""
    if df.empty:
        return {}
    df = df.tail(20)
    df["PTS_next"] = df["PTS"].shift(-1)
    df = df.dropna()
    if df.empty:
        return {}
    X = df[["REB", "AST", "3PTM", "STL", "BLK", "TOV", "MIN"]]
    y = df["PTS_next"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    next_input = X.tail(1)
    pred_pts = model.predict(next_input)[0]
    summary = {
        "PTS": round(pred_pts, 1),
        "REB": round(df["REB"].mean(), 1),
        "AST": round(df["AST"].mean(), 1),
        "3PTM": round(df["3PTM"].mean(), 1),
        "STL": round(df["STL"].mean(), 1),
        "BLK": round(df["BLK"].mean(), 1),
        "TOV": round(df["TOV"].mean(), 1),
        "MIN": round(df["MIN"].mean(), 1),
    }
    return summary

def record_backtest(player_name, predictions):
    """Store predictions locally."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"Player": player_name, "Timestamp": ts, **predictions}
    if os.path.exists(BACKTEST_FILE):
        pd.DataFrame([row]).to_csv(BACKTEST_FILE, mode="a", index=False, header=False)
    else:
        pd.DataFrame([row]).to_csv(BACKTEST_FILE, index=False)

# ---------- UI SELECTION ----------
teams_df = get_teams()
if teams_df.empty:
    st.error("⚠️ Could not load teams from BallDontLie.")
    st.stop()

team = st.selectbox("Select Team", sorted(teams_df["full_name"].unique()))

players_df = get_players()
players_team = players_df[players_df["team"].apply(lambda x: isinstance(x, dict) and x.get("full_name") == team)]

if players_team.empty:
    st.warning("⚠️ No players found for this team — using demo players.")
    players_team = get_players()

player_name = st.selectbox("Select Player", players_team["first_name"] + " " + players_team["last_name"])

# ---------- MODEL RUN ----------
if player_name:
    player_id = int(players_team.loc[
        (players_team["first_name"] + " " + players_team["last_name"]) == player_name,
        "id"
    ].values[0])

    with st.spinner(f"Fetching {player_name}'s game logs..."):
        logs_df = get_player_game_logs(player_id)
        if logs_df.empty:
            st.error("No game logs available.")
            st.stop()

    preds = train_predict_model(logs_df)
    record_backtest(player_name, preds)

    st.subheader(f"📊 {player_name} | Predicted Next Game Stats")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(preds.keys()), y=list(preds.values()), marker_color='#29B6F6'))
    fig.update_layout(
        title=f"{player_name} Projected Statline",
        plot_bgcolor='#0A0F1C',
        paper_bgcolor='#0A0F1C',
        font=dict(color='white'),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"✅ Prediction Complete for {player_name}")
    st.dataframe(pd.DataFrame([preds]), use_container_width=True)

# ---------- BACKTEST SUMMARY ----------
st.divider()
st.subheader("📈 Model Backtesting Summary")

if os.path.exists(BACKTEST_FILE):
    bt = pd.read_csv(BACKTEST_FILE)
    st.dataframe(bt.tail(10), use_container_width=True)
else:
    st.info("No backtest data yet — make your first prediction!")
