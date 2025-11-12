# -------------------------------------------------
# Hot Shot Props | AI Player Prediction Lab (BallDontLie Edition - Stable)
# -------------------------------------------------
# Features:
# - Team → Player selection
# - BallDontLie API for data (with instant static fallback)
# - RandomForest predicts next game statline
# - Blue/Black ESPN-style visuals
# - Auto-backtesting log stored locally
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

# ---------- UTILS ----------
@st.cache_data(ttl=3600)
def get_teams():
    """Fetch teams from BallDontLie with a static fallback."""
    try:
        res = requests.get(API_BASE + "teams", timeout=10)
        if res.status_code == 200:
            data = res.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                if "full_name" not in df.columns:
                    df["full_name"] = df["city"] + " " + df["name"]
                return df[["id", "full_name", "abbreviation", "city", "conference", "division"]]
    except Exception as e:
        print(f"⚠️ BallDontLie team fetch failed: {e}")

    # --- fallback static team list ---
    fallback = [
        {"id": 1, "full_name": "Atlanta Hawks"}, {"id": 2, "full_name": "Boston Celtics"},
        {"id": 3, "full_name": "Brooklyn Nets"}, {"id": 4, "full_name": "Charlotte Hornets"},
        {"id": 5, "full_name": "Chicago Bulls"}, {"id": 6, "full_name": "Cleveland Cavaliers"},
        {"id": 7, "full_name": "Dallas Mavericks"}, {"id": 8, "full_name": "Denver Nuggets"},
        {"id": 9, "full_name": "Detroit Pistons"}, {"id":10, "full_name": "Golden State Warriors"},
        {"id":11, "full_name": "Houston Rockets"}, {"id":12, "full_name": "Indiana Pacers"},
        {"id":13, "full_name": "Los Angeles Clippers"}, {"id":14, "full_name": "Los Angeles Lakers"},
        {"id":15, "full_name": "Memphis Grizzlies"}, {"id":16, "full_name": "Miami Heat"},
        {"id":17, "full_name": "Milwaukee Bucks"}, {"id":18, "full_name": "Minnesota Timberwolves"},
        {"id":19, "full_name": "New Orleans Pelicans"}, {"id":20, "full_name": "New York Knicks"},
        {"id":21, "full_name": "Oklahoma City Thunder"}, {"id":22, "full_name": "Orlando Magic"},
        {"id":23, "full_name": "Philadelphia 76ers"}, {"id":24, "full_name": "Phoenix Suns"},
        {"id":25, "full_name": "Portland Trail Blazers"}, {"id":26, "full_name": "Sacramento Kings"},
        {"id":27, "full_name": "San Antonio Spurs"}, {"id":28, "full_name": "Toronto Raptors"},
        {"id":29, "full_name": "Utah Jazz"}, {"id":30, "full_name": "Washington Wizards"}
    ]
    return pd.DataFrame(fallback)

@st.cache_data(ttl=3600)
def get_players():
    """Fetch all players (paged)."""
    players = []
    page = 1
    while True:
        res = requests.get(API_BASE + f"players?page={page}&per_page=100")
        if res.status_code != 200:
            break
        data = res.json()["data"]
        if not data:
            break
        players.extend(data)
        page += 1
        if len(data) < 100:
            break
    df = pd.DataFrame(players)
    return df[["id", "first_name", "last_name", "team"]]

def get_player_game_logs(player_id, num_games=20):
    """Pull last N games for player."""
    logs = []
    page = 1
    while len(logs) < num_games:
        res = requests.get(API_BASE + f"stats?player_ids[]={player_id}&per_page=100&page={page}")
        if res.status_code != 200:
            break
        data = res.json()["data"]
        if not data:
            break
        logs.extend(data)
        page += 1
        if len(data) < 100:
            break
    df = pd.DataFrame(logs)
    if df.empty:
        return df
    df = pd.json_normalize(df)
    df = df.rename(columns={
        "pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL",
        "blk": "BLK", "turnover": "TOV", "min": "MIN", "fg3m": "3PTM"
    })
    cols = ["game.date", "PTS", "REB", "AST", "3PTM", "STL", "BLK", "TOV", "MIN"]
    df = df[[c for c in cols if c in df.columns]].dropna()
    return df

def train_predict_model(df):
    """Train RandomForest and return predicted next statline."""
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
    prediction = model.predict(next_input)[0]
    summary = {
        "PTS": round(prediction, 1),
        "REB": round(df["REB"].mean(), 1),
        "AST": round(df["AST"].mean(), 1),
        "3PTM": round(df["3PTM"].mean(), 1),
        "STL": round(df["STL"].mean(), 1),
        "BLK": round(df["BLK"].mean(), 1),
        "TOV": round(df["TOV"].mean(), 1),
        "MIN": round(df["MIN"].mean(), 1)
    }
    return summary

def record_backtest(player_name, predictions):
    """Save prediction to CSV."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"Player": player_name, "Timestamp": ts, **predictions}
    if os.path.exists(BACKTEST_FILE):
        pd.DataFrame([row]).to_csv(BACKTEST_FILE, mode="a", index=False, header=False)
    else:
        pd.DataFrame([row]).to_csv(BACKTEST_FILE, index=False)

# ---------- UI SELECTION ----------
teams_df = get_teams()
if teams_df.empty:
    st.warning("⚠️ Could not fetch live teams — using fallback.")
    teams_df = pd.DataFrame([{"id": 1, "full_name": "Boston Celtics"}, {"id": 2, "full_name": "Denver Nuggets"}])

team = st.selectbox("Select Team", sorted(teams_df["full_name"].unique()))

players_df = get_players()
players_team = players_df[players_df["team"].apply(lambda x: isinstance(x, dict) and x.get("full_name") == team)]

if players_team.empty:
    st.warning("⚠️ No players found for that team. Try another.")
    st.stop()

player_name = st.selectbox("Select Player", players_team["first_name"] + " " + players_team["last_name"])

# ---------- RUN MODEL ----------
if player_name:
    player_id = int(players_team.loc[
        (players_team["first_name"] + " " + players_team["last_name"]) == player_name,
        "id"
    ].values[0])

    with st.spinner(f"Fetching {player_name}'s recent games..."):
        logs_df = get_player_game_logs(player_id, num_games=20)
        if logs_df.empty:
            st.error("No recent game logs found.")
            st.stop()

    preds = train_predict_model(logs_df)
    record_backtest(player_name, preds)

    # ---------- VISUAL CARD ----------
    st.subheader(f"📊 {player_name} | Predicted Next Game Stats")
    fig = go.Figure()
    categories = list(preds.keys())
    values = list(preds.values())
    fig.add_trace(go.Bar(x=categories, y=values, marker_color='#29B6F6'))
    fig.update_layout(title=f"{player_name} Projected Statline",
                      plot_bgcolor='#0A0F1C', paper_bgcolor='#0A0F1C',
                      font=dict(color='white'), height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ Prediction Complete for {player_name}")
    st.dataframe(pd.DataFrame([preds]), use_container_width=True)

# ---------- SUMMARY PAGE ----------
st.divider()
st.subheader("📈 Model Backtesting Summary")

if os.path.exists(BACKTEST_FILE):
    bt = pd.read_csv(BACKTEST_FILE)
    st.dataframe(bt.tail(10), use_container_width=True)
else:
    st.info("No backtest data yet. Make a prediction to start tracking accuracy!")
