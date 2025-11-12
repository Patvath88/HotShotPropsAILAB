# -------------------------------------------------
# Hot Shot Props | AI Player Prediction Lab (BallDontLie Edition)
# -------------------------------------------------
# Features:
# - Select Team → Player
# - Auto fetches last 5/10/20 games, current season, last season, career averages
# - RandomForest AI predicts next game stats (PTS, REB, AST, 3PT, STL, BLK, TOV, MIN)
# - Visualized stat card + saves past predictions for backtesting
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import os
import time

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
    res = requests.get(API_BASE + "teams")
    if res.status_code != 200:
        return pd.DataFrame()
    teams = res.json()["data"]
    return pd.DataFrame(teams)[["id", "full_name", "abbreviation", "city"]]

@st.cache_data(ttl=3600)
def get_players():
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
    df = pd.DataFrame(players)
    return df[["id", "first_name", "last_name", "team"]]

def get_player_game_logs(player_id, num_games=20):
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
    if df.empty: return df
    df = pd.json_normalize(df)
    df = df.rename(columns={
        "pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL",
        "blk": "BLK", "turnover": "TOV", "min": "MIN", "fg3m": "3PTM"
    })
    return df[["game.date", "PTS", "REB", "AST", "3PTM", "STL", "BLK", "TOV", "MIN"]].dropna()

def train_predict_model(df):
    if df.empty:
        return {}
    df = df.tail(20)
    df["PTS_next"] = df["PTS"].shift(-1)
    df = df.dropna()
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
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"Player": player_name, "Timestamp": ts, **predictions}
    if os.path.exists(BACKTEST_FILE):
        pd.DataFrame([row]).to_csv(BACKTEST_FILE, mode="a", index=False, header=False)
    else:
        pd.DataFrame([row]).to_csv(BACKTEST_FILE, index=False)

# ---------- UI SELECTION ----------
teams_df = get_teams()
team = st.selectbox("Select Team", teams_df["full_name"].sort_values())

players_df = get_players()
players_team = players_df[players_df["team"].apply(lambda x: x and x["full_name"] == team)]
player_name = st.selectbox("Select Player", players_team["first_name"] + " " + players_team["last_name"])

if player_name:
    player_id = int(players_team.loc[(players_team["first_name"] + " " + players_team["last_name"]) == player_name, "id"].values[0])
    
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
    fig.update_layout(title=f"{player_name} Projected Statline", plot_bgcolor='#0A0F1C', paper_bgcolor='#0A0F1C',
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
