# -------------------------------------------------
# 🏀 Hot Shot Props | Player AI Tracker (Single App)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog, playercareerstats
from datetime import datetime
import os, io, base64

# ---------- CONFIG ----------
st.set_page_config(page_title="Hot Shot Props | Player AI Tracker",
                   page_icon="🏀", layout="wide")
st.markdown("""
<style>
body {background-color:#0D1117;color:#E0E0E0;font-family:'Roboto',sans-serif;}
.sidebar .sidebar-content {background-color:#111827;}
h1,h2,h3 {color:#1E88E5;font-family:'Oswald',sans-serif;}
.stButton>button {background:#1E88E5;color:white;border-radius:8px;}
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")

# ---------- HELPERS ----------
@st.cache_data(ttl=86400)
def get_teams():
    return pd.DataFrame(teams.get_teams())

@st.cache_data(ttl=86400)
def get_players():
    return pd.DataFrame(players.get_active_players())

def get_player_logs(player_id, last_n=20):
    gl = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]
    return gl.head(last_n)

def get_career_stats(player_id):
    df = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
    return df

def compute_features(logs: pd.DataFrame):
    # basic averages
    feats = {}
    for col in ["PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"]:
        for n in [5,10,20]:
            feats[f"{col}_L{n}"] = logs.head(n)[col].mean()
    feats["GAMES_PLAYED"] = len(logs)
    return feats

def train_randomforest(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

def predict_stats(logs):
    feats = compute_features(logs)
    X = pd.DataFrame([feats]).fillna(0)
    preds = {}
    # mock training quickly using same player's logs
    for stat in ["PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"]:
        try:
            model = train_randomforest(logs[[stat]].join(X, how="left").fillna(0), stat)
            preds[stat] = float(model.predict(X)[0])
        except Exception:
            preds[stat] = logs[stat].mean()
    # combos
    preds["PA"] = preds["PTS"] + preds["AST"]
    preds["PR"] = preds["PTS"] + preds["REB"]
    preds["RA"] = preds["REB"] + preds["AST"]
    preds["PRA"] = preds["PTS"] + preds["REB"] + preds["AST"]
    return preds

def save_prediction(player_name, preds):
    now = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame([{"date": now, "player": player_name, **preds}])
    if os.path.exists(HISTORY_FILE):
        old = pd.read_csv(HISTORY_FILE)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()

# ---------- UI ----------
tab1, tab2 = st.tabs(["🔮 Player Analyzer", "📈 Model Tracker"])

with tab1:
    st.title("🔮 Player Stat Predictor")
    tdf = get_teams()
    team_name = st.selectbox("Select Team", tdf["full_name"].sort_values())
    pdf = get_players()
    players_team = pdf[pdf["team_id"] == tdf[tdf["full_name"]==team_name]["id"].values[0]]
    player_name = st.selectbox("Select Player", players_team["full_name"].sort_values())

    if player_name:
        player_id = int(pdf[pdf["full_name"]==player_name]["id"].values[0])
        with st.spinner(f"Fetching game logs for {player_name}..."):
            logs = get_player_logs(player_id)
        if logs.empty:
            st.error("No recent games found.")
        else:
            preds = predict_stats(logs)
            save_prediction(player_name, preds)

            # Visuals
            st.subheader(f"Predicted Stats for {player_name}")
            chart_data = {
                "Stat":["PTS","REB","AST","3PM","STL","BLK","TOV","MIN"],
                "Predicted":[preds["PTS"],preds["REB"],preds["AST"],preds["FG3M"],preds["STL"],preds["BLK"],preds["TOV"],preds["MIN"]],
                "SeasonAvg":[logs["PTS"].mean(),logs["REB"].mean(),logs["AST"].mean(),logs["FG3M"].mean(),logs["STL"].mean(),logs["BLK"].mean(),logs["TOV"].mean(),logs["MIN"].mean()]
            }
            df_chart = pd.DataFrame(chart_data)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_chart["Stat"], y=df_chart["SeasonAvg"], name="Season Avg", marker_color="#555"))
            fig.add_trace(go.Bar(x=df_chart["Stat"], y=df_chart["Predicted"], name="Predicted", marker_color="#1E88E5"))
            fig.update_layout(barmode="group", plot_bgcolor="#0D1117", paper_bgcolor="#0D1117", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

            # Radar Chart
            radar_stats = ["PTS","REB","AST","3PM","STL","BLK","TOV"]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(
                r=[preds.get(s,0) for s in radar_stats],
                theta=radar_stats, fill='toself', name='Predicted', line_color="#1E88E5"))
            fig2.add_trace(go.Scatterpolar(
                r=[logs[s].mean() for s in radar_stats],
                theta=radar_stats, fill='toself', name='Avg', line_color="#888"))
            fig2.update_layout(polar=dict(bgcolor="#0D1117"), showlegend=True,
                               paper_bgcolor="#0D1117", font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

            # Export stat card
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            b64 = base64.b64encode(buf.getbuffer()).decode("utf-8")
            href = f'<a href="data:file/png;base64,{b64}" download="{player_name}_prediction.png">📸 Download Stat Card</a>'
            st.markdown(href, unsafe_allow_html=True)

with tab2:
    st.title("📈 Model Tracker & Backtesting")
    hist = load_history()
    if hist.empty:
        st.info("No predictions logged yet.")
    else:
        st.dataframe(hist.tail(20))
        # summary stats
        avg_stats = hist.drop(columns=["date","player"]).mean().round(2)
        st.metric("Average Predicted PTS", avg_stats["PTS"])
        st.metric("Average Predicted REB", avg_stats["REB"])
        st.metric("Average Predicted AST", avg_stats["AST"])
        st.line_chart(hist[["PTS","REB","AST","PRA"]].tail(50))

st.sidebar.markdown("### ⚙️ Refresh")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()
