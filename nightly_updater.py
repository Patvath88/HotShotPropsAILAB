# -------------------------------------------------
# 🕒 Hot Shot Props AI Lab — Nightly Accuracy Updater
# -------------------------------------------------
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import os

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")

def fetch_actuals(player_name, game_date):
    try:
        p = players.find_players_by_full_name(player_name)
        if not p:
            return None
        player_id = p[0]["id"]
        logs = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
        match = logs[logs["GAME_DATE"].dt.strftime("%Y-%m-%d") == game_date]
        if match.empty:
            return None
        g = match.iloc[0]
        return {
            "PTS": g["PTS"],
            "REB": g["REB"],
            "AST": g["AST"],
            "FG3M": g["FG3M"],
            "STL": g["STL"],
            "BLK": g["BLK"],
            "TOV": g["TOV"],
            "MIN": g["MIN"],
        }
    except Exception:
        return None

def update_history():
    if not os.path.exists(HISTORY_FILE):
        print("❌ No history.csv found.")
        return

    df = pd.read_csv(HISTORY_FILE)
    updated = False

    for i, row in df.iterrows():
        if pd.isna(row.get("PTS")) or pd.notna(row.get("Actual_PTS")):
            continue

        actuals = fetch_actuals(row["player"], row["date"])
        if actuals:
            for stat, val in actuals.items():
                df.loc[i, f"Actual_{stat}"] = val
                df.loc[i, f"Error_{stat}"] = round(row.get(stat, 0) - val, 2)
            updated = True
            print(f"✅ Updated {row['player']} ({row['date']})")

        time.sleep(0.3)  # avoid hitting rate limit

    if updated:
        df.to_csv(HISTORY_FILE, index=False)
        print("💾 Saved updated accuracy file.")
    else:
        print("🔹 No updates needed today.")

if __name__ == "__main__":
    print("🕒 Running Hot Shot Props nightly updater...")
    update_history()
    print("✅ Done!")
