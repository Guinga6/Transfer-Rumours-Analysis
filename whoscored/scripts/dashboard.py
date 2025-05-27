# Step-by-step guide to create an interactive dashboard using Streamlit

# 1. Install the necessary libraries (run these commands in your terminal or command prompt)
# pip install pandas openpyxl streamlit plotly

# 2. Save your Excel file in your project folder and use the following code:

import streamlit as st
import pandas as pd
import plotly.express as px

# --- Load and preprocess data ---
@st.cache_data

def load_data():
    xls = pd.ExcelFile(r"c:\Users\Utilisateur\Desktop\dataset_transfers\premier_league_players_20250521_1529.xlsx")
    teams = xls.sheet_names
    df_list = []
    for team in teams:
        df = xls.parse(team)
        df["Team"] = team
        df_list.append(df)
    all_players = pd.concat(df_list, ignore_index=True)
    all_players.columns = all_players.columns.str.strip()
    all_players = all_players.rename(columns={"Player": "PlayerInfo"})
    # Extract player name
    all_players["Player"] = all_players["PlayerInfo"].str.extract(r"\d+(.*?)\d")
    all_players["Player"] = all_players["Player"].fillna(all_players["PlayerInfo"]).str.strip()
    return all_players

players_df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("Filters")
teams = sorted(players_df["Team"].unique())
selected_team = st.sidebar.selectbox("Select a Team", ["All"] + teams)

if selected_team != "All":
    filtered_df = players_df[players_df["Team"] == selected_team]
else:
    filtered_df = players_df

# --- Main Interface ---
st.title("Premier League Players Dashboard")

# Show player table
st.subheader("Player Stats Table")
st.dataframe(filtered_df[["Player", "Team", "Goals", "Assists", "SpG", "PS%", "AerialsWon", "MotM"]])

# Top performers
st.subheader("Top Scorers")
top_goals = filtered_df.sort_values(by="Goals", ascending=False).head(10)
fig_goals = px.bar(top_goals, x="Player", y="Goals", color="Team", title="Top 10 Goal Scorers")
st.plotly_chart(fig_goals)

st.subheader("Top Assists")
top_assists = filtered_df.sort_values(by="Assists", ascending=False).head(10)
fig_assists = px.bar(top_assists, x="Player", y="Assists", color="Team", title="Top 10 Assist Providers")
st.plotly_chart(fig_assists)

# Player comparison
st.subheader("Compare Two Players")
players = filtered_df["Player"].dropna().unique()
player1 = st.selectbox("Select First Player", players, index=0)
player2 = st.selectbox("Select Second Player", players, index=1)

comp_df = filtered_df[filtered_df["Player"].isin([player1, player2])][
    ["Player", "Goals", "Assists", "SpG", "PS%", "AerialsWon", "MotM"]
].set_index("Player")

st.dataframe(comp_df.transpose(), use_container_width=True)

# --- Instructions to Run ---
# 1. Save this script in a file, e.g., dashboard.py
# 2. Place the Excel file in the same folder as this script
# 3. Open terminal and navigate to the folder
# 4. Run the app using: streamlit run dashboard.py

# Streamlit will open a browser window showing your dashboard
