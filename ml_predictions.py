import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from preprocess_data_prediction import process_nba_data
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv
from geopy.distance import geodesic
import joblib

# Load the trained model
model_path = "./best_xgb_model_20250404.pkl"
best_xgb = joblib.load(model_path)

def create_engine_from_env():
    """
    Creates a SQLAlchemy engine using the DATABASE_URL stored in the environment.
    
    Returns:
        engine: SQLAlchemy engine connected to the PostgreSQL database.
    """
    load_dotenv()  # Load environment variables from .env file
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable not set.")
    # Ensure the URL uses "postgresql://" instead of "postgres://"
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    return create_engine(DATABASE_URL, echo=False)

def get_new_games():
    # Get list of upcoming nba games
    # Use today's date and calculate the end date 7 days from now
    today = datetime.today()
    today_date = today.date()
    end_date = (today + timedelta(days=7)).date()
    # Build SQL query to retrieve games scheduled between today and the next 6 days.
    # Cast the "date" column to DATE to ensure proper comparison.
    engine = create_engine_from_env()
    query = text("""
        SELECT id, date, season, home_team_full_name, visitor_team_full_name, home_team_city, postseason
        FROM nba_games
        WHERE CAST(date AS DATE) >= :today_date AND CAST(date AS DATE) < :end_date
    """)
    with engine.connect() as conn:
        upcoming_games = pd.read_sql(query, conn, params={"today_date": today_date, "end_date": end_date})
    # Rename columns as required:
    upcoming_games.rename(columns={
        "id": "game_id",
        "home_team_full_name": "home_team",
        "visitor_team_full_name": "away_team",
        "home_team_city": "game_city",
        "postseason": "game_postseason"
    }, inplace=True)

    return upcoming_games

def get_active_players():
    """
    Retrieves all active NBA players from the 'nba_active_players' table.
    
    Returns:
        pd.DataFrame: DataFrame containing active NBA players data.
    """
    engine = create_engine_from_env()
    query = "SELECT * FROM nba_active_players"
    df_players = pd.read_sql(query, engine)
    return df_players

def get_injured_players():
    """
    Retrieves all injured player data from the 'nba_player_injuries' table.
    
    Returns:
        pd.DataFrame: DataFrame containing injured NBA players data.
    """
    engine = create_engine_from_env()
    query = "SELECT * FROM nba_player_injuries"
    df_injuries = pd.read_sql(query, engine)
    return df_injuries

def get_nba_games():
    """
    Retrieves all NBA games from the 'nba_games' table.
    
    Returns:
        pd.DataFrame: DataFrame containing all NBA games data.
    """
    engine = create_engine_from_env()
    query = "SELECT * FROM nba_games"
    nba_games = pd.read_sql(query, engine)
    # Drop duplicated rows
    nba_games = nba_games.drop_duplicates()
    return nba_games

def upsert_game_predictions(upcoming_games_df):
    """
    Upserts prediction rows from upcoming_games_df into the nba_predictions table.
    For each row, if a record with the same game_id exists, only updates the
    predicted_winner and prediction_date columns. Otherwise, inserts a new record.
    
    The function also ensures that the "game_postseason" field is a Boolean.
    
    Parameters:
        upcoming_games_df (DataFrame): DataFrame containing at least:
            - game_id (unique key)
            - predicted_winner
            - prediction_date
            - game_postseason (as a string, e.g. "true" or "false")
    """
    engine = create_engine_from_env()
    metadata = MetaData()
    # Reflect the existing nba_predictions table
    nba_predictions = Table("nba_predictions", metadata, autoload_with=engine)
    
    # Define columns that should not be updated (immutable)
    skip_update = {"game_id", "date", "season", "game_postseason", "game_city"}
    
    with engine.begin() as conn:
        for idx, row in upcoming_games_df.iterrows():
            row_data = row.to_dict()
            
            # Convert game_postseason to boolean if it's a string.
            if "game_postseason" in row_data:
                if isinstance(row_data["game_postseason"], str):
                    row_data["game_postseason"] = True if row_data["game_postseason"].lower() == "true" else False
            
            stmt = insert(nba_predictions).values(**row_data)
            
            # Build update dictionary for columns that might change.
            update_dict = {
                key: getattr(stmt.excluded, key)
                for key in row_data.keys() if key not in skip_update
            }
            
            stmt = stmt.on_conflict_do_update(
                index_elements=["game_id"],
                set_=update_dict
            )
            conn.execute(stmt)
    print("Upsert of game predictions completed.")

def insert_team_features(df):

    # -------------------------------------------------------------------------
    # 1. FILTER OUT ROWS WHERE game_won IS NaN
    # -------------------------------------------------------------------------
    df = df[df["game_won"].notna()]

    # -------------------------------------------------------------------------
    # 2. SORT BY team_id ASC, date DESC
    # -------------------------------------------------------------------------
    df = df.sort_values(["team_id", "date"], ascending=[True, False])

    # -------------------------------------------------------------------------
    # 3. PICK MOST RECENT ROW PER team_id
    #    i.e. group by team_id, take the first row in each group
    # -------------------------------------------------------------------------
    df_most_recent = df.groupby("team_id", as_index=False).head(1)

    # -------------------------------------------------------------------------
    # 4. WRITE TO Postgres team_features TABLE
    #    Overwrite all rows each time by TRUNCATING first.
    # -------------------------------------------------------------------------
    # a) Connect via SQLAlchemy
    engine = create_engine_from_env()

    # b) Truncate table to remove old rows
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE team_features;"))

    # c) Insert new rows
    # Make sure df_most_recent columns match the table columns in order & dtype
    # If the table has exactly the columns in selected_columns (in any order),
    # we just need to ensure df_most_recent has the same columns.
    df_most_recent.to_sql("team_features", engine, if_exists="append", index=False)

    print("✅ Wrote most recent row per team_id (with game_won not NaN) to 'team_features'.")

# ------------------------------
# Data Processing
# ------------------------------
# Process the NBA data (this function returns merged_df, team_absent_impact, team_aggregated_df data)
merged_df, team_absent_impact, team_aggregated_df = process_nba_data()

# Import list of upcoming games for next 7 days
upcoming_games = get_new_games()

# Import active player 
df_players = get_active_players()

# Import Injured Player 
df_injuries = get_injured_players()

# Import NBA games
nba_games = get_nba_games()


# **Merge impact of absent players per team per game to team_aggregated_df
team_aggregated_df = team_aggregated_df.merge(
    team_absent_impact, on=["team_id", "game_id"], how="left"
)
# Fill NaN values (if no absences, impact should be 0)
team_aggregated_df["team_absent_impact"] = team_aggregated_df["team_absent_impact"].fillna(0)

# Aggregate player-game data to Team-game data

# Display all columns
absent_counts = df_players[df_players["id"].isin(df_injuries["player_id"])].groupby("team_full_name")["id"].count()

# print(absent_counts)

# Step 1: Get list of injured player_ids
injured_player_ids = df_players[df_players["id"].isin(df_injuries["player_id"])]["id"]

# Step 2: Filter merged_df for injured players and get most recent entry per player
recent_importance = (
    merged_df[merged_df["player_id"].isin(injured_player_ids)]
    .sort_values("date", ascending=False)
    .drop_duplicates("player_id", keep="first")[["player_id", "player_importance"]]
    .set_index("player_id")["player_importance"]
)

# Step 3: Display or use the result
recent_importance


# 1. Filter most recent importance values for injured players
recent_importance_with_team = (
    merged_df[merged_df["player_id"].isin(injured_player_ids)]
    .sort_values("date", ascending=False)
    .drop_duplicates("player_id", keep="first")[["player_id", "player_importance", "team"]]
)

# 2. Group by team and sum player_importance
total_importance_by_team = recent_importance_with_team.groupby("team")["player_importance"].sum()

# 3. Display result
total_importance_by_team.sort_values(ascending=False)


# Step 1: Create copies for home and away rows
home_rows = upcoming_games.copy()
away_rows = upcoming_games.copy()

# Step 2: Build a mapping from (team_name + season) to team_id
lookup_df = team_aggregated_df[["team_name", "season", "team_id"]].drop_duplicates()
home_rows = home_rows.merge(lookup_df, left_on=["home_team", "season"], right_on=["team_name", "season"], how="left")
home_rows["team_name"] = home_rows["home_team"]
home_rows["is_home_game"] = True

away_rows = away_rows.merge(lookup_df, left_on=["away_team", "season"], right_on=["team_name", "season"], how="left")
away_rows["team_name"] = away_rows["away_team"]
away_rows["is_home_game"] = False

# Step 3: Keep only relevant columns that exist in team_aggregated_df
shared_cols = list(set(home_rows.columns) & set(team_aggregated_df.columns))
home_rows = home_rows[shared_cols]
away_rows = away_rows[shared_cols]

# Step 4: Add number of absent players for each team
home_rows["absent_players"] = home_rows["team_name"].map(absent_counts).fillna(0).astype(int)
away_rows["absent_players"] = away_rows["team_name"].map(absent_counts).fillna(0).astype(int)

# Step 5: Add impact of absent players for each team
home_rows["team_absent_impact"] = home_rows["team_name"].map(total_importance_by_team).fillna(0).astype(int)
away_rows["team_absent_impact"] = away_rows["team_name"].map(total_importance_by_team).fillna(0).astype(int)

# Step 6: Combine and append
upcoming_rows = pd.concat([home_rows, away_rows], ignore_index=True)
team_aggregated_df = pd.concat([team_aggregated_df, upcoming_rows], ignore_index=True)

# Step 7: Optional - re-sort and reset index
team_aggregated_df["date"] = pd.to_datetime(team_aggregated_df["date"])
team_aggregated_df.sort_values("date", inplace=True)
team_aggregated_df.reset_index(drop=True, inplace=True)


team_aggregated_df[team_aggregated_df["team_absent_impact"] > 0]


# Ensure sorting by team_id and date
team_aggregated_df.sort_values(by=["team_id", "date"], inplace=True)

# 1. For each team, compute difference in days between consecutive games within a season
team_aggregated_df.sort_values(by=["team_id", "date"], inplace=True)

team_aggregated_df['days_since_last_game'] = (
    team_aggregated_df.groupby(["team_id", "season"])["date"]
      .diff()
      .dt.days
)

# 2. Compute Win Streak (positive for winning streaks, negative for losing streaks)
def compute_win_streak(series):
    streak = (series.shift(1) * 2 - 1)  # Convert 1 -> 1 (win), 0 -> -1 (loss)
    return streak.groupby((streak != streak.shift()).cumsum()).cumcount() * streak
# Apply function and reset index to avoid misalignment issues
team_aggregated_df["win_streak"] = (
    team_aggregated_df.groupby("team_id")["game_won"]
    .apply(compute_win_streak)
    .reset_index(level=0, drop=True)  # Ensures correct indexing
)

# Sort by date before updating ELO:
team_aggregated_df.sort_values(by=["season", "date"], inplace=True)

# 3. Compute ELO Rating
def update_elo(elo_a, elo_b, score_a, k=20):
    """ELO rating update function"""
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    change = k * (score_a - expected_a)
    return elo_a + change, elo_b - change

# Initialize Elo ratings
initial_elo = 1500
elo_ratings_all_time = {team: initial_elo for team in team_aggregated_df["team_id"].unique()}
season_elo_ratings = {}  # Dictionary to track season-based Elo ratings

team_aggregated_df["team_elo"] = np.nan  # All-time Elo
team_aggregated_df["opponent_elo"] = np.nan
team_aggregated_df["season_elo"] = np.nan  # Elo that resets each season
team_aggregated_df["opponent_season_elo"] = np.nan  # Opponent's season Elo

# Process ELO updates
for index, row in team_aggregated_df.iterrows():
    team_id = row["team_id"]
    game_id = row["game_id"]
    season = row["season"]

    # Reset season Elo ratings at the start of each season
    if season not in season_elo_ratings:
        season_elo_ratings[season] = {team: initial_elo for team in team_aggregated_df["team_id"].unique()}

    # Find opponent
    opponent = team_aggregated_df[(team_aggregated_df["game_id"] == game_id) & 
                                  (team_aggregated_df["team_id"] != team_id)]

    if not opponent.empty:
        opponent_id = opponent.iloc[0]["team_id"]

        # Assign all-time Elo ratings
        if pd.isna(team_aggregated_df.at[index, "team_elo"]):
            team_aggregated_df.at[index, "team_elo"] = elo_ratings_all_time.get(team_id, initial_elo)
        if pd.isna(team_aggregated_df.at[index, "opponent_elo"]):
            team_aggregated_df.at[index, "opponent_elo"] = elo_ratings_all_time.get(opponent_id, initial_elo)

        # Assign season Elo ratings
        if pd.isna(team_aggregated_df.at[index, "season_elo"]):
            team_aggregated_df.at[index, "season_elo"] = season_elo_ratings[season].get(team_id, initial_elo)
        if pd.isna(team_aggregated_df.at[index, "opponent_season_elo"]):
            team_aggregated_df.at[index, "opponent_season_elo"] = season_elo_ratings[season].get(opponent_id, initial_elo)

        # Get current ELOs
        team_elo = team_aggregated_df.at[index, "team_elo"]
        opponent_elo = team_aggregated_df.at[index, "opponent_elo"]

        season_elo = team_aggregated_df.at[index, "season_elo"]
        opponent_season_elo = team_aggregated_df.at[index, "opponent_season_elo"]

        # Update Elo based on game result
        score_a = 1 if row["game_won"] else 0
        new_team_elo, new_opponent_elo = update_elo(team_elo, opponent_elo, score_a)
        new_season_team_elo, new_season_opponent_elo = update_elo(season_elo, opponent_season_elo, score_a)

        # Update dictionaries
        elo_ratings_all_time[team_id] = new_team_elo
        elo_ratings_all_time[opponent_id] = new_opponent_elo

        season_elo_ratings[season][team_id] = new_season_team_elo
        season_elo_ratings[season][opponent_id] = new_season_opponent_elo

# 4. Compute Time-in-Season (Days since the first game of the season)
season_start_dates = team_aggregated_df.groupby("season")["date"].min().to_dict()
team_aggregated_df["time_in_season"] = team_aggregated_df.apply(lambda row: (row["date"] - season_start_dates[row["season"]]).days +1, axis=1)

# Merge home_team_city from nba_games into team_aggregated_df based on game_id
team_aggregated_df = team_aggregated_df.merge(
    nba_games[["id", "home_team_city"]],  # Select relevant columns from nba_games
    left_on="game_id", 
    right_on="id", 
    how="left"
)

# Rename the merged column to "game_city"
team_aggregated_df.rename(columns={"home_team_city": "game_city"}, inplace=True)

# Drop redundant 'id' column from nba_games after merge
team_aggregated_df.drop(columns=["id"], inplace=True)


# Fix city names
city_mapping = {
    "Utah": "Salt Lake City",
    "Minnesota": "Minneapolis",
    "LA": "Los Angeles",
    "Golden State": "San Francisco",
    "Indiana": "Indianapolis"
}

# Apply the mapping
team_aggregated_df["game_city"] = team_aggregated_df["game_city"].replace(city_mapping)


# Lat & lon for NBA team cities
city_coords = {
    "Atlanta": (33.7490, -84.3880),
    "Boston": (42.3601, -71.0589),
    "Brooklyn": (40.6782, -73.9442),
    "Charlotte": (35.2271, -80.8431),
    "Chicago": (41.8781, -87.6298),
    "Cleveland": (41.4993, -81.6944),
    "Dallas": (32.7767, -96.7970),
    "Denver": (39.7392, -104.9903),
    "Detroit": (42.3314, -83.0458),
    "Houston": (29.7604, -95.3698),
    "Indianapolis": (39.7684, -86.1581),
    "Los Angeles": (34.0522, -118.2437),
    "Memphis": (35.1495, -90.0490),
    "Miami": (25.7617, -80.1918),
    "Milwaukee": (43.0389, -87.9065),
    "Minneapolis": (44.9778, -93.2650),
    "New Orleans": (29.9511, -90.0715),
    "New York": (40.7128, -74.0060),
    "Oklahoma City": (35.4676, -97.5164),
    "Orlando": (28.5383, -81.3792),
    "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740),
    "Portland": (45.5152, -122.6784),
    "Sacramento": (38.5816, -121.4944),
    "San Antonio": (29.4241, -98.4936),
    "Salt Lake City": (40.7608, -111.8910),
    "San Francisco": (37.7749, -122.4194),
    "Toronto": (43.651070, -79.347015),
    "Washington": (38.9072, -77.0369)
}

# Extract unique non-null game_city values
unique_game_cities = team_aggregated_df["game_city"].dropna().unique()

# Find cities that are not in city_coords
missing_cities = [city for city in unique_game_cities if city not in city_coords]

# # Display results
# if missing_cities:
#     print("❌ The following cities are missing from city_coords:", missing_cities)
# else:
#     print("✅ All game_city values are present in city_coords.")


# Infer missing values for game_city values:

# Create a Mapping: team_id -> home_city (from rows where is_home_game is true)
home_games = team_aggregated_df[(team_aggregated_df['is_home_game'] == 1) & (team_aggregated_df['game_city'].notna())]
team_home_city = home_games.groupby('team_id')['game_city'].first().to_dict()
# print("Team Home City Mapping (sample):", dict(list(team_home_city.items())[:5]))

# Fill Missing game_city for Home Games Using the Mapping
mask_home_missing = (team_aggregated_df['is_home_game'] == 1) & (team_aggregated_df['game_city'].isna())
team_aggregated_df.loc[mask_home_missing, 'game_city'] = team_aggregated_df.loc[mask_home_missing, 'team_id'].map(team_home_city)

# Fill Missing game_city for Remaining Rows by Grouping by game_id 
team_aggregated_df['game_city'] = team_aggregated_df.groupby('game_id')['game_city'].transform(lambda x: x.ffill().bfill())

# Report Missing Values
missing_after = team_aggregated_df['game_city'].isna().sum()
# print("Missing game_city count after filling:", missing_after)


# Ensure dataset is sorted by team_id, season, and date
team_aggregated_df.sort_values(["team_id", "season", "date"], inplace=True)

# Function to calculate travel distance using city_coords dictionary
def calc_travel_distance(row):
    # Get current and previous game city
    current_city = row["game_city"]
    prev_city = row["prev_game_city"]

    # If it's the first game of the season or game_city is missing/null, return NaN
    if pd.isna(prev_city) or pd.isna(current_city):
        return np.nan  # Return NaN instead of 0

    # Get coordinates from city_coords dictionary
    if current_city in city_coords and prev_city in city_coords:
        lat1, lon1 = city_coords[prev_city]
        lat2, lon2 = city_coords[current_city]
        return int(round(geodesic((lat1, lon1), (lat2, lon2)).miles))  # Convert to integer miles

    return np.nan  # Return NaN if cities are not found in city_coords

# Create a column for the "previous" game city per team per season
team_aggregated_df["prev_game_city"] = (
    team_aggregated_df.groupby(["team_id", "season"])["game_city"].shift(1)
)

# Apply the function to calculate travel distance
team_aggregated_df["travel_distance"] = team_aggregated_df.apply(calc_travel_distance, axis=1)

# print("✅ Travel distances calculated in miles. First game of season or missing city returns NaN!")

# Create 15-game rolling average team performance metrics

# Sort by team_id, then by date
team_aggregated_df.sort_values(by=["team_id", "date"], inplace=True)

# Define columns for rolling averages
rolling_columns = [
    "team_points",
    "point_diff",
    "pts",
    "pie",
    "field_goals_made",
    "field_goals_attempted",
    "field_goal_percentage",
    "three_pointers_made",
    "three_pointers_attempted",
    "three_point_percentage",
    "free_throws_made",
    "free_throws_attempted",
    "free_throw_percentage",
    "effective_field_goal_percentage",
    "true_shooting_percentage",
    "oreb",
    "dreb",
    "reb",
    "offensive_rebound_percentage",
    "defensive_rebound_percentage",
    "rebound_percentage",
    "ast",
    "assist_percentage",
    "assist_ratio",
    "assist_to_turnover",
    "stl",
    "blk",
    "turnover",
    "turnover_ratio",
    "pf",
    "pace",
    "net_rating",
    "offensive_rating",
    "defensive_rating",
    "usage_percentage"
]

# Use transform to ensure alignment with the original DataFrame
for col in rolling_columns:
    team_aggregated_df[f"{col}_rolling15"] = (
        team_aggregated_df.groupby("team_id")[col]
        .transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())
    )


# Inspect results
team_aggregated_df.tail(10)


team_aggregated_df.sort_values(by=["date"], inplace=True)

team_aggregated_df.tail(10)


team_aggregated_df[team_aggregated_df["team_points_rolling15"].isna()].count()


# Organizing columns for readability

# Define the columns that should be moved to the end
columns_to_move = ["absent_players", "team_absent_impact", "days_since_last_game", "game_city", "travel_distance", "win_streak", "team_elo", "opponent_elo",
    "season_elo", "opponent_season_elo", "time_in_season"]

# Get the remaining columns in their current order, excluding the ones to move
remaining_columns = [col for col in team_aggregated_df.columns if col not in columns_to_move]

# Define the new column order
ordered_columns = remaining_columns + columns_to_move

# Apply the new column order to the dataframe
team_aggregated_df = team_aggregated_df[ordered_columns]

# Display subset of data for visual validation
selected_columns = [
    "game_id", "date", "season", "team_id", "team_name", "is_home_game",
    "game_postseason", "game_won", "win_streak", "team_elo", "opponent_elo",
    "season_elo", "opponent_season_elo", "time_in_season"
]

# Filter for team_id 2 in season 2024 and select only the specified columns
filtered_df = team_aggregated_df.loc[
    (team_aggregated_df["team_id"] == 23) & (team_aggregated_df["season"] == 2024),
    selected_columns
]

# Display the filtered dataframe
filtered_df

# Create a copy of team_aggregated_df with only the selected columns
team_mini_features_df = team_aggregated_df[selected_columns].copy()

# Insert condensed list of team features into team_features table to display in streamlit app
insert_team_features(team_mini_features_df)


# Preparing Team-game dataset for ML model prediction


# Create separate home and away dataframes
home_df = team_aggregated_df[team_aggregated_df["is_home_game"] == True].copy()
away_df = team_aggregated_df[team_aggregated_df["is_home_game"] == False].copy()

# Rename columns to distinguish home and away data
home_df = home_df.add_prefix("home_")
away_df = away_df.add_prefix("away_")

# Merge the two datasets on game_id
merged_games_df = home_df.merge(
    away_df, left_on="home_game_id", right_on="away_game_id", suffixes=("", "")
)

# Drop duplicate columns for game_id
merged_games_df.drop(columns=["away_game_id"], inplace=True)
merged_games_df.rename(columns={"home_game_id": "game_id"}, inplace=True)

merged_games_df


# Rename columns and drop duplicates

# Validate that home_date and away_date are the same, then drop one and rename
if (merged_games_df["home_date"] == merged_games_df["away_date"]).all():
    merged_games_df = merged_games_df.drop(columns=["away_date"]).rename(columns={"home_date": "date"})

# Validate that home_season and away_season are the same, then drop one and rename
if (merged_games_df["home_season"] == merged_games_df["away_season"]).all():
    merged_games_df = merged_games_df.drop(columns=["away_season"]).rename(columns={"home_season": "season"})

# Drop home_is_home_game and away_is_home_game as they are redundant
merged_games_df = merged_games_df.drop(columns=["home_is_home_game", "away_is_home_game"])

# Validate that home_game_postseason and away_game_postseason are the same, then drop one and rename
if (merged_games_df["home_game_postseason"] == merged_games_df["away_game_postseason"]).all():
    merged_games_df = merged_games_df.drop(columns=["away_game_postseason"]).rename(columns={"home_game_postseason": "game_postseason"})

# print("merged_games_df missing values")
# missing_values = merged_games_df.isnull().sum()
# print(missing_values.to_string())

# Convert the 'game_postseason' column to numeric (0/1)
mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
merged_games_df["game_postseason"] = (
    merged_games_df["game_postseason"]
    .astype(str)
    .str.lower()
    .map(mapping)
    .astype(int)
)

# print(merged_games_df["game_postseason"].unique())

# Rename home_game_won and away_game_won for clarity
merged_games_df = merged_games_df.rename(columns={"home_game_won": "home_team_won", "away_game_won": "away_team_won"})

# Rename home team win streaks for clarity
merged_games_df = merged_games_df.rename(columns={"home_win_streak": "home_team_win_streak", "home_loss_streak": "home_team_loss_streak", "home_home_win_streak" : "home_team_home_games_win_streak", "home_away_win_streak" : "home_team_away_games_win_streak"})

# Rename away team win streaks for clarity
merged_games_df = merged_games_df.rename(columns={"away_win_streak": "away_team_win_streak", "away_loss_streak": "away_team_loss_streak", "away_home_win_streak" : "away_team_home_games_win_streak", "away_away_win_streak" : "away_team_away_games_win_streak"})

# Validate that home_game_city and away_game_city are the same, then drop one and rename
if (merged_games_df["home_game_city"] == merged_games_df["away_game_city"]).all():
    merged_games_df = merged_games_df.drop(columns=["away_game_city"]).rename(columns={"home_game_city": "game_city"})

# Validate that away_point_diff and home_point_diff are the same, then drop one and rename
if (merged_games_df["away_point_diff"] == merged_games_df["home_point_diff"]).all():
    merged_games_df = merged_games_df.drop(columns=["away_point_diff"]).rename(columns={"home_point_diff": "point_diff"})

# Validate that away_time_in_season and home_time_in_season are the same, then drop one and rename
if (merged_games_df["away_time_in_season"] == merged_games_df["home_time_in_season"]).all():
    merged_games_df = merged_games_df.drop(columns=["away_time_in_season"]).rename(columns={"home_time_in_season": "time_in_season"})    


# Filter rows where game_id matches upcoming games
upcoming_game_ids = upcoming_games["game_id"].unique()
upcoming_game_rows = merged_games_df[merged_games_df["game_id"].isin(upcoming_game_ids)]

# # Display result
# print(upcoming_game_rows.shape)
# upcoming_game_rows.head()


# Upcoming Games Outcome Prediction
# Features

# Define the list of feature columns
feature_cols = [
    "game_postseason",
    "time_in_season",
    
    # Home Rolling Stats & Context
    "home_team_win_streak",
    "home_team_elo",
    "home_opponent_elo",
    "home_pts_rolling15",
    "home_pie_rolling15",
    "home_field_goals_made_rolling15",
    "home_field_goals_attempted_rolling15",
    "home_field_goal_percentage_rolling15",
    "home_three_pointers_made_rolling15",
    "home_three_pointers_attempted_rolling15",
    "home_three_point_percentage_rolling15",
    "home_free_throws_made_rolling15",
    "home_free_throws_attempted_rolling15",
    "home_free_throw_percentage_rolling15",
    "home_effective_field_goal_percentage_rolling15",
    "home_true_shooting_percentage_rolling15",
    "home_oreb_rolling15",
    "home_dreb_rolling15",
    "home_reb_rolling15",
    "home_offensive_rebound_percentage_rolling15",
    "home_defensive_rebound_percentage_rolling15",
    "home_rebound_percentage_rolling15",
    "home_ast_rolling15",
    "home_assist_percentage_rolling15",
    "home_assist_ratio_rolling15",
    "home_assist_to_turnover_rolling15",
    "home_stl_rolling15",
    "home_blk_rolling15",
    "home_turnover_rolling15",
    "home_turnover_ratio_rolling15",
    "home_pf_rolling15",
    "home_pace_rolling15",
    "home_net_rating_rolling15",
    "home_offensive_rating_rolling15",
    "home_defensive_rating_rolling15",
    "home_usage_percentage_rolling15",
    "home_absent_players",
    "home_team_absent_impact",
    "home_days_since_last_game",
    "home_travel_distance",
    
    # Away Rolling Stats & Context
    "away_team_win_streak",
    "away_team_elo",
    "away_opponent_elo",
    "away_pts_rolling15",
    "away_pie_rolling15",
    "away_field_goals_made_rolling15",
    "away_field_goals_attempted_rolling15",
    "away_field_goal_percentage_rolling15",
    "away_three_pointers_made_rolling15",
    "away_three_pointers_attempted_rolling15",
    "away_three_point_percentage_rolling15",
    "away_free_throws_made_rolling15",
    "away_free_throws_attempted_rolling15",
    "away_free_throw_percentage_rolling15",
    "away_effective_field_goal_percentage_rolling15",
    "away_true_shooting_percentage_rolling15",
    "away_oreb_rolling15",
    "away_dreb_rolling15",
    "away_reb_rolling15",
    "away_offensive_rebound_percentage_rolling15",
    "away_defensive_rebound_percentage_rolling15",
    "away_rebound_percentage_rolling15",
    "away_ast_rolling15",
    "away_assist_percentage_rolling15",
    "away_assist_ratio_rolling15",
    "away_assist_to_turnover_rolling15",
    "away_stl_rolling15",
    "away_blk_rolling15",
    "away_turnover_rolling15",
    "away_turnover_ratio_rolling15",
    "away_pf_rolling15",
    "away_pace_rolling15",
    "away_net_rating_rolling15",
    "away_offensive_rating_rolling15",
    "away_defensive_rating_rolling15",
    "away_usage_percentage_rolling15",
    "away_absent_players",
    "away_team_absent_impact",
    "away_days_since_last_game",
    "away_travel_distance"
]


# Predictions


# Make predictions using the trained model
predictions = best_xgb.predict(upcoming_game_rows[feature_cols])

# Create a DataFrame with predictions and game_id for correct mapping
pred_df = pd.DataFrame({
    "game_id": upcoming_game_rows["game_id"].values,
    "predicted_home_win": predictions
})

# Merge predictions into upcoming_games on game_id to ensure correct alignment
upcoming_games = upcoming_games.merge(pred_df, on="game_id", how="left")

# Add a new column 'prediction_date' set to today's date
upcoming_games["prediction_date"] = pd.to_datetime("today").date()

# Create a new column 'predicted_winner' that displays the winning team name
upcoming_games["predicted_winner"] = np.where(
    upcoming_games["predicted_home_win"] == 1, upcoming_games["home_team"], upcoming_games["away_team"]
)

# Drop the intermediate predicted_home_win column if not needed
upcoming_games = upcoming_games.drop(columns=["predicted_home_win"])

# Display the updated DataFrame
print(upcoming_games)

# Update prediction table
upsert_game_predictions(upcoming_games)


