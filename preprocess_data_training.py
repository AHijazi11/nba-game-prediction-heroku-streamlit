def process_nba_data():
    import os
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    from geopy.distance import geodesic

    # ------------------------------------
    # Load environment variables & setup DB
    # ------------------------------------
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable not set.")

    # Ensure the DATABASE_URL is in the correct format for SQLAlchemy
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    # Create the SQLAlchemy engine
    engine = create_engine(DATABASE_URL, echo=False)

    # ------------------------------------
    # 1) Load Advanced Stats
    # ------------------------------------
    advanced_stats = pd.read_sql("SELECT * FROM nba_game_advanced_stats", engine)

    # Exclude 'game_time' column if it exists due to inconsistent data types
    if "game_time" in advanced_stats.columns:
        advanced_stats = advanced_stats.drop(columns=["game_time"])

    # Drop unneeded columns
    columns_to_drop = [
        "player_position", "player_height", "player_weight", "player_jersey_number",
        "player_team", "game_home_team", "game_visitor_team", "player_college", "player_country",
        "player_draft_year", "player_draft_round", "player_draft_number"
    ]
    advanced_stats = advanced_stats.drop(columns=columns_to_drop)

    # Change data types as needed
    advanced_stats["player_team_id"] = advanced_stats["player_team_id"].astype(int)
    advanced_stats["game_home_team_id"] = advanced_stats["game_home_team_id"].astype(int)
    advanced_stats["game_visitor_team_id"] = advanced_stats["game_visitor_team_id"].astype(int)
    advanced_stats["game_date"] = pd.to_datetime(advanced_stats["game_date"])
    advanced_stats["game_period"] = advanced_stats["game_period"].astype(int)

    # ------------------------------------
    # 2) Load Box Score Data
    # ------------------------------------
    box_scores_combined = pd.read_sql("SELECT * FROM nba_box_scores", engine)

    # Convert "MM:SS" to total minutes as float
    def convert_minutes(value):
        if pd.isna(value):
            return 0.0
        if isinstance(value, str) and ":" in value:
            minutes, seconds = map(int, value.split(":"))
            return minutes + (seconds / 60)
        try:
            return float(value)
        except ValueError:
            return 0.0

    box_scores_combined["minutes"] = box_scores_combined["minutes"].apply(convert_minutes)

    # Additional type conversions
    box_scores_combined["date"] = pd.to_datetime(box_scores_combined["date"])
    box_scores_combined["season"] = box_scores_combined["season"].astype(int)
    box_scores_combined["home_score"] = box_scores_combined["home_score"].astype(int)
    box_scores_combined["visitor_score"] = box_scores_combined["visitor_score"].astype(int)

    # ------------------------------------
    # 3) Merge Box Scores with Advanced Stats
    # ------------------------------------
    merged_df = box_scores_combined.merge(
        advanced_stats,
        how='left',
        left_on=['player_id', 'date', 'home_team_id', 'visitor_team_id'],
        right_on=['player_id', 'game_date', 'game_home_team_id', 'game_visitor_team_id']
    )
    # Drop columns from the advanced_stats side that are duplicated
    merged_df = merged_df.drop(columns=['game_date', 'game_home_team_id', 'game_visitor_team_id'])

    # Drop redundant columns
    columns_to_drop = [
        'player_first_name_y', 'player_last_name_y', 'player_team_id',
        'game_season', 'game_status', 'game_home_team_score', 'game_visitor_team_score',
        'id'
    ]
    merged_df = merged_df.drop(columns=columns_to_drop)

    # Rename for clarity
    merged_df = merged_df.rename(columns={
        'player_first_name_x': 'player_first_name',
        'player_last_name_x': 'player_last_name'
    })

    # Reorder columns
    column_order = [
        'date', 'season', 'status',
        'home_team_id', 'home_team', 'visitor_team_id', 'visitor_team', 'home_score', 'visitor_score',
        'player_id', 'player_first_name', 'player_last_name', 'team', 'minutes',
        'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct',
        'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'turnover', 'pf', 'pts',
        'pie', 'pace', 'assist_percentage', 'assist_ratio', 'assist_to_turnover',
        'defensive_rating', 'defensive_rebound_percentage', 'effective_field_goal_percentage',
        'net_rating', 'offensive_rating', 'offensive_rebound_percentage', 'rebound_percentage',
        'true_shooting_percentage', 'turnover_ratio', 'usage_percentage',
        'player_college', 'player_country', 'team_id',
        'game_id', 'game_period', 'game_time', 'game_postseason'
    ]
    merged_df = merged_df[[col for col in column_order if col in merged_df.columns]]

    # Replace null FG%, 3P%, FT% with 0.0 if attempts == 0
    merged_df[['fg_pct', 'fg3_pct', 'ft_pct']] = merged_df[['fg_pct', 'fg3_pct', 'ft_pct']].fillna(0.0)

    # Rename shooting columns
    columns_to_rename = {
        'fgm': 'field_goals_made',
        'fga': 'field_goals_attempted',
        'fg_pct': 'field_goal_percentage',
        'fg3m': 'three_pointers_made',
        'fg3a': 'three_pointers_attempted',
        'fg3_pct': 'three_point_percentage',
        'ftm': 'free_throws_made',
        'fta': 'free_throws_attempted',
        'ft_pct': 'free_throw_percentage',
    }
    merged_df.rename(columns=columns_to_rename, inplace=True)

    # ------------------------------------
    # 4) Marking Unexpected Absences
    # ------------------------------------
    player_avg_minutes_per_season = merged_df.groupby(["season", "player_id"])["minutes"].mean().reset_index()
    player_avg_minutes = player_avg_minutes_per_season.set_index(["season", "player_id"])["minutes"].to_dict()

    def was_absent(row):
        avg_minutes = player_avg_minutes.get((row["season"], row["player_id"]), 0)
        return True if avg_minutes > 5 and row["minutes"] == 0 else False

    merged_df["is_absent"] = merged_df.apply(was_absent, axis=1)

    # Drop rows where 'is_absent' is False AND 'minutes' is 0
    merged_df = merged_df[~((merged_df["is_absent"] == False) & (merged_df["minutes"] == 0))]
    merged_df.reset_index(drop=True, inplace=True)

    # ------------------------------------
    # 5) Calculate Player Importance & Team Absent Impact
    # ------------------------------------
    # Rolling columns
    rolling_columns = ["minutes", "pie", "usage_percentage", "net_rating", "pts"]

    merged_df.sort_values(by=["player_id", "date"], inplace=True)
    for col in rolling_columns:
        merged_df[f"{col}_rolling15"] = (
            merged_df.groupby("player_id")[col]
            .transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())
        )

    merged_df["player_importance"] = (
        merged_df["minutes_rolling15"] * 0.4 +
        merged_df["pie_rolling15"] * 0.2 +
        merged_df["usage_percentage_rolling15"] * 0.15 +
        merged_df["net_rating_rolling15"] * 0.15 +
        merged_df["pts_rolling15"] * 0.1
    )

    team_absent_impact = (
        merged_df[merged_df["is_absent"] == 1]
        .groupby(["team_id", "game_id"])["player_importance"]
        .sum()
        .reset_index()
        .rename(columns={"player_importance": "team_absent_impact"})
    )

    # Infer missing game_id, game_period, and game_postseason
    merged_df["game_id"] = merged_df.groupby(["date", "home_team_id", "visitor_team_id"])["game_id"]\
        .transform(lambda x: x.ffill().bfill())
    merged_df["game_period"] = merged_df.groupby("game_id")["game_period"]\
        .transform(lambda x: x.ffill().bfill())
    merged_df["game_postseason"] = merged_df.groupby("game_id")["game_postseason"]\
        .transform(lambda x: x.ffill().bfill())

    # If team_id is NaN, infer from home/visitor
    merged_df["team_id"] = merged_df.apply(
        lambda row: row["home_team_id"] if pd.isna(row["team_id"]) and row["team"] == row["home_team"]
        else (row["visitor_team_id"] if pd.isna(row["team_id"]) and row["team"] == row["visitor_team"] else row["team_id"]),
        axis=1
    )

    # Performance metrics set to 0 for players with 0 minutes
    performance_cols = [
        "pie", "pace", "assist_percentage", "assist_ratio", "assist_to_turnover",
        "defensive_rating", "defensive_rebound_percentage", "effective_field_goal_percentage",
        "net_rating", "offensive_rating", "offensive_rebound_percentage", "rebound_percentage",
        "true_shooting_percentage", "turnover_ratio", "usage_percentage"
    ]
    merged_df.loc[merged_df["minutes"] == 0, performance_cols] = merged_df.loc[
        merged_df["minutes"] == 0, performance_cols
    ].fillna(0)

    # ------------------------------------
    # 6) Aggregate Player-Game Data -> Team-Game Data
    # ------------------------------------
    team_aggregated_df = merged_df.groupby(["game_id", "team_id"]).agg({
        "minutes": "sum",
        "field_goals_made": "sum",
        "field_goals_attempted": "sum",
        "three_pointers_made": "sum",
        "three_pointers_attempted": "sum",
        "free_throws_made": "sum",
        "free_throws_attempted": "sum",
        "oreb": "sum",
        "dreb": "sum",
        "reb": "sum",
        "ast": "sum",
        "stl": "sum",
        "blk": "sum",
        "turnover": "sum",
        "pf": "sum",
        "pts": "sum",
        "pie": "sum",
        "is_absent": "sum",
        "field_goal_percentage": lambda x: (x * merged_df.loc[x.index, "minutes"]).sum() / merged_df.loc[x.index, "minutes"].sum(),
        "three_point_percentage": lambda x: (x * merged_df.loc[x.index, "minutes"]).sum() / merged_df.loc[x.index, "minutes"].sum(),
        "free_throw_percentage": lambda x: (x * merged_df.loc[x.index, "minutes"]).sum() / merged_df.loc[x.index, "minutes"].sum(),
        "effective_field_goal_percentage": lambda x: (x * merged_df.loc[x.index, "minutes"]).sum() / merged_df.loc[x.index, "minutes"].sum(),
        "true_shooting_percentage": lambda x: (x * merged_df.loc[x.index, "minutes"]).sum() / merged_df.loc[x.index, "minutes"].sum(),
        "pace": "mean",
        "assist_percentage": "mean",
        "assist_ratio": "mean",
        "assist_to_turnover": "mean",
        "defensive_rating": "mean",
        "defensive_rebound_percentage": "mean",
        "net_rating": "mean",
        "offensive_rating": "mean",
        "offensive_rebound_percentage": "mean",
        "rebound_percentage": "mean",
        "turnover_ratio": "mean",
        "usage_percentage": "mean",
    }).reset_index()

    # Merge in date, season, home/visitor data, final scores
    team_aggregated_df = team_aggregated_df.merge(
        merged_df[["game_id", "date"]].drop_duplicates(),
        on="game_id", how="left"
    )
    team_aggregated_df = team_aggregated_df.merge(
        merged_df[["game_id", "home_team_id", "season", "visitor_team_id", "home_team",
                   "visitor_team", "home_score", "visitor_score", "game_postseason"]]
        .drop_duplicates(),
        on="game_id", how="left"
    )

    # is_home_game
    team_aggregated_df["is_home_game"] = team_aggregated_df["team_id"] == team_aggregated_df["home_team_id"]

    # team_name
    team_aggregated_df["team_name"] = team_aggregated_df.apply(
        lambda row: row["home_team"] if row["is_home_game"] else row["visitor_team"], axis=1
    )

    # team_points
    team_aggregated_df["team_points"] = team_aggregated_df.apply(
        lambda row: row["home_score"] if row["is_home_game"] else row["visitor_score"], axis=1
    )

    # game_won
    team_aggregated_df["game_won"] = team_aggregated_df.apply(
        lambda row: (row["home_score"] > row["visitor_score"]) if row["is_home_game"]
        else (row["visitor_score"] > row["home_score"]),
        axis=1
    )

    # point_diff
    team_aggregated_df["point_diff"] = team_aggregated_df.apply(
        lambda row: abs(row["home_score"] - row["visitor_score"]), axis=1
    )

    # Rename 'is_absent' -> 'absent_players'
    team_aggregated_df.rename(columns={"is_absent": "absent_players"}, inplace=True)

    # Drop unneeded columns
    team_aggregated_df.drop(columns=["home_team_id", "visitor_team_id", "home_team", "visitor_team", "home_score", "visitor_score"], inplace=True)

    # Reorder columns
    column_order = [
        "game_id", "date", "season", "team_id", "team_name", "is_home_game", "game_postseason", "game_won", "team_points", "point_diff",
        "minutes", "pts", "pie",
        "field_goals_made", "field_goals_attempted", "field_goal_percentage",
        "three_pointers_made", "three_pointers_attempted", "three_point_percentage",
        "free_throws_made", "free_throws_attempted", "free_throw_percentage",
        "effective_field_goal_percentage", "true_shooting_percentage",
        "oreb", "dreb", "reb", "offensive_rebound_percentage", "defensive_rebound_percentage", "rebound_percentage",
        "ast", "assist_percentage", "assist_ratio", "assist_to_turnover",
        "stl", "blk", "turnover", "turnover_ratio", "pf",
        "pace", "net_rating", "offensive_rating", "defensive_rating", "usage_percentage", "absent_players"
    ]
    team_aggregated_df["team_id"] = team_aggregated_df["team_id"].astype(int)
    team_aggregated_df["date"] = pd.to_datetime(team_aggregated_df["date"])
    team_aggregated_df["team_points"] = team_aggregated_df["team_points"].astype(int)
    team_aggregated_df["point_diff"] = team_aggregated_df["point_diff"].astype(int)
    team_aggregated_df["game_postseason"] = team_aggregated_df["game_postseason"].astype(str).str.lower().map({"true": 1, "false": 0})
    team_aggregated_df["game_won"] = team_aggregated_df["game_won"].astype(int)
    team_aggregated_df = team_aggregated_df[column_order]

    # ------------------------------------
    # 7) Feature Engineering
    # ------------------------------------
    team_aggregated_df.sort_values(by=["team_id", "date"], inplace=True)

    # 7.1 days_since_last_game (per season)
    team_aggregated_df["days_since_last_game"] = (
        team_aggregated_df.groupby(["team_id", "season"])["date"]
        .diff()
        .dt.days
    )

    # 7.2 Win streak
    def compute_win_streak(series):
        streak = (series.shift(1) * 2 - 1)  # convert 1->1 (win), 0->-1 (loss)
        return streak.groupby((streak != streak.shift()).cumsum()).cumcount() * streak

    team_aggregated_df["win_streak"] = (
        team_aggregated_df.groupby("team_id")["game_won"]
        .apply(compute_win_streak)
        .reset_index(level=0, drop=True)
    )

    # 7.3 Elo Ratings
    def update_elo(elo_a, elo_b, score_a, k=20):
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        change = k * (score_a - expected_a)
        return elo_a + change, elo_b - change

    initial_elo = 1500
    elo_ratings_all_time = {team: initial_elo for team in team_aggregated_df["team_id"].unique()}
    season_elo_ratings = {}

    team_aggregated_df["team_elo"] = np.nan
    team_aggregated_df["opponent_elo"] = np.nan
    team_aggregated_df["season_elo"] = np.nan
    team_aggregated_df["opponent_season_elo"] = np.nan

    team_aggregated_df.sort_values(by=["season", "date"], inplace=True)

    for index, row in team_aggregated_df.iterrows():
        team_id = row["team_id"]
        game_id = row["game_id"]
        season = row["season"]

        # Reset Elo at start of each season
        if season not in season_elo_ratings:
            season_elo_ratings[season] = {t: initial_elo for t in team_aggregated_df["team_id"].unique()}

        opponent = team_aggregated_df[
            (team_aggregated_df["game_id"] == game_id) & 
            (team_aggregated_df["team_id"] != team_id)
        ]

        if not opponent.empty:
            opponent_id = opponent.iloc[0]["team_id"]

            if pd.isna(team_aggregated_df.at[index, "team_elo"]):
                team_aggregated_df.at[index, "team_elo"] = elo_ratings_all_time.get(team_id, initial_elo)
            if pd.isna(team_aggregated_df.at[index, "opponent_elo"]):
                team_aggregated_df.at[index, "opponent_elo"] = elo_ratings_all_time.get(opponent_id, initial_elo)

            if pd.isna(team_aggregated_df.at[index, "season_elo"]):
                team_aggregated_df.at[index, "season_elo"] = season_elo_ratings[season].get(team_id, initial_elo)
            if pd.isna(team_aggregated_df.at[index, "opponent_season_elo"]):
                team_aggregated_df.at[index, "opponent_season_elo"] = season_elo_ratings[season].get(opponent_id, initial_elo)

            team_elo = team_aggregated_df.at[index, "team_elo"]
            opponent_elo = team_aggregated_df.at[index, "opponent_elo"]

            season_elo = team_aggregated_df.at[index, "season_elo"]
            opponent_season_elo = team_aggregated_df.at[index, "opponent_season_elo"]

            score_a = 1 if row["game_won"] else 0
            new_team_elo, new_opponent_elo = update_elo(team_elo, opponent_elo, score_a)
            new_season_team_elo, new_season_opponent_elo = update_elo(season_elo, opponent_season_elo, score_a)

            elo_ratings_all_time[team_id] = new_team_elo
            elo_ratings_all_time[opponent_id] = new_opponent_elo

            season_elo_ratings[season][team_id] = new_season_team_elo
            season_elo_ratings[season][opponent_id] = new_season_opponent_elo

    # 7.4 Time-in-Season
    season_start_dates = team_aggregated_df.groupby("season")["date"].min().to_dict()
    team_aggregated_df["time_in_season"] = team_aggregated_df.apply(
        lambda row: (row["date"] - season_start_dates[row["season"]]).days + 1, axis=1
    )

    # 7.5 Merge in City Info
    nba_games = pd.read_sql("SELECT * FROM nba_games", engine).drop_duplicates()
    team_aggregated_df = team_aggregated_df.merge(
        nba_games[["id", "home_team_city"]],
        left_on="game_id",
        right_on="id",
        how="left"
    )
    team_aggregated_df.rename(columns={"home_team_city": "game_city"}, inplace=True)
    team_aggregated_df.drop(columns=["id"], inplace=True)

    # Fix city names
    city_mapping = {
        "Utah": "Salt Lake City",
        "Minnesota": "Minneapolis",
        "LA": "Los Angeles",
        "Golden State": "San Francisco",
        "Indiana": "Indianapolis"
    }
    team_aggregated_df["game_city"] = team_aggregated_df["game_city"].replace(city_mapping)

    # Known city coords
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

    # Infer missing cities for home teams if needed
    home_games = team_aggregated_df[
        (team_aggregated_df['is_home_game'] == 1) & (team_aggregated_df['game_city'].notna())
    ]
    team_home_city = home_games.groupby('team_id')['game_city'].first().to_dict()

    mask_home_missing = (team_aggregated_df['is_home_game'] == 1) & (team_aggregated_df['game_city'].isna())
    team_aggregated_df.loc[mask_home_missing, 'game_city'] = (
        team_aggregated_df.loc[mask_home_missing, 'team_id'].map(team_home_city)
    )

    # Fill remaining missing city values by ffill/bfill within the same game_id
    team_aggregated_df['game_city'] = (
        team_aggregated_df.groupby('game_id')['game_city']
        .transform(lambda x: x.ffill().bfill())
    )

    # 7.6 Travel Distance
    team_aggregated_df["prev_game_city"] = (
        team_aggregated_df.groupby(["team_id", "season"])["game_city"].shift(1)
    )

    def calc_travel_distance(row):
        current_city = row["game_city"]
        prev_city = row["prev_game_city"]
        if pd.isna(prev_city) or pd.isna(current_city):
            return np.nan
        if current_city in city_coords and prev_city in city_coords:
            lat1, lon1 = city_coords[prev_city]
            lat2, lon2 = city_coords[current_city]
            return int(round(geodesic((lat1, lon1), (lat2, lon2)).miles))
        return np.nan

    team_aggregated_df["travel_distance"] = team_aggregated_df.apply(calc_travel_distance, axis=1)

    # 7.7 Merge team_absent_impact
    team_aggregated_df = team_aggregated_df.merge(
        team_absent_impact, on=["team_id", "game_id"], how="left"
    )
    team_aggregated_df["team_absent_impact"] = team_aggregated_df["team_absent_impact"].fillna(0)

    # 7.8 Rolling 15-Game Averages
    team_aggregated_df.sort_values(by=["team_id", "date"], inplace=True)
    rolling_columns = [
        "team_points", "point_diff", "pts", "pie", "field_goals_made", "field_goals_attempted",
        "field_goal_percentage", "three_pointers_made", "three_pointers_attempted",
        "three_point_percentage", "free_throws_made", "free_throws_attempted",
        "free_throw_percentage", "effective_field_goal_percentage", "true_shooting_percentage",
        "oreb", "dreb", "reb", "offensive_rebound_percentage", "defensive_rebound_percentage",
        "rebound_percentage", "ast", "assist_percentage", "assist_ratio", "assist_to_turnover",
        "stl", "blk", "turnover", "turnover_ratio", "pf", "pace", "net_rating",
        "offensive_rating", "defensive_rating", "usage_percentage"
    ]
    for col in rolling_columns:
        team_aggregated_df[f"{col}_rolling15"] = (
            team_aggregated_df.groupby("team_id")[col]
            .transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())
        )

    # Reorganize columns
    columns_to_move = [
        "absent_players", "team_absent_impact", "days_since_last_game", "game_city",
        "travel_distance", "win_streak", "team_elo", "opponent_elo", "season_elo",
        "opponent_season_elo", "time_in_season"
    ]
    remaining_columns = [c for c in team_aggregated_df.columns if c not in columns_to_move]
    ordered_columns = remaining_columns + columns_to_move
    team_aggregated_df = team_aggregated_df[ordered_columns]

    # ------------------------------------
    # 8) Combine Home/Away Team Rows -> One Row per Game
    # ------------------------------------
    home_df = team_aggregated_df[team_aggregated_df["is_home_game"] == True].copy()
    away_df = team_aggregated_df[team_aggregated_df["is_home_game"] == False].copy()

    home_df = home_df.add_prefix("home_")
    away_df = away_df.add_prefix("away_")

    merged_games_df = home_df.merge(
        away_df,
        left_on="home_game_id",
        right_on="away_game_id",
        suffixes=("", "")
    )

    merged_games_df.drop(columns=["away_game_id"], inplace=True)
    merged_games_df.rename(columns={"home_game_id": "game_id"}, inplace=True)

    # Validate and rename
    if (merged_games_df["home_date"] == merged_games_df["away_date"]).all():
        merged_games_df.drop(columns=["away_date"], inplace=True)
        merged_games_df.rename(columns={"home_date": "date"}, inplace=True)

    if (merged_games_df["home_season"] == merged_games_df["away_season"]).all():
        merged_games_df.drop(columns=["away_season"], inplace=True)
        merged_games_df.rename(columns={"home_season": "season"}, inplace=True)

    if (merged_games_df["home_game_postseason"] == merged_games_df["away_game_postseason"]).all():
        merged_games_df.drop(columns=["away_game_postseason"], inplace=True)
        merged_games_df.rename(columns={"home_game_postseason": "game_postseason"}, inplace=True)

    merged_games_df.rename(
        columns={
            "home_game_won": "home_team_won",
            "away_game_won": "away_team_won"
        },
        inplace=True
    )
    
    # Rename home team win streaks for clarity
    merged_games_df = merged_games_df.rename(columns={"home_win_streak": "home_team_win_streak"})
    # Rename away team win streaks for clarity
    merged_games_df = merged_games_df.rename(columns={"away_win_streak": "away_team_win_streak"})
    
    # If home_game_city == away_game_city, drop one
    if (merged_games_df["home_game_city"] == merged_games_df["away_game_city"]).all():
        merged_games_df.drop(columns=["away_game_city"], inplace=True)
        merged_games_df.rename(columns={"home_game_city": "game_city"}, inplace=True)

    # If away_point_diff == home_point_diff, drop one
    if "home_point_diff" in merged_games_df.columns and "away_point_diff" in merged_games_df.columns:
        if (merged_games_df["away_point_diff"] == merged_games_df["home_point_diff"]).all():
            merged_games_df.drop(columns=["away_point_diff"], inplace=True)
            merged_games_df.rename(columns={"home_point_diff": "point_diff"}, inplace=True)

    # If away_time_in_season == home_time_in_season, drop one
    if "away_time_in_season" in merged_games_df.columns and "home_time_in_season" in merged_games_df.columns:
        if (merged_games_df["away_time_in_season"] == merged_games_df["home_time_in_season"]).all():
            merged_games_df.drop(columns=["away_time_in_season"], inplace=True)
            merged_games_df.rename(columns={"home_time_in_season": "time_in_season"}, inplace=True)

    # Return the final merged dataframe
    return merged_games_df