import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from balldontlie import BalldontlieAPI
from balldontlie.exceptions import RateLimitError, ServerError
from sqlalchemy import create_engine, text

# ------------------------------
# Load Environment Variables and Initialize API Client
# ------------------------------
load_dotenv()
API_KEY = os.getenv("BALLDONTLIE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
if not API_KEY:
    raise ValueError("API key not found. Set BALLDONTLIE_API_KEY environment variable.")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# Correct DATABASE_URL if necessary (SQLAlchemy expects "postgresql://")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=True)
api = BalldontlieAPI(api_key=API_KEY)

# ------------------------------
# Helper Functions
# ------------------------------
def extract_values(data_list):
    """Extracts and flattens nested data from API response items."""
    cleaned_data = []
    for item in data_list:
        cleaned_item = {}
        for key, value in item.__dict__.items():
            if hasattr(value, "__dict__"):
                nested_data = {f"{key}_{k}": v for k, v in value.__dict__.items()}
                cleaned_item.update(nested_data)
            else:
                cleaned_item[key] = value
        cleaned_data.append(cleaned_item)
    return cleaned_data

def fetch_all_data(endpoint_func, params=None):
    """
    Paginate through the API and return a DataFrame of results.
    Data is retrieved using the provided parameters.
    """
    if params is None:
        params = {}
    all_data = []
    cursor = None
    while True:
        try:
            if cursor:
                params["cursor"] = cursor
            params["per_page"] = 100
            response = endpoint_func(**params)
            extracted = extract_values(response.data)
            all_data.extend(extracted)
            cursor = response.meta.next_cursor
            print(f"✅ Retrieved {len(all_data)} records...")
            if not cursor:
                break
        except RateLimitError:
            print("⚠️ Rate limit reached. Waiting for 10 seconds before retrying...")
            time.sleep(10)
    return pd.DataFrame(all_data)

def fetch_single_page(endpoint_func, params={}):
    """
    Fetch a single page of API data and return as a DataFrame.
    """
    try:
        response = endpoint_func(**params)
        data_list = response.data
        if not data_list:
            print("⚠️ No data returned!")
            return pd.DataFrame()
        cleaned_data = extract_values(data_list)
        return pd.DataFrame(cleaned_data)
    except RateLimitError:
        print("⚠️ Rate limit reached. Waiting for 10 seconds before retrying...")
        time.sleep(10)
        return fetch_single_page(endpoint_func, params)
    
def extract_box_scores_data(all_box_scores):
    """
    Extracts detailed player-level statistics from the box scores data.
    
    Returns a DataFrame with one row per player per game containing the following columns:
    date, season, status, home_team_id, home_team, visitor_team_id, visitor_team,
    home_score, visitor_score, player_id, player_first_name, player_last_name, team,
    minutes, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, oreb, dreb,
    reb, ast, stl, blk, turnover, pf, pts.
    """
    rows = []
    for game in all_box_scores:
        # Extract game-level data
        game_date = game.get("date", None)
        season = game.get("season", None)
        status = game.get("status", None)
        home_team = game.get("home_team", {})
        visitor_team = game.get("visitor_team", {})
        home_team_id = home_team.get("id", None)
        home_team_name = home_team.get("full_name", None)
        visitor_team_id = visitor_team.get("id", None)
        visitor_team_name = visitor_team.get("full_name", None)
        home_score = game.get("home_team_score", None)
        visitor_score = game.get("visitor_team_score", None)
        
        # Process both home and visitor teams
        for team_key in ["home_team", "visitor_team"]:
            team = game.get(team_key, {})
            team_full_name = team.get("full_name", None)
            players = team.get("players", [])
            for player_stat in players:
                # Extract player information from the nested dictionary
                player = player_stat.get("player", {})
                player_id = player.get("id", None)
                player_first_name = player.get("first_name", None)
                player_last_name = player.get("last_name", None)
                
                # Extract statistical fields for this player
                minutes = player_stat.get("min", None)
                fgm = player_stat.get("fgm", None)
                fga = player_stat.get("fga", None)
                fg_pct = player_stat.get("fg_pct", None)
                fg3m = player_stat.get("fg3m", None)
                fg3a = player_stat.get("fg3a", None)
                fg3_pct = player_stat.get("fg3_pct", None)
                ftm = player_stat.get("ftm", None)
                fta = player_stat.get("fta", None)
                ft_pct = player_stat.get("ft_pct", None)
                oreb = player_stat.get("oreb", None)
                dreb = player_stat.get("dreb", None)
                reb = player_stat.get("reb", None)
                ast = player_stat.get("ast", None)
                stl = player_stat.get("stl", None)
                blk = player_stat.get("blk", None)
                turnover = player_stat.get("turnover", None)
                pf = player_stat.get("pf", None)
                pts = player_stat.get("pts", None)
                
                row = {
                    "date": game_date,
                    "season": season,
                    "status": status,
                    "home_team_id": home_team_id,
                    "home_team": home_team_name,
                    "visitor_team_id": visitor_team_id,
                    "visitor_team": visitor_team_name,
                    "home_score": home_score,
                    "visitor_score": visitor_score,
                    "player_id": player_id,
                    "player_first_name": player_first_name,
                    "player_last_name": player_last_name,
                    "team": team_full_name,
                    "minutes": minutes,
                    "fgm": fgm,
                    "fga": fga,
                    "fg_pct": fg_pct,
                    "fg3m": fg3m,
                    "fg3a": fg3a,
                    "fg3_pct": fg3_pct,
                    "ftm": ftm,
                    "fta": fta,
                    "ft_pct": ft_pct,
                    "oreb": oreb,
                    "dreb": dreb,
                    "reb": reb,
                    "ast": ast,
                    "stl": stl,
                    "blk": blk,
                    "turnover": turnover,
                    "pf": pf,
                    "pts": pts
                }
                rows.append(row)
    return pd.DataFrame(rows)
    
def convert_minutes(value):
    """
    Convert a time string in "MM:SS" format to total minutes as a float.
    If the value does not contain a colon, it attempts to cast it to float.
    If the conversion fails or the value is NaN, returns 0.0.
    """
    if pd.isna(value):
        return 0.0
    # Ensure the value is a string and strip any extra whitespace.
    value_str = str(value).strip()
    
    # If the value is in "MM:SS" format, convert it.
    if ":" in value_str:
        try:
            minutes_str, seconds_str = value_str.split(":")
            minutes = int(minutes_str.strip())
            seconds = int(seconds_str.strip())
            return minutes + seconds / 60.0
        except Exception as e:
            print(f"Error converting time string '{value_str}': {e}")
            return 0.0
    else:
        try:
            return float(value_str)
        except ValueError:
            return 0.0

def get_start_date(engine):
    """
    Retrieves the latest game date from both nba_game_advanced_stats and nba_box_scores.
    If the dates match, returns the next day; otherwise, returns the day after the later date.
    
    Parameters:
        engine : SQLAlchemy engine instance
        
    Returns:
        start_date (Timestamp): The date from which new data should be fetched.
    """

    # Query the last row from nba_game_advanced_stats ordered by game_date descending
    query_adv = text("SELECT * FROM nba_game_advanced_stats ORDER BY game_date DESC LIMIT 1")
    df_adv = pd.read_sql(query_adv, engine)
    # Convert the date to datetime and extract the value
    last_date_adv = pd.to_datetime(df_adv.iloc[0]["game_date"])

    # Query the last row from nba_box_scores ordered by date descending
    query_box = text("SELECT * FROM nba_box_scores ORDER BY date DESC LIMIT 1")
    df_box = pd.read_sql(query_box, engine)
    last_date_box = pd.to_datetime(df_box.iloc[0]["date"])

    # Compare dates and set start_date accordingly
    if last_date_adv == last_date_box:
        print("Dates in nba_game_advanced_stats & nba_box_scores match")
        start_date = last_date_box + pd.Timedelta(days=1)
    else:
        print("Dates do not match! Using the later date plus one day.")
        start_date = max(last_date_adv, last_date_box) + pd.Timedelta(days=1)
        
    return start_date.date()

def get_latest_season(engine):
    """
    Retrieves the most recent season value from the nba_box_scores table.
    
    Parameters:
        engine : SQLAlchemy engine instance
        
    Returns:
        latest_season (int): The most recent season found in the nba_box_scores table.
    """
    # Query the maximum season value from nba_box_scores
    query = text("SELECT MAX(season) AS max_season FROM nba_box_scores")
    df = pd.read_sql(query, engine)
    latest_season = int(df.iloc[0]["max_season"])
    return latest_season

def get_date_range(start_date_str, end_date_str):
    """
    Generate a list of dates (as strings) in YYYY-MM-DD format from start_date to end_date (inclusive).
    """
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    delta = end - start
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]          

# ------------------------------
# Date Range for Season-Dependent Data
# ------------------------------
start_date = get_start_date(engine).strftime("%Y-%m-%d")
yesterday_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

print(start_date,yesterday_date)

# ------------------------------
# Fetch Date-Dependent Data (Advanced Stats and Box Scores)
# ------------------------------

# Generate a date range from start_date to today
date_list = get_date_range(start_date, yesterday_date)

# 1. Fetch Advanced Stats and save to table nba_game_advanced_stats (append)
params_adv_stats = {"dates":date_list}
df_adv_stats = fetch_all_data(api.nba.advanced_stats.list, params_adv_stats)
# Convert any empty strings (or strings with only whitespace) to None
df_adv_stats = df_adv_stats.replace(r'^\s*$', None, regex=True)
if not df_adv_stats.empty:
    df_adv_stats.to_sql("nba_game_advanced_stats", engine, if_exists="append", index=False)
    print("Advanced stats data saved to table nba_game_advanced_stats.")
else:
    print("No advanced stats data retrieved.")

# 2. Fetch box scores for each date in the range
all_box_scores = []
for single_date in date_list:
    try:
        print(f"📅 Fetching box scores for {single_date}...")
        api_response = api.nba.box_scores.get_by_date(date=single_date)
        data_json = api_response.model_dump_json()
        data_dict = json.loads(data_json)
        games_data = data_dict.get("data", [])
        all_box_scores.extend(games_data)
    except ServerError as e:
        print(f"⚠️ Error fetching box scores for {single_date}: {e}. Skipping this date...")
        time.sleep(2)

# Extract detailed player-level box score data
df_box_scores = extract_box_scores_data(all_box_scores)

# Replace empty strings with None
df_box_scores = df_box_scores.replace(r'^\s*$', None, regex=True)

# Convert the "minutes" column, if present, using convert_minutes
if "minutes" in df_box_scores.columns:
    df_box_scores["minutes"] = df_box_scores["minutes"].apply(convert_minutes)

# Optional: print the DataFrame information to verify
print("Extracted Box Scores DataFrame Info:")
df_box_scores.info()

# Append the detailed box scores data into the PostgreSQL table "nba_box_scores"
if not df_box_scores.empty:
    df_box_scores.to_sql("nba_box_scores", engine, if_exists="append", index=False)
    print("Box scores data saved to table nba_box_scores.")
else:
    print("No box score data retrieved.") 

# ------------------------------
# Fetch Season-Dependent Data
# ------------------------------

season = get_latest_season(engine)
print(season)
# 3. Fetch All NBA Games
df_games = fetch_all_data(api.nba.games.list, {"seasons": [season]})
if not df_games.empty:
    df_games.to_sql("nba_games", engine, if_exists="replace", index=False)
    print("Game data saved to table nba_games.")
else:
    print("No game data retrieved.")

# ------------------------------
# Fetch Date-Independent Data
# ------------------------------

# 4. Fetch Active Players (overwrite table nba_active_players)
df_active_players = fetch_all_data(api.nba.players.list_active)
if not df_active_players.empty:
    df_active_players.to_sql("nba_active_players", engine, if_exists="replace", index=False)
    print("Active players data saved to table nba_active_players.")
else:
    print("No active players data retrieved.")  

# 5. Fetch Injuries Data (overwrite table nba_player_injuries)
df_injuries = fetch_all_data(api.nba.injuries.list)
if not df_injuries.empty:
    df_injuries.to_sql("nba_player_injuries", engine, if_exists="replace", index=False)
    print("Injuries data saved to table nba_player_injuries.")
else:
    print("No injury data retrieved.")

print("✅ All data successfully retrieved and saved to PostgreSQL!")