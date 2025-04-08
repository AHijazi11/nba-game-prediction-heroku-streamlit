import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# Ensure the URL uses "postgresql://" instead of "postgres://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=False)

# Query to get the last 5 rows from nba_game_advanced_stats (ordered by id descending)
query_adv = text("SELECT * FROM nba_game_advanced_stats ORDER BY game_date DESC LIMIT 5")
df_adv = pd.read_sql(query_adv, engine)

# Query to get the last 5 rows from nba_box_scores (ordered by id descending)
query_box = text("SELECT * FROM nba_box_scores ORDER BY date DESC LIMIT 5")
df_box = pd.read_sql(query_box, engine)

print("Last 5 rows from nba_game_advanced_stats:")
print(df_adv['game_date'])

print("\nLast 5 rows from nba_box_scores:")
print(df_box["date"])