import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# Ensure the URL uses "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=True)

# SQL query to drop the existing nba_predictions table (if it exists)
drop_table_query = text("DROP TABLE IF EXISTS nba_predictions;")

# SQL query to create the new nba_predictions table with the desired columns and data types
create_table_query = text("""
CREATE TABLE nba_predictions (
    game_id BIGINT PRIMARY KEY,
    date DATE NOT NULL,
    season INTEGER NOT NULL,
    home_team VARCHAR(255) NOT NULL,
    away_team VARCHAR(255) NOT NULL,
    game_city VARCHAR(255),
    game_postseason BOOLEAN,
    predicted_winner VARCHAR(255),
    prediction_date DATE
);
""")

# Execute the drop and create queries within a transaction
with engine.begin() as conn:
    conn.execute(drop_table_query)
    conn.execute(create_table_query)

print("Table 'nba_predictions' created successfully.")