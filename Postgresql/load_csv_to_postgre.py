import os
import glob
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# SQLAlchemy expects "postgresql://" instead of "postgres://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create the SQLAlchemy engine for Heroku Postgres
engine = create_engine(DATABASE_URL, echo=True)

# Directory where CSV files are stored
data_dir = "./"

# Define table mapping and file patterns
table_files = {
    "nba_game_advanced_stats": os.path.join(data_dir, "nba_game_advanced_stats_*.csv"),
    "nba_games": os.path.join(data_dir, "nba_games_*.csv")
}

for table_name, pattern in table_files.items():
    file_list = glob.glob(pattern)
    if not file_list:
        print(f"No files found for table {table_name} with pattern {pattern}")
        continue

    print(f"\nLoading data into table '{table_name}' from {len(file_list)} file(s)...")
    for file in file_list:
        print(f"Processing file: {file}")
        try:
            df = pd.read_csv(file)
            # Append data to the table
            df.to_sql(table_name, engine, if_exists="append", index=False)
            print(f"Successfully loaded {len(df)} records from {file} into {table_name}.")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

print("\n✅ All CSV files have been processed and loaded into PostgreSQL!")