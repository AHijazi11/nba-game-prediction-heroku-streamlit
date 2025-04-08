import os
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, text
from sqlalchemy.types import Integer, Float, String, DateTime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# Fix URL if needed
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create the SQLAlchemy engine and metadata object
engine = create_engine(DATABASE_URL, echo=True)
metadata = MetaData()

# Function to map pandas dtypes to SQLAlchemy types
def map_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return Integer
    elif pd.api.types.is_float_dtype(dtype):
        return Float
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return DateTime
    else:
        return String(255)

# Path to a sample CSV file for nba_game_advanced_stats (adjust as needed)
sample_csv = "./nba_games_2024.csv"
if not os.path.exists(sample_csv):
    raise FileNotFoundError(f"Sample file {sample_csv} not found. Please ensure it exists.")

# Read the CSV file to infer the schema
df_sample = pd.read_csv(sample_csv)

# Create a list of SQLAlchemy Column objects based on the DataFrame columns and dtypes
columns = [Column(col, map_dtype(dtype)) for col, dtype in df_sample.dtypes.items()]

# Define the table name
table_name = "nba_games"

# Create the table object with the inferred columns
nba_game_advanced_stats_table = Table(table_name, metadata, *columns)

# Create the table in PostgreSQL (if it doesn't exist)
metadata.create_all(engine)

print(f"Table '{table_name}' created with columns:")
print(list(df_sample.columns))


# Create model performance tracking table
# SQL query to create the model_performance table
create_table_query = text("""
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    date_trained DATE NOT NULL,
    best_params TEXT,
    best_cv_accuracy NUMERIC,
    validation_accuracy NUMERIC,
    validation_precision NUMERIC,
    validation_recall NUMERIC,
    validation_f1 NUMERIC,
    validation_auc NUMERIC,
    test_accuracy NUMERIC,
    test_precision NUMERIC,
    test_recall NUMERIC,
    test_f1 NUMERIC,
    test_auc NUMERIC,
    model_filename VARCHAR(255)
);
""")

# Execute the query to create the table
with engine.connect() as conn:
    conn.execute(create_table_query)
    conn.commit()

print("✅ Model performance table created successfully.")