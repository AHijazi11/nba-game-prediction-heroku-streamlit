import os
import glob
import time
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ------------------------------
# Load environment variables and initialize engine
# ------------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# SQLAlchemy expects "postgresql://" instead of "postgres://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=True)

# ------------------------------
# Function to convert "MM:SS" to total minutes as a float
# ------------------------------
def convert_minutes(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, str) and ":" in value:
        try:
            minutes, seconds = map(int, value.split(":"))
            return minutes + (seconds / 60)
        except Exception:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0

# ------------------------------
# Step 1: Delete all rows from nba_box_scores table
# ------------------------------
with engine.connect() as connection:
    trans = connection.begin()
    try:
        connection.execute(text("DELETE FROM nba_box_scores"))
        trans.commit()
        print("✅ All rows deleted from nba_box_scores.")
    except Exception as e:
        trans.rollback()
        print(f"Error deleting rows: {e}")
        exit()

# ------------------------------
# Step 2: Loop through all CSV files for box scores and load data
# ------------------------------
data_folder = "./"
csv_pattern = os.path.join(data_folder, "nba_box_scores*.csv")
csv_files = glob.glob(csv_pattern)

if not csv_files:
    print("No CSV files found for box scores.")
    exit()

success_files = []
error_files = []

for csv_file in csv_files:
    print(f"Processing file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        # Convert "minutes" column using the conversion function, if present
        if "minutes" in df.columns:
            df["minutes"] = df["minutes"].apply(convert_minutes)
        # Append the DataFrame into the nba_box_scores table
        df.to_sql("nba_box_scores", engine, if_exists="append", index=False)
        success_files.append(csv_file)
        print(f"✅ Loaded {len(df)} rows from {csv_file} into nba_box_scores.")
    except Exception as e:
        error_files.append((csv_file, str(e)))
        print(f"❌ Error processing {csv_file}: {e}")

# ------------------------------
# Step 3: Print summary of results
# ------------------------------
print("\nSummary of CSV Loading:")
if success_files:
    print("Successfully loaded files:")
    for f in success_files:
        print(f" - {f}")
else:
    print("No files loaded successfully.")

if error_files:
    print("\nFiles with errors:")
    for f, err in error_files:
        print(f" - {f}: {err}")
else:
    print("\nNo errors encountered.")

print("\n✅ All CSV file processing completed!")