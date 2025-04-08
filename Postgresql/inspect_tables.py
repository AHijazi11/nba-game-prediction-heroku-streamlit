import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# Fix URL if needed
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=True)

# List all tables in the database
inspector = inspect(engine)
tables = inspector.get_table_names()
print("Tables in the database:")
for table in tables:
    print(f" - {table}")

# For each table, query the first 5 rows and print the output
for table in tables:
    print(f"\nData in table '{table}':")
    query = f"SELECT * FROM {table} LIMIT 5"
    df = pd.read_sql(query, engine)
    print(df)

# Loop through each table to count columns and rows
for table in tables:
    # Get list of columns
    columns = inspector.get_columns(table)
    num_columns = len(columns)
    
    # Execute a query to count rows in the table
    query = f"SELECT COUNT(*) AS row_count FROM {table}"
    result = pd.read_sql(query, engine)
    row_count = result.iloc[0]['row_count']
    
    print(f"Table: {table} | Columns: {num_columns} | Rows: {row_count}")    