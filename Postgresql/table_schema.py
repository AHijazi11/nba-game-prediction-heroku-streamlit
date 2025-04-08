# script_to_get_schema.py
import os
import psycopg2

def get_table_info(table_name):
    """Returns a list of (column_name, data_type) tuples for a given table."""
    query = f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = '{table_name}';
    """
    return query

def main():
    db_url = os.getenv("DATABASE_URL")  # or replace with your actual connection string
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set.")

    # Connect to your Postgres database
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    tables = [
        "model_performance",
        "nba_game_advanced_stats",
        "nba_box_scores",
        "nba_active_players",
        "nba_player_injuries",
        "nba_games",
        "nba_predictions",
        "team_features"
    ]

    for table in tables:
        print(f"\n=== {table} ===")
        query = get_table_info(table)
        cursor.execute(query)
        results = cursor.fetchall()
        for col_name, col_type in results:
            print(f"{col_name} | {col_type}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()