import os
import psycopg2

def create_team_features_table():
    """
    Creates a table named 'team_features' in a Postgres database.
    Adjust data types as necessary:
      - game_id: BIGINT vs INT
      - date: DATE vs TIMESTAMP
      - is_home_game, game_postseason, game_won: BOOLEAN (or INT)
      - ELO & time_in_season: DOUBLE PRECISION or NUMERIC
    """

    # Get the Postgres connection URL from environment variable
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set.")

    # Create SQL to create the table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS team_features (
        game_id BIGINT,
        
        date DATE,

        season INT,

        team_id BIGINT,

        team_name VARCHAR(100),

        is_home_game BOOLEAN,
        game_postseason INT,
        game_won INT,

        win_streak INT,

        team_elo DOUBLE PRECISION,
        opponent_elo DOUBLE PRECISION,
        season_elo DOUBLE PRECISION,
        opponent_season_elo DOUBLE PRECISION,
        time_in_season INT
    );
    """

    # Connect to the database and create the table
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            conn.commit()
            print("Table 'team_features' created successfully (if it didn't already exist).")
    finally:
        conn.close()

if __name__ == "__main__":
    create_team_features_table()